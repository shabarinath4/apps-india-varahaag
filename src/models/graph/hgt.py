import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HGTConv
from sklearn.metrics import accuracy_score

# https://pytorch-geometric.readthedocs.io/en/2.6.0/tutorial/heterogeneous.html

class HGTCropTrainer:
    def __init__(self, X_train, y_train, X_test, y_test,
                 num_epochs=50, lr=1e-3, hidden_dim=64, use_cuda=True):

        self.device = torch.device('cuda' if torch.cuda.is_available() and use_cuda else 'cpu')
        # flatten input: [samples, time*features]
        self.X_train = torch.tensor(X_train.reshape(X_train.shape[0], -1), dtype=torch.float32).to(self.device)
        self.X_test = torch.tensor(X_test.reshape(X_test.shape[0], -1), dtype=torch.float32).to(self.device)
        self.y_train_raw = y_train
        self.y_test_raw = y_test
        self.num_epochs = num_epochs
        self.lr = lr
        self.hidden_dim = hidden_dim

        self.graph = None
        self.model = None
        self.class_to_idx = None
        self.idx_to_class = None
        self.crop_to_idx = None

    def build_graph(self, use_intercrop_relations=True):
        unique_classes = sorted(list(set(self.y_train_raw)))
        self.class_to_idx = {cls: i for i, cls in enumerate(unique_classes)}
        self.idx_to_class = {i: cls for cls, i in self.class_to_idx.items()}
        num_classes = len(unique_classes)

        self.y_train = torch.tensor([self.class_to_idx[y] for y in self.y_train_raw],
                                    dtype=torch.long).to(self.device)
        self.y_test = torch.tensor([self.class_to_idx.get(y, 0) for y in self.y_test_raw],
                                   dtype=torch.long).to(self.device)

        num_samples, num_features = self.X_train.shape

        data = HeteroData()
        # sample nodes
        data['sample'].x = self.X_train

        # class nodes: same feature dimension as samples
        data['class'].x = torch.rand(num_classes, num_features).to(self.device)

        # edges sample -> class
        src = torch.arange(num_samples, dtype=torch.long)
        dst = self.y_train
        data['sample', 'belongs_to', 'class'].edge_index = torch.stack([src, dst], dim=0)

        if use_intercrop_relations:
            # individual crops
            individual_crops = set()
            for cls in unique_classes:
                crops_in_label = cls.split('+')
                individual_crops.update([c.strip() for c in crops_in_label])
            individual_crops = sorted(list(individual_crops))
            self.crop_to_idx = {c: i for i, c in enumerate(individual_crops)}
            num_crops = len(individual_crops)

            # crop nodes
            data['crop'].x = torch.rand(num_crops, num_features).to(self.device)

            # edges crop <-> class
            crop_src, class_dst = [], []
            for c_idx, crop in enumerate(individual_crops):
                for cls_idx, cls in enumerate(unique_classes):
                    if crop in cls:
                        crop_src.append(c_idx)
                        class_dst.append(cls_idx)
            if len(crop_src) > 0:
                edge_index = torch.tensor([crop_src, class_dst], dtype=torch.long).to(self.device)
                data['crop', 'is_in', 'class'].edge_index = edge_index
                data['class', 'has_crop', 'crop'].edge_index = edge_index.flip(0)

        self.graph = data
        return data

    class HGTClassifier(nn.Module):
        def __init__(self, in_dims, hidden_dim, out_dim, metadata, num_heads=2):
            super().__init__()
            # HGTConv supports dict of input dims
            self.conv1 = HGTConv(in_channels=in_dims, out_channels=hidden_dim,
                                 metadata=metadata, heads=num_heads)
            self.conv2 = HGTConv(in_channels={k: hidden_dim for k in in_dims},
                                 out_channels=out_dim, metadata=metadata, heads=num_heads)

        def forward(self, data):
            x_dict = self.conv1(data.x_dict, data.edge_index_dict)
            x_dict = {k: F.relu(v) for k, v in x_dict.items()}
            x_dict = self.conv2(x_dict, data.edge_index_dict)
            return x_dict

    def train_model(self):
        data = self.graph.to(self.device)

        # build in_dims dict
        in_dims = {k: v.shape[1] for k, v in data.x_dict.items()}

        self.model = self.HGTClassifier(
            in_dims=in_dims,
            hidden_dim=self.hidden_dim,
            out_dim=len(self.class_to_idx),
            metadata=data.metadata()
        ).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()
        self.model.train()

        for epoch in range(self.num_epochs):
            optimizer.zero_grad()
            out = self.model(data)
            logits = out['sample']
            loss = criterion(logits, self.y_train)
            loss.backward()
            optimizer.step()

            if (epoch+1) % 10 == 0 or epoch == 0:
                pred = logits.argmax(dim=1)
                acc = (pred == self.y_train).float().mean().item()
                print(f'Epoch {epoch+1}/{self.num_epochs} | Loss: {loss.item():.4f} | Train Acc: {acc:.4f}')

    def evaluate(self):
        self.model.eval()
        with torch.no_grad():
            data = self.graph.to(self.device)
            out = self.model(data)
            logits = out['sample']
            pred = logits.argmax(dim=1)
            train_acc = accuracy_score(self.y_train.cpu().numpy(), pred.cpu().numpy())
            print(f"Train Accuracy: {train_acc:.4f}")

            # Test graph
            num_samples_test, num_features = self.X_test.shape
            data_test = HeteroData()
            data_test['sample'].x = self.X_test
            data_test['class'].x = self.graph['class'].x
            data_test['crop'].x = self.graph['crop'].x

            # dummy edges sample -> class
            src = torch.arange(num_samples_test, dtype=torch.long)
            dst = torch.zeros(num_samples_test, dtype=torch.long)
            data_test['sample', 'belongs_to', 'class'].edge_index = torch.stack([src, dst], dim=0)

            # reuse crop-class edges
            if 'crop' in self.graph and ('crop', 'is_in', 'class') in self.graph.edge_index_dict:
                data_test['crop', 'is_in', 'class'].edge_index = self.graph['crop', 'is_in', 'class'].edge_index
                data_test['class', 'has_crop', 'crop'].edge_index = self.graph['class', 'has_crop', 'crop'].edge_index

            logits_test = self.model(data_test.to(self.device))['sample']
            pred_test = logits_test.argmax(dim=1)
            test_acc = accuracy_score(self.y_test.cpu().numpy(), pred_test.cpu().numpy())
            print(f"Test Accuracy: {test_acc:.4f}")

        return pred_test.cpu().numpy()
