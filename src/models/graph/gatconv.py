import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import add_self_loops
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import pacmap
import colorsys
import joblib

# ------------------------ Seeds & Directories ------------------------
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

base_dir = r"D:\Users\shaba\Desktop\1.1 Crop ID - Domain Adaptation\decifr-ag\saved_models\trials\fsl\99_fsl_20251007-084443_dc983b70-fb2d-4152-9b24-435c7f081938"
folders = ["raw", "class_balanced", "vanilla", "domain_aware"]
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ------------------------ Utility Classes ------------------------
class VisualiseGraphEmbeddings:
    def __init__(self, RESULTS_DIR, node_features, trained_embeddings, labels, y_pred=None, label_mapping=None):
        self.RESULTS_DIR = RESULTS_DIR
        self.node_features = node_features
        self.trained_embeddings = trained_embeddings
        self.labels = labels
        self.y_pred = y_pred
        self.label_mapping = label_mapping if label_mapping else {i: str(i) for i in np.unique(labels)}

    def generate_color_dict(self, n, seed=42):
        np.random.seed(seed)
        hues = np.linspace(0, 1, n)
        saturations = np.linspace(0.7, 1.0, n)
        values = np.linspace(0.5, 1.0, n)
        colors = [colorsys.hsv_to_rgb(hues[i], saturations[i], values[i]) for i in range(n)]
        return {i: colors[i] for i in range(n)}

    def pacplots(self, plot_folder):
        os.makedirs(plot_folder, exist_ok=True)
        categories = np.unique(self.labels)
        color_dict = self.generate_color_dict(len(categories))
        datasets = [
            (self.node_features, self.labels, os.path.join(plot_folder, "node_features.pdf")),
            (self.trained_embeddings, self.labels, os.path.join(plot_folder, "trained_embeddings.pdf"))
        ]
        if self.y_pred is not None:
            datasets.append((self.trained_embeddings, self.y_pred, os.path.join(plot_folder, "pred_embeddings.pdf")))
        param_combinations = [(5, 0.5, 2.0), (10, 1.0, 3.0)]
        for n_neighbors, mn_ratio, fp_ratio in param_combinations:
            for data_array, labels_array, fname in datasets:
                embedding = pacmap.PaCMAP(n_components=2, n_neighbors=n_neighbors, MN_ratio=mn_ratio, FP_ratio=fp_ratio)
                X_transformed = embedding.fit_transform(data_array, init="pca")
                plt.figure(figsize=(6, 6))
                for category in categories:
                    mask = labels_array == category
                    plt.scatter(X_transformed[mask,0], X_transformed[mask,1], color=color_dict[category],
                                label=self.label_mapping[category], s=6, alpha=0.7)
                plt.title(f'PaCMAP (n={n_neighbors}, MN={mn_ratio}, FP={fp_ratio})\nTotal Nodes: {len(labels_array)}')
                plt.legend(markerscale=2)
                plt.savefig(fname, format='pdf', bbox_inches="tight")
                plt.close()

    def pacmap(self):
        self.pacplots(os.path.join(self.RESULTS_DIR, "plots"))

# ------------------------ Model ------------------------
class SampleGAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes):
        super().__init__()
        self.gat1 = GATConv(in_dim, hidden_dim, heads=2, dropout=0.2)
        self.gat2 = GATConv(hidden_dim*2, num_classes, heads=1, concat=False)
    def forward(self, x, edge_index=None):
        if edge_index is not None:
            x = F.elu(self.gat1(x, edge_index))
            x = F.dropout(x, p=0.2, training=self.training)
            x = self.gat2(x, edge_index)
        else:
            # For isolated nodes (test), apply GAT layers without aggregation
            x = F.elu(self.gat1(x, torch.empty((2,0), dtype=torch.long, device=x.device)))
            x = F.dropout(x, p=0.2, training=self.training)
            x = self.gat2(x, torch.empty((2,0), dtype=torch.long, device=x.device))
        return x

# ------------------------ Sample-Level Graph ------------------------
class MakeSampleGraph:
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels
        self.num_samples = len(labels)

    def build_graph(self):
        edges = []

        label_to_idx = {label: i for i, label in enumerate(self.labels)}

        for i, label in enumerate(self.labels):
            if '+' in label:
                crops = [c.strip() for c in label.split('+')]
                for c in crops:
                    if c in label_to_idx:
                        edges.append((label_to_idx[c], i))

        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.empty((2,0), dtype=torch.long)
        edge_index, _ = add_self_loops(edge_index, num_nodes=self.num_samples)
        x = torch.tensor(self.embeddings, dtype=torch.float32)
        return x, edge_index


# ------------------------ Helper ------------------------
def flatten(x: np.ndarray) -> np.ndarray:
    return x.reshape(x.shape[0], -1) if len(x.shape) > 2 else x

# ------------------------ Training + Inference ------------------------
for folder in folders:
    folder_dir = os.path.join(base_dir, folder)
    print(f"\nTraining on folder: {folder}")

    data_np = np.load(os.path.join(folder_dir, 'base_embeddings.npy'))
    labels_np = np.load(os.path.join(folder_dir, 'base_labels.npy'))
    test_np = np.load(os.path.join(folder_dir, 'test_embeddings.npy'))
    test_labels_np = np.load(os.path.join(folder_dir, 'y_test_bt.npy'))
    le_bt = joblib.load(os.path.join(folder_dir, "le_bt.pkl"))

    labels_str = le_bt.inverse_transform(labels_np)
    test_labels_str = le_bt.inverse_transform(test_labels_np)

    data_np = flatten(data_np)
    test_np = flatten(test_np)

    X_train, X_val, y_train, y_val = train_test_split(data_np, labels_str, test_size=0.2,
                                                      random_state=SEED, stratify=labels_str)

    # ---------------- Build train/val graph ----------------
    X_train_val = np.concatenate([X_train, X_val])
    y_train_val = np.concatenate([y_train, y_val])
    graph_builder_train = MakeSampleGraph(X_train_val, y_train_val)
    node_features, edge_index = graph_builder_train.build_graph()

    le_enc = LabelEncoder()
    y_train_val_enc = le_enc.fit_transform(y_train_val)
    train_data = Data(x=node_features, edge_index=edge_index, y=torch.tensor(y_train_val_enc, dtype=torch.long))

    # ---------------- Define train/val indices ----------------
    train_idx = torch.tensor(range(len(X_train)), dtype=torch.long)
    val_idx = torch.tensor(range(len(X_train), len(X_train_val)), dtype=torch.long)

    # ---------------- NeighborLoader ----------------
    train_loader = NeighborLoader(
        train_data,
        num_neighbors=[10,10],
        input_nodes=train_idx,
        batch_size=64,
        shuffle=True
    )
    val_loader = NeighborLoader(
        train_data,
        num_neighbors=[10,10],
        input_nodes=val_idx,
        batch_size=64,
        shuffle=False
    )

    # ---------------- Model ----------------
    model = SampleGAT(in_dim=node_features.shape[1], hidden_dim=32, num_classes=len(le_enc.classes_)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = np.inf
    patience_counter = 0
    best_model_path = os.path.join(RESULTS_DIR, f"{folder}_best_model.pt")

    # ---------------- Training Loop ----------------
    for epoch in range(1, 201):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index)
            target = batch.y[:batch.batch_size]
            loss = criterion(out[:len(target)], target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(target)
        total_loss /= len(train_idx)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index)
                target = batch.y[:batch.batch_size]
                loss = criterion(out[:len(target)], target)
                val_loss += loss.item() * len(target)
        val_loss /= len(val_idx)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1
            if patience_counter >= 20:
                print(f"Early stopping at epoch {epoch}")
                break

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Train Loss: {total_loss:.4f}, Val Loss: {val_loss:.4f}")

    # ---------------- Test inference ----------------
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    with torch.no_grad():
        test_tensor = torch.tensor(test_np, dtype=torch.float32).to(device)
        out_test = model(test_tensor, edge_index=None)  # no edges for inductive
        preds = out_test.argmax(dim=1).cpu().numpy()

    # ---------------- Visualisation ----------------
    visualiser = VisualiseGraphEmbeddings(
        RESULTS_DIR=os.path.join(RESULTS_DIR, folder),
        node_features=test_np,
        trained_embeddings=out_test.cpu().numpy(),
        labels=le_enc.transform(test_labels_str),
        y_pred=preds,
        label_mapping={i: le_enc.inverse_transform([i])[0] for i in range(len(le_enc.classes_))}
    )
    os.makedirs(os.path.join(RESULTS_DIR, folder), exist_ok=True)
    visualiser.pacmap()

    # ---------------- Classification report ----------------
    report = classification_report(le_enc.transform(test_labels_str), preds, target_names=le_enc.classes_, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(os.path.join(RESULTS_DIR, folder, "classification_report.csv"), index=True)

    # ---------------- Confusion matrix ----------------
    cm = confusion_matrix(le_enc.transform(test_labels_str), preds)
    cm_df = pd.DataFrame(cm, index=le_enc.classes_, columns=le_enc.classes_)
    cm_df.to_csv(os.path.join(RESULTS_DIR, folder, "confusion_matrix.csv"))
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le_enc.classes_, yticklabels=le_enc.classes_)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix - {folder}")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, folder, "confusion_matrix.pdf"))
    plt.close()
    print(f"Finished folder: {folder}")
