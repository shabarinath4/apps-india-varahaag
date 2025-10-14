
import torch
import torch.nn as nn
from torch_geometric.data import HeteroData, Data 
from torch_geometric.nn import GATConv, HGTConv
from torch_geometric.utils import to_networkx
from sklearn.manifold import TSNE
import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np
import pickle

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
rng = np.random.default_rng(seed=SEED)

class CropGAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim=64):  # out_dim fixed to 64
        super().__init__()
        self.gat1 = GATConv(in_dim, hidden_dim, heads=1)
        self.gat2 = GATConv(hidden_dim, out_dim, heads=1)  # final embedding 64

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.gat1(x, edge_index).relu()
        x = self.gat2(x, edge_index)
        return x

class HGTNet(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim=64, metadata=None, num_heads=2):
        super().__init__()
        self.conv1 = HGTConv(in_dim, hidden_dim, metadata, heads=num_heads)
        self.conv2 = HGTConv(hidden_dim, out_dim, metadata, heads=num_heads)  # final embedding 64

    def forward(self, data):
        x_dict = self.conv1(data.x_dict, data.edge_index_dict)
        x_dict = {k: v.relu() for k, v in x_dict.items()}
        x_dict = self.conv2(x_dict, data.edge_index_dict)  # final output
        return x_dict

class MakeGraph():
    def __init__(self, labels):
        self.labels = labels

    def intercrop_graph(self, use_hgt=False, node_feature_dim=None):
        full_labels = list(set(self.labels))
        individual_crops = set()

        for label in full_labels:
            individual_crops.update([c.strip() for c in label.split('+')])
        individual_crops = sorted(list(individual_crops))
        crop_to_idx = {c: i for i, c in enumerate(individual_crops)}
        nodes = sorted(set(full_labels).union(individual_crops))
        node_to_idx = {name: i for i, name in enumerate(nodes)}
        edges = []

        for label in full_labels:
            crops_in_label = [c.strip() for c in label.split('+')]
            for crop in crops_in_label:
                if crop != label:
                    edges.append((crop, label))

        num_nodes = len(nodes)
        num_crops = len(individual_crops)
        if node_feature_dim is None:
            node_feature_dim = max(64, num_crops)  # ensure at least 64-dim for graph embeddings
        node_features = np.zeros((num_nodes, node_feature_dim), dtype=np.float32)

        for i, node in enumerate(nodes):
            vec = np.zeros(num_crops, dtype=np.float32)
            for c in node.split('+'):
                if c in crop_to_idx:
                    vec[crop_to_idx[c]] = 1.0
            node_features[i, :num_crops] = vec

        node_features = torch.tensor(node_features, dtype=torch.float32)
        idx_to_node = {i: name for i, name in enumerate(nodes)}

        if use_hgt:
            data = HeteroData()
            data['crop'].x = node_features
            src = [node_to_idx[s] for s, _ in edges]
            dst = [node_to_idx[d] for _, d in edges]
            data['crop', 'is_in', 'crop'].edge_index = torch.tensor([src, dst], dtype=torch.long)
            model = HGTNet(in_dim=node_feature_dim, hidden_dim=32, out_dim=64, metadata=data.metadata())
            output_dict = model(data)
            embeddings = output_dict['crop'].detach().cpu().numpy()
        else:
            edge_index = torch.tensor([[node_to_idx[src], node_to_idx[dst]] for src, dst in edges], dtype=torch.long).t().contiguous()
            data = Data(x=node_features, edge_index=edge_index)
            model = CropGAT(in_dim=node_feature_dim, hidden_dim=32, out_dim=64)
            output = model(data)
            embeddings = output.detach().cpu().numpy()

        # ---------- Graph visualization ----------
        G = nx.DiGraph() if use_hgt else to_networkx(data, to_undirected=False)
        if not use_hgt:
            G = nx.relabel_nodes(G, {i: name for i, name in enumerate(nodes)})
        
        for n in G.nodes():
            print(f"{n!r}: in_degree = {G.in_degree(n)}")
        
        for e in G.edges():
            print(e)
            
        node_colors = ['lightgreen' if G.in_degree(n) == 0 else 'lightblue' for n in G.nodes()]
        plt.figure(figsize=(10, 8))
        pos = nx.spring_layout(G, seed=42)
        nx.draw(G, pos, with_labels=True, node_color=node_colors, edge_color='gray', node_size=1800, font_size=10, arrows=True, arrowsize=20)
        plt.title("Crop Composition Graph", fontsize=14)
        plt.show()

        # ---------- Embedding t-SNE visualization ----------
        reduced = TSNE(n_components=2, perplexity=min(5, len(embeddings)-1), random_state=42).fit_transform(embeddings)
        plt.figure(figsize=(10, 8))
        for i, label in enumerate(nodes):
            x, y = reduced[i]
            plt.scatter(x, y, marker='o', color='blue')
            plt.text(x+0.01, y+0.01, label, fontsize=9)
        plt.title("Node Embeddings Visualization")
        plt.xlabel("Dim 1")
        plt.ylabel("Dim 2")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        return embeddings, idx_to_node
    

