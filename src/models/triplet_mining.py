import numpy as np
import random
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import LabelEncoder
from typing import Tuple, List, Dict

class TripletMiner:
    def __init__(self, comb_per_sample: int = 50, alpha: float = 0.3,
                 triplet_logic_ratio: Dict[str, float] = {'semi-hard': 0.7, 'hard': 0.15, 'easy': 0.15},
                 seed: int = 42):
        self.comb_per_sample = comb_per_sample
        self.alpha = alpha
        self.triplet_logic_ratio = triplet_logic_ratio
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)

    def standard_triplets(self, x: np.ndarray, embeddings: np.ndarray, y: np.ndarray,
                                  margin: float = 0.2, rng_seed: int = 42):
        rng = np.random.default_rng(rng_seed)
        triplets = []
        triplet_classes = []
        num_classes = len(np.unique(y))
        indices = [np.where(y == i)[0] for i in range(num_classes)]
        for cls in range(num_classes):
            class_indices = indices[cls]
            if len(class_indices) < 2:
                continue
            for anchor_idx in class_indices:
                anchor_embed = embeddings[anchor_idx].reshape(1, -1)
                pos_candidates = np.delete(class_indices, np.where(class_indices == anchor_idx))
                if len(pos_candidates) == 0:
                    continue
                pos_idx = rng.choice(pos_candidates)
                positive_embed = embeddings[pos_idx].reshape(1, -1)
                d_ap = np.sum((anchor_embed - positive_embed)**2)
                neg_indices = np.where(y != cls)[0]
                neg_embeds = embeddings[neg_indices]
                d_an = np.sum((anchor_embed - neg_embeds)**2, axis=1)
                mask = (d_an > d_ap) & (d_an < d_ap + margin)
                semi_hard_negatives = np.where(mask)[0]
                if len(semi_hard_negatives) > 0:
                    neg_idx = neg_indices[rng.choice(semi_hard_negatives)]
                else:
                    neg_idx = neg_indices[np.argmax(d_an)]
                triplets.append([anchor_idx, pos_idx, neg_idx])
                triplet_classes.append(cls)
        triplets_all = np.array(triplets, dtype=np.int32)
        return triplets_all, triplet_classes

    def batch_hard_triplets(self, x: np.ndarray, embeddings: np.ndarray, y: np.ndarray,
                            samples_per_class: int = 5, margin: float = 0.5, rng_seed: int = 42):
        rng = np.random.default_rng(rng_seed)
        triplets = []
        triplet_classes = []
        classes = np.unique(y)
        batch_indices = []
        for cls in classes:
            cls_idx = np.where(y == cls)[0]
            if len(cls_idx) < 2:
                continue
            chosen = rng.choice(cls_idx, size=min(samples_per_class, len(cls_idx)),
                                replace=len(cls_idx) < samples_per_class)
            batch_indices.extend(chosen)
        batch_indices = np.array(batch_indices)
        batch_x = x[batch_indices]
        batch_embeds = embeddings[batch_indices]
        batch_y = y[batch_indices]
        for i, anchor_idx in enumerate(batch_indices):
            anchor_x = x[anchor_idx]
            anchor_embed = embeddings[anchor_idx].reshape(1, -1)
            anchor_label = y[anchor_idx]
            pos_mask = (batch_y == anchor_label) & (batch_indices != anchor_idx)
            pos_indices = batch_indices[pos_mask]
            if len(pos_indices) == 0:
                continue
            for pos_idx in pos_indices:
                positive_x = x[pos_idx]
                positive_embed = embeddings[pos_idx].reshape(1, -1)
                d_ap = pairwise_distances(anchor_embed, positive_embed)[0, 0]
                neg_mask = batch_y != anchor_label
                neg_indices = batch_indices[neg_mask]
                neg_embeds = embeddings[neg_indices]
                neg_x = x[neg_indices]
                if len(neg_indices) == 0:
                    continue
                d_an = pairwise_distances(anchor_embed, neg_embeds).flatten()
                semi_hard_mask = (d_an > d_ap) & (d_an < d_ap + margin)
                semi_hard_negatives = np.where(semi_hard_mask)[0]
                if len(semi_hard_negatives) > 0:
                    neg_idx = rng.choice(semi_hard_negatives)
                else:
                    neg_idx = np.argmax(d_an)
                negative_x = neg_x[neg_idx]
                triplets.append([anchor_x, positive_x, negative_x])
                triplet_classes.append(anchor_label)
        triplets_all = np.array(triplets, dtype=np.float32)
        return triplets_all, triplet_classes

    def domain_aware_triplets(self, x: np.ndarray, embeddings: np.ndarray, y: np.ndarray,
                              domain: np.ndarray, num_classes: int, alpha: float = 0.3):
        triplets = []
        triplet_classes = []
        rng = np.random.default_rng(42)
        num_classes = len(np.unique(y))
        indices = [np.where(y == i)[0] for i in range(num_classes)]
        for cls in range(num_classes):
            class_indices = indices[cls]
            if len(class_indices) < 2:
                continue
            class_embeds = embeddings[class_indices]
            class_x = x[class_indices]
            class_domain = domain[class_indices]
            for i, anchor_idx in enumerate(class_indices):
                anchor_embed = embeddings[anchor_idx].reshape(1, -1)
                anchor_x = x[anchor_idx]
                anchor_domain = domain[anchor_idx]
                same_domain_idx = np.where((class_domain == anchor_domain) & (np.arange(len(class_indices)) != i))[0]
                other_domain_idx = np.where(class_domain != anchor_domain)[0]
                if len(same_domain_idx) > 0:
                    pos_idx = rng.choice(same_domain_idx)
                elif len(other_domain_idx) > 0:
                    pos_idx = rng.choice(other_domain_idx)
                else:
                    continue
                positive_embed = class_embeds[pos_idx].reshape(1, -1)
                positive_x = class_x[pos_idx]
                d_ap = np.linalg.norm(anchor_embed - positive_embed)
                neg_indices = np.where(y != cls)[0]
                if len(neg_indices) == 0:
                    continue
                neg_embeds = embeddings[neg_indices]
                neg_x = x[neg_indices]
                d_an = pairwise_distances(anchor_embed, neg_embeds).flatten()
                semi_hard_mask = (d_an > d_ap) & (d_an < d_ap + alpha)
                semi_hard_negatives = np.where(semi_hard_mask)[0]
                if len(semi_hard_negatives) > 0:
                    neg_idx = rng.choice(semi_hard_negatives)
                else:
                    neg_idx = np.argmax(d_an)
                negative_x = neg_x[neg_idx]
                triplets.append([anchor_x, positive_x, negative_x])
                triplet_classes.append(cls)
        triplets_all = np.array(triplets, dtype=np.float32)
        return triplets_all, triplet_classes

    @staticmethod
    def encode_domains(df: 'pd.DataFrame', year_col: str = "Year", source_col: str = "Source"):
        df = df.copy()
        year_encoder = LabelEncoder()
        df["Year_enc"] = year_encoder.fit_transform(df[year_col].astype(str))
        source_encoder = LabelEncoder()
        df["Source_enc"] = source_encoder.fit_transform(df[source_col].astype(str))
        df["Domain"] = df[year_col].astype(str) + "_" + df[source_col].astype(str)
        domain_encoder = LabelEncoder()
        df["Domain_enc"] = domain_encoder.fit_transform(df["Domain"])
        encoders = {"year": year_encoder, "source": source_encoder, "domain": domain_encoder}
        return df, encoders
