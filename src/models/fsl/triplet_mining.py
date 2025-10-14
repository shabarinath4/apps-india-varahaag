import numpy as np
import random
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import LabelEncoder
from typing import Tuple, List, Dict
import pandas as pd
import numpy as np
import math

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

    def batch_semihard_triplets(self, x: np.ndarray, embeddings: np.ndarray, y: np.ndarray,
                          margin: float = 0.2, rng_seed: int = 42):
        """
        Semi-hard mining within a batch (classic online semi-hard).
        """
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

    def batch_all_triplets(self, x: np.ndarray, embeddings: np.ndarray, y: np.ndarray, margin: float = 0.2):
        """
        Generate all valid triplets in the batch (batch-all online mining).
        """
        triplets = []
        triplet_classes = []
        batch_size = len(x)
        for anchor_idx in range(batch_size):
            anchor_embed = embeddings[anchor_idx].reshape(1, -1)
            anchor_label = y[anchor_idx]
            pos_indices = np.where(y == anchor_label)[0]
            pos_indices = pos_indices[pos_indices != anchor_idx]
            neg_indices = np.where(y != anchor_label)[0]
            if len(pos_indices) == 0 or len(neg_indices) == 0:
                continue
            for pos_idx in pos_indices:
                positive_embed = embeddings[pos_idx].reshape(1, -1)
                d_ap = np.sum((anchor_embed - positive_embed)**2)
                neg_embeds = embeddings[neg_indices]
                d_an = np.sum((anchor_embed - neg_embeds)**2, axis=1)
                semi_hard_mask = (d_an > d_ap) & (d_an < d_ap + margin)
                candidates = np.where(semi_hard_mask)[0]
                if len(candidates) > 0:
                    neg_idx = neg_indices[np.random.choice(candidates)]
                else:
                    neg_idx = neg_indices[np.argmax(d_an)]
                triplets.append([anchor_idx, pos_idx, neg_idx])
                triplet_classes.append(anchor_label)
        return np.array(triplets, dtype=np.int32), triplet_classes

    def batch_hard_triplets(self, x: np.ndarray, embeddings: np.ndarray, y: np.ndarray):
        """
        Select hardest positive and hardest negative for each anchor within the batch.
        """
        triplets = []
        triplet_classes = []
        batch_size = len(x)
        for anchor_idx in range(batch_size):
            anchor_embed = embeddings[anchor_idx].reshape(1, -1)
            anchor_label = y[anchor_idx]
            pos_indices = np.where(y == anchor_label)[0]
            pos_indices = pos_indices[pos_indices != anchor_idx]
            neg_indices = np.where(y != anchor_label)[0]
            if len(pos_indices) == 0 or len(neg_indices) == 0:
                continue
            # Hardest positive = farthest from anchor
            pos_embeds = embeddings[pos_indices]
            d_ap = np.sum((anchor_embed - pos_embeds)**2, axis=1)
            hardest_pos_idx = pos_indices[np.argmax(d_ap)]
            # Hardest negative = closest to anchor
            neg_embeds = embeddings[neg_indices]
            d_an = np.sum((anchor_embed - neg_embeds)**2, axis=1)
            hardest_neg_idx = neg_indices[np.argmin(d_an)]
            triplets.append([anchor_idx, hardest_pos_idx, hardest_neg_idx])
            triplet_classes.append(anchor_label)
        return np.array(triplets, dtype=np.int32), triplet_classes

    def batch_domainaware_triplets(self, x: np.ndarray, embeddings: np.ndarray, y: np.ndarray,
                                domain: np.ndarray, alpha: float = 0.3):
        x = np.array(x, dtype=np.float32)
        embeddings = np.array(embeddings, dtype=np.float32)
        y = np.array(y)
        domain = np.array(domain)

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
            class_domains = domain[class_indices]

            for i, anchor_idx in enumerate(class_indices):
                anchor_embed = embeddings[anchor_idx].reshape(1, -1)
                available_domains = np.unique(class_domains)

                for d in available_domains:
                    pos_candidates = class_indices[class_domains == d]
                    pos_candidates = pos_candidates[pos_candidates != anchor_idx]
                    if len(pos_candidates) == 0:
                        continue
                    pos_idx = rng.choice(pos_candidates)
                    positive_embed = embeddings[pos_idx].reshape(1, -1)
                    d_ap = np.linalg.norm(anchor_embed - positive_embed)

                    neg_indices = np.where(y != cls)[0]
                    if len(neg_indices) == 0:
                        continue
                    neg_embeds = embeddings[neg_indices]
                    d_an = np.sum((anchor_embed - neg_embeds) ** 2, axis=1)
                    semi_hard_mask = (d_an > d_ap) & (d_an < d_ap + alpha)
                    candidates = np.where(semi_hard_mask)[0]
                    if len(candidates) > 0:
                        neg_idx = neg_indices[np.random.choice(candidates)]
                    else:
                        neg_idx = neg_indices[np.argmax(d_an)]

                    triplets.append([anchor_idx, pos_idx, neg_idx])
                    triplet_classes.append(cls)

        return np.array(triplets, dtype=np.int32), triplet_classes

    def batch_domainaware_triplets_v2(self, x: np.ndarray, embeddings: np.ndarray, y: np.ndarray,
                                      domain: np.ndarray, alpha: float = 0.3):
        x = np.array(x, dtype=np.float32)
        embeddings = np.array(embeddings, dtype=np.float32)
        y = np.array(y)
        domain = np.array(domain)

        triplets = []
        triplet_classes = []
        rng = np.random.default_rng(42)
        num_classes = len(np.unique(y))
        unique_domains = np.unique(domain)

        for cls in range(num_classes):
            class_indices = np.where(y == cls)[0]
            if len(class_indices) < 2:
                continue
            for d in unique_domains:
                anchor_indices = class_indices[domain[class_indices] == d]
                if len(anchor_indices) == 0:
                    continue
                for anchor_idx in anchor_indices:
                    anchor_embed = embeddings[anchor_idx].reshape(1, -1)

                    pos_candidates = class_indices[class_indices != anchor_idx]
                    if len(pos_candidates) == 0:
                        continue
                    pos_idx = rng.choice(pos_candidates)
                    positive_embed = embeddings[pos_idx].reshape(1, -1)
                    d_ap = np.linalg.norm(anchor_embed - positive_embed)

                    neg_indices = np.where(y != cls)[0]
                    if len(neg_indices) == 0:
                        continue
                    neg_embeds = embeddings[neg_indices]
                    d_an = np.sum((anchor_embed - neg_embeds) ** 2, axis=1)
                    semi_hard_mask = (d_an > d_ap) & (d_an < d_ap + alpha)
                    candidates = np.where(semi_hard_mask)[0]
                    if len(candidates) > 0:
                        neg_idx = neg_indices[np.random.choice(candidates)]
                    else:
                        neg_idx = neg_indices[np.argmax(d_an)]

                    triplets.append([anchor_idx, pos_idx, neg_idx])
                    triplet_classes.append(cls)

        return np.array(triplets, dtype=np.int32), triplet_classes

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
    
    @staticmethod
    def generate_domain_variants_compact(df, crop_col="Intercrop_Consolidated", year_col="Year_enc", 
                                        source_col="Source_enc", n_variants=10, 
                                        drop_fraction=0.3, min_samples=5, random_state=42, meta_path=None):
        rng = np.random.default_rng(random_state)
        variants = []

        for _ in range(n_variants):
            keep_idx = []
            for crop in df[crop_col].unique():
                crop_df = df[df[crop_col] == crop]
                for year in crop_df[year_col].unique():
                    year_df = crop_df[crop_df[year_col] == year]
                    sources = list(year_df[source_col].unique())
                    if len(sources) == 1:
                        keep = sources
                    else:
                        n_drop = max(1, math.ceil(drop_fraction * len(sources)))
                        shuffled = list(rng.permutation(sources))
                        keep = [s for s in shuffled if len(year_df[year_df[source_col].isin([x for x in shuffled if x!=s])]) >= min_samples][:len(sources)-n_drop]
                    keep_idx.extend(year_df[year_df[source_col].isin(keep)].index.tolist())
            variants.append(df.loc[keep_idx].copy())

        import geopandas as gpd
        for i, variant in enumerate(variants, 1):
            field_ids = variant["Field_Id"].unique()
            meta_variant = df[df["Field_Id"].isin(field_ids)].copy()
            save_path = str(meta_path).replace(".geojson", f"_{i}.geojson")
            meta_variant.to_file(save_path, driver="GeoJSON")
            print(df['Intercrop_Consolidated'].value_counts())
            print(meta_variant['Intercrop_Consolidated'].value_counts())
            print('....')

        return variants

    