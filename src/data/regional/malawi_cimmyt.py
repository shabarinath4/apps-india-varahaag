import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

import numpy as np
import logging
import pandas as pd
import os
import rasterio as rio
import json
import sys
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import joblib
import yaml
import faiss
from tabulate import tabulate
from scipy.stats import mode
from pathlib import Path
from typing import Optional
from sklearn.preprocessing import LabelEncoder        
import rasterio as rio
import numpy as np
import pickle
from shapely.geometry import Point, Polygon
from scipy.spatial import ConvexHull

from src.data.base.data_loader import DataLoader
from src.data.base.preprocess import Preprocess
from src.models.graph.intercrops import MakeGraph
from src.models.fsl.inference import InferSC
from src.models.cnn.inference import InferCNN
from src.models.rfc.inference import InferRFC
from src.models.fsl.triplet_mining import TripletMiner
from scipy.spatial import ConvexHull
from shapely.geometry import Point, Polygon

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

base_path = Path(__file__).resolve().parents[2]
sys.path.append('C:\\Users\\shaba\\Desktop\\1.1 Crop ID - Domain Adaptation\\algos-crop_identification-deCIFR')

class MalawiCIMMYT:
    def __init__(
        self,
        base_path: Optional[str] = None,
        path_id: Optional[str] = None,
        results_dir: Optional[str] = None,
        model_dir: Optional[str] = None,
        mode: Optional[str] = None,
        plot_level_train: Optional[bool] = None,
        mask_value: Optional[float] = None
    ) -> None:
        """
        Initialize CIMMYTMalawi class with dataset configuration.

        Args:
            base_path (Optional[Path]): Base project directory path.
            path_id (Optional[str]): Identifier for dataset path.
            results_dir (Optional[Path]): Directory to save results.
            model_dir (Optional[Path]): Directory containing models.
            mode (Optional[str]): Mode of operation (e.g., "train", "test").
            plot_level_train (Optional[bool]): Aggregate at plot level if True.
            mask_value (Optional[float]): Value for masking invalid entries.
        """
        self.base_path = base_path
        self.path_id = path_id
        self.results_dir = results_dir
        self.model_dir = model_dir
        self.mode = mode
        self.plot_level_train = plot_level_train
        self.mask_value = mask_value

        # Load YAML metadata for all datasets
        with open(self.base_path / "configs" /"malawi_cimmyt.yaml", "r") as f:
            self.dataset_meta = yaml.safe_load(f)
        
        # Extract metadata for this dataset
        self.meta = self.dataset_meta[self.path_id]
        
        # Paths
        self.REGION_PATH = self.base_path / "data" / "train_ready" / self.path_id
        self.DATA_PATH = self.REGION_PATH / self.meta["data_file"]
        self.IDS_PATH = self.REGION_PATH / self.meta["ids_file"]
        self.COORDS_PATH = self.REGION_PATH / self.meta["coords_file"]
        self.META_PATH = self.REGION_PATH / self.meta["meta_file"]
        self.CLD_MODEL_PATH = self.REGION_PATH / self.meta["cld_model_file"]
        
        # Dataset specific params
        self.YEARS = self.meta["years"]
        self.CROPS_TO_DROP = self.meta["crops_to_drop"]
        self.OUTLIER_PER_TIMESTAMP = self.meta["outlier_per_timestamp"]
        self.NORMALISE_STRATEGY = self.meta["normalise_strategy"]
        self.plot_level_train = self.meta["plot_level_train"]
        self.pre_train_test_split = self.meta["pre_train_test_split"]
        
        # Time settings
        self.TOTAL_TS = np.arange(0, self.meta["total_ts"], 1)
        # self.TOTAL_TS = list(np.arange(0, 10, 1)) + list(np.arange(15, self.meta["total_ts"], 1))
        self.TS_SLICE_WIDTH = self.meta["ts_slice_width"]
        self.TS_STEP = self.meta["ts_step"]

        # Satellite band mapping
        self.BAND_NAME_MAP = self.meta["band_name_map"]
        self.BAND_NAMES = list(self.BAND_NAME_MAP.keys())
        self.VIs = self.meta["VIs"]

        # Bands selected 
        self.BANDS_SELECTED = [v for v in self.BAND_NAME_MAP.values() if v not in ['COASTALBLUE', 'ALPHA']]

        # Compute band indices in data array
        band_names_values = list(self.BAND_NAME_MAP.values())
        self.band_indices = [band_names_values.index(b) for b in self.BANDS_SELECTED if b in band_names_values]
        
        # Vegetation indices indices (assuming VIs follow bands)
        vi_start_idx = len(band_names_values)
        self.vi_indices = list(range(vi_start_idx, vi_start_idx + len(self.VIs)))
        
        # Final features and indices
        self.FEATURES = self.BANDS_SELECTED + list(self.VIs.keys())
        self.FEATURES_IDX = self.band_indices + self.vi_indices

    def malawi_specific_grouping(self, meta, data, coords, org_ids, req_ids, idx):
        """
        Consolidate intercrop categories into defined groups.

        Args:
            meta (pd.DataFrame): Metadata with field and crop details.
            data (np.ndarray): Time-series data (samples, time, features).
            coords (np.ndarray): Coordinates for each sample.
            org_ids (list[int]): Original field IDs.
            req_ids (list[int]): Required field IDs.
            idx (np.ndarray): Sample indices.

        Returns:
            tuple: Updated metadata, labels, data, coordinates, field IDs, and indices.
        """
        
        def more_than_three_crops(crops_str):
            crops = [c.strip() for c in crops_str.split(',')]
            return len(crops) > 3
        
        def assign_group(crop_str):
            crops = set(c.strip() for c in crop_str.split(','))
            matched_groups = []
            for group_name, required_crops in priority_groups:
                if required_crops.issubset(crops):
                    matched_groups.append(group_name)
            if len(matched_groups) > 1:
                return 'Conflicted / Mixed'
            elif len(matched_groups) == 1:
                return matched_groups[0]
            if 'Maize' not in crops:
                if crops.issubset(single_crops) and len(crops) == 1:
                    return list(crops)[0]
                else:
                    return 'Other / Mixed Non-Maize'
            if crops == {'Maize'}:
                return 'Maize'
            return 'Other / Mixed'

        meta = meta[~meta['Intercrop'].apply(more_than_three_crops)].reset_index(drop=True)

        all_crops = set()
        for crops_str in meta['Intercrop']:
            crops = set(c.strip() for c in crops_str.split(','))
            all_crops.update(crops)

        single_crops = set()
        for crops_str in meta['Intercrop']:
            crops = set(c.strip() for c in crops_str.split(','))
            if len(crops) == 1:
                single_crops.update(crops)

        priority_groups = [
            ('Maize + Pigeon_Pea', {'Maize', 'Pigeon_Pea'}),
            ('Maize + Cowpea', {'Maize', 'Cowpea'}),
            ('Cowpea + Pigeon_Pea', {'Cowpea', 'Pigeon_Pea'}),
            ('Maize + Soybean', {'Maize', 'Soybean'}),
            ('Maize + Groundnut', {'Maize', 'Groundnut'})
        ]

        meta['Intercrop_Consolidated'] = meta['Intercrop'].apply(assign_group)

        # meta['Intercrop_Consolidated'] = meta['Intercrop_Consolidated'].replace(
        #     ['Maize + Cowpea', 'Maize + Soybean', 'Maize + Groundnut'], 'Maize + Legumes'
        # )

        drop_labels = ['Other / Mixed', 'Other / Mixed Non-Maize', 'Conflicted / Mixed', 'Other']
        meta = meta[~meta['Intercrop_Consolidated'].isin(drop_labels)].reset_index(drop=True)

        assert meta['Field_Id'].is_unique, "Field_Id is not unique — mapping may be inconsistent."

        if self.path_id.endswith('_ft'):
            field_id_to_group = meta.set_index('Field_Id')['Intercrop_Consolidated'].to_dict()
        else:
            field_id_to_group = meta.set_index('Field_Id')['Intercrop'].to_dict()

        meta['Field_Id_Consolidated'] = meta['Field_Id'].map(field_id_to_group)
        org_ids = np.array(org_ids, dtype=int)
        valid_indices = []
        labels_list = []

        for uid in req_ids:
            uid = int(uid)
            consolidated_label = field_id_to_group.get(uid)
            if consolidated_label is not None:
                pos = np.where(org_ids == uid)[0]
                for p in pos:
                    valid_indices.append(p)
                    labels_list.append(consolidated_label)

        valid_indices = np.array(valid_indices)
        labels_list = np.array(labels_list)
        sort_order = np.argsort(org_ids[valid_indices])
        valid_indices = valid_indices[sort_order]
        labels_list = labels_list[sort_order]

        ids = org_ids[valid_indices]
        data = data[valid_indices]
        coords = coords[valid_indices]
        idx = idx[valid_indices]
        labels = labels_list

        return meta, labels, data, coords, ids, idx

    def harmonise_CIMMYTMalawi_planet(self, data: np.ndarray, BAND_NAMES: list[str], BAND_NAME_MAP: dict[str, str]) -> tuple[np.ndarray, list[str], dict[str, str]]:
        """
        Harmonise spectral features and compute vegetation indices.

        Args:
            data (np.ndarray): Input array of shape (samples, time, features).
            BAND_NAMES (list[str]): List of band names.
            BAND_NAME_MAP (dict[str, str]): Mapping from keys to standard band names.

        Returns:
            tuple: Processed data, updated feature names, and updated band mapping.
        """
        logger.info('Harmonising features and adding vegetation indices')
        dataloader = DataLoader(list(BAND_NAME_MAP.keys()), self.mask_value)
        data, features, band_name_map = dataloader.make_VIs(data, self.VIs, BAND_NAMES, BAND_NAME_MAP)
        return data, features, band_name_map

    def apply_field_median(self, data: np.ndarray, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute median across samples grouped by unique field IDs.

        Args:
            data (np.ndarray): Input array of shape (samples, time, bands).
            ids (np.ndarray): Field IDs for each sample.

        Returns:
            tuple: Median aggregated data and unique field IDs.
        """
        unique_ids = np.unique(ids)
        median_data = np.zeros((len(unique_ids), data.shape[1], data.shape[2]), dtype=data.dtype)

        for i, uid in enumerate(unique_ids):
            sample_mask = np.where(ids == uid)[0]
            group_samples = data[sample_mask]
            median_data[i] = np.nanmedian(group_samples, axis=0)

        return median_data, unique_ids

    def process_field_pixels(self, field_pixels, remove_low_std=False, ndvi_band_index=-2, std_percentile=95):

        if remove_low_std:
            ndvi_std_per_pixel = np.nanstd(field_pixels[:, :, ndvi_band_index], axis=1)
            threshold = np.percentile(ndvi_std_per_pixel, std_percentile)
            keep_mask = ndvi_std_per_pixel > threshold
            if np.sum(keep_mask) == 0:
                keep_mask = np.ones_like(ndvi_std_per_pixel, dtype=bool)
            field_pixels = field_pixels[keep_mask]

        return np.nanmedian(field_pixels, axis=0)

    def process_fields(
            self,
            ids,
            coords,
            data,
            labels,
            idx,
            apply_area_filter=False,
            area_threshold=500,        # m² = 0.1 ha
            apply_erosion=False,
            erosion_distance=5,        # buffer distance in same units as coords
            apply_low_std=False,
            percentile_threshold=25,   # drop bottom percentile by std
            representative="centroid"  # "centroid" or "median"
        ):
            """
            Aggregate pixel data at field level with optional filters.

            Args:
                ids (np.ndarray): Field IDs for samples.
                coords (np.ndarray): Coordinates for samples.
                data (np.ndarray): Time-series data (samples, time, features).
                labels (np.ndarray): Labels for samples.
                idx (np.ndarray): Indices for samples.
                apply_area_filter (bool): Drop polygons below area threshold.
                area_threshold (float): Minimum polygon area.
                apply_erosion (bool): Apply polygon erosion.
                erosion_distance (float): Distance for erosion.
                apply_low_std (bool): Drop low-variance pixels.
                percentile_threshold (float): Percentile for variance filtering.
                representative (str): Aggregation method ("centroid" or "median").

            Returns:
                tuple: Aggregated data, majority labels, indices, coordinates, and field IDs.
            """

            unique_ids = np.unique(ids)
            data_processed, labels_mode, idx_processed, coords_processed, ids_processed = [], [], [], [], []

            le = LabelEncoder()
            le.fit(labels)

            for uid in unique_ids:
                mask = ids == uid
                field_coords = coords[mask]
                field_pixels = data[mask]
                field_labels = labels[mask]
                field_ids = idx[mask]

                if len(field_coords) < 3:
                    continue

                # Convex hull polygon
                hull = ConvexHull(field_coords)
                hull_points = field_coords[hull.vertices]
                hull_polygon = Polygon(hull_points)

                # Polygon area filter 
                if apply_area_filter and hull_polygon.area < area_threshold:
                    continue

                # Optional erosion 
                polygon = hull_polygon
                if apply_erosion and erosion_distance > 0:
                    polygon = hull_polygon.buffer(-erosion_distance)
                    if polygon.is_empty:
                        continue

                # Centroid
                centroid = np.array(polygon.centroid.coords[0])

                # Remove low-std pixels if requested 
                keep_mask = np.ones(len(field_pixels), dtype=bool)
                if apply_low_std:
                    band6_std = np.nanstd(field_pixels[:, :, -1], axis=1)
                    threshold = np.percentile(band6_std, percentile_threshold)
                    keep_mask = band6_std > threshold
                    if np.sum(keep_mask) == 0:
                        keep_mask[:] = True

                valid_coords = field_coords[keep_mask]
                valid_pixels = field_pixels[keep_mask]
                valid_ids = field_ids[keep_mask]

                if len(valid_coords) == 0:
                    continue

                # Representative pixel 
                if representative == "centroid":
                    distances = np.linalg.norm(valid_coords - centroid, axis=1)
                    closest_idx = np.argmin(distances)
                    rep_pixel = valid_pixels[closest_idx]
                    rep_coord = valid_coords[closest_idx]
                    rep_idx = valid_ids[closest_idx]

                elif representative == "median":
                    rep_pixel = np.nanmedian(valid_pixels, axis=0)
                    distances = np.linalg.norm(valid_coords - centroid, axis=1)
                    closest_idx = np.argmin(distances)
                    rep_coord = valid_coords[closest_idx]
                    rep_idx = valid_ids[closest_idx]

                else:
                    raise ValueError("representative must be 'centroid' or 'median'")

                # Majority label
                field_labels_encoded = le.transform(field_labels)
                counts = np.bincount(field_labels_encoded)
                majority_label_encoded = np.argmax(counts)
                majority_label = le.inverse_transform([majority_label_encoded])[0]

                # Save results
                data_processed.append(rep_pixel)
                labels_mode.append(majority_label)
                coords_processed.append(rep_coord)
                idx_processed.append(rep_idx)
                ids_processed.append(uid)

            return (
                np.array(data_processed),
                np.array(labels_mode),
                np.array(idx_processed),
                np.array(coords_processed),
                np.array(ids_processed),
            )
    
    def load_CIMMYTMalawi_planet(self):
        """
        Load, clean, and preprocess CIMMYT Malawi dataset.

        Returns:
            tuple: Processed data, labels, field IDs, indices, feature names,
                   feature indices, outlier dictionary, and normalization dictionary.
        """
        use_knw = True

        import numpy as np
        data = np.load(self.DATA_PATH)
        ids = np.load(self.IDS_PATH)

        try:
            org_ids = [int(i) for i in ids]
        except:
            org_ids = [int(i.split('_')[-1]) for i in ids]
        
        try:
            coords = np.load(self.COORDS_PATH)
        except:
            coords = ids

        meta = gpd.read_file(self.META_PATH)
        idx = np.arange(0, len(ids), 1)

        v4_ids = pd.read_csv(r"D:\Users\shaba\Desktop\1.1 Crop ID - Domain Adaptation\decifr-ag\data\train_ready\malawi_250812_v4_ft\v4_duplicate_field_ids.csv")
        meta = meta[~meta['Field_Id'].isin(list(v4_ids['Field_Id_2']))]
        
        # For differently named inputs
        try:
            meta['Source'] = meta['Provider']
        except:
            pass
        
        try:
            meta['Intercrop'] = meta['Crop_Type']
        except:
            pass
        
        # Filters for source and year
        meta = meta.drop_duplicates(subset=['Field_Id'])
        counts = meta['Intercrop'].value_counts()
        sample_counts = (counts * 0.3).astype(int) 

        # Drop crops with less that 10 fields
        crop_counts = meta["Intercrop"].value_counts()
        valid_crops = crop_counts[crop_counts >= 5].index
        meta = meta[meta["Intercrop"].isin(valid_crops)]
        logger.info(f"****REMOVING**** crops with less than 10 samples")
        
        # Map Cloudy pixels
        CLOUD_CLASSES = [0, 3, 5, 9]
        mask_value = self.mask_value
        rfc = joblib.load(self.CLD_MODEL_PATH)
        cld_ts = [rfc.predict(data[:, i, :-1]) for i in range(data.shape[1])]
        cld = np.stack(cld_ts, axis=1)
        cloud_mask = np.isin(cld, CLOUD_CLASSES)
        cloud_mask_expanded = np.repeat(cloud_mask[:, :, None], data.shape[2], axis=2)
        data[cloud_mask_expanded] = mask_value
        logger.info(f"Cloud Mask done {data.shape}")
        logger.info(f"Loaded train data {data.shape}")
        logger.info(f"Loaded ids data {ids[0]}")
        logger.info(f"Value counts : Filtered by Source : {meta['Intercrop'].value_counts()}")

        # For differently named inputs
        try:
            meta['Field_Id'] = meta['Field_Id'].apply(lambda x:int(x))
        except:
            meta['Field_Id'] = meta['Field_Id'].apply(lambda x:int(x.split('_')[-1]))

        req_ids = meta['Field_Id'] 
        req_ids = [int(r) for r in req_ids]
        idx = np.arange(0, len(ids), 1)

        # Grouping Intercrop species into similar groups
        meta, labels, data, coords, ids, idx = self.malawi_specific_grouping(meta, data, coords, org_ids, req_ids, idx)
        # Field level preprocessing 
        data, labels, idx, coords, ids = self.process_fields(
            ids, coords, data, labels, idx,
            apply_area_filter=False,
            apply_erosion=False, erosion_distance=5,
            apply_low_std=False, percentile_threshold=25,
            representative="median"
        )

        # Select timestamps
        data = data[:, self.TOTAL_TS, :]
        logger.info("TS selection done")

        # Drop unwanted crops
        mask = ~np.isin(labels, self.CROPS_TO_DROP)
        data = data[mask]
        labels = labels[mask]
        coords = coords[mask]
        ids = ids[mask]
        idx = idx[mask]
        logger.info(f"Dropped crops {self.CROPS_TO_DROP}")

        # Knowledge Input
        make_graph_obj = MakeGraph(labels)
        knw_emb, idx_to_node = make_graph_obj.intercrop_graph()
        with open(os.path.join(self.model_dir, 'le_knw.pickle'), 'wb') as f:
            pickle.dump(idx_to_node, f)
        np.save(os.path.join(self.model_dir, 'knw_embeddings.npy'), knw_emb)

        # Mask values
        data[data == 9999.0] = self.mask_value
        logger.info("Masked no data values")
        logger.info(f"Loaded train data {data.shape}")
        logger.info(f"Loaded ids data {ids.shape}")

        # Harmonise and add vegetation indices
        data, _, _ = self.harmonise_CIMMYTMalawi_planet(data, self.BAND_NAMES, self.BAND_NAME_MAP)
        logger.info("Added vegetation indices")
        
        # Standard preprocessing pipeline
        preprocess = Preprocess(data, {}, {}, self.mode, self.model_dir)
        data, outlier_dict, normalise_dict = preprocess.prep(self.OUTLIER_PER_TIMESTAMP, self.mask_value)
        logger.info("Preprocessing complete")

        # Select features
        data = data[:, :, self.FEATURES_IDX]
        features = self.FEATURES
        logger.info("Band and VI selection done")
        logger.info(self.FEATURES)
        logger.info(self.FEATURES_IDX)

        # Define domains
        meta_domain = meta[meta['Field_Id'].isin(ids)]
        miner = TripletMiner()
        meta_domain, encoders = miner.encode_domains(meta_domain, year_col="Year", source_col="Src_File")
        domains = np.array(meta_domain["Domain_enc"].values)

        # variants = miner.generate_domain_variants_compact(meta_domain, crop_col="Intercrop_Consolidated", year_col="Year_enc", 
        #                                 source_col="Source_enc", n_variants=10, 
        #                                 drop_fraction=0.3, min_samples=5, random_state=42, meta_path=self.META_PATH)
        return data, labels, ids, idx, features, self.FEATURES_IDX, outlier_dict, normalise_dict, domains, knw_emb, self.pre_train_test_split
    
    def deploy_MalawiCIMMYT_planet_fsl(
            self, 
            MODEL_DIR,
            in_path,
            out_dir,
            model_clf_strat,
        ):
    
        no_data = 255
        cloud_classes = [0, 3, 5, 9]
        data = np.load(Path(in_path))
        metadata = np.load(Path(in_path).parent / "metadata.pickle.npy", allow_pickle=True).item()

        cld_msk = np.load(Path(in_path).parent / "cloudmask.npy")
        valid_mask = ~(np.all(data[:, :, :] == 0, axis=-1))  
        logger.info(f"Masking no data")

        # Cloud Masking
        cloud_mask = np.isin(cld_msk, cloud_classes)
        cloud_mask_expanded = np.repeat(cloud_mask[..., np.newaxis], data.shape[-1], axis=-1)
        data[cloud_mask_expanded] = self.mask_value
        logger.info(f"Cloud-masked")

        # Load dict  
        with open(Path(MODEL_DIR / "normalise_dict.json"), 'r') as f_in:
            normalise_dict = json.load(f_in)
        with open(Path(MODEL_DIR / "outlier_dict.json"), 'r') as f_in:
            outlier_dict = json.load(f_in)

        original_shape = data.shape
        data = data.transpose(1, 2, 0, 3).reshape(-1, original_shape[0], original_shape[3])
        
        # Select timestamps
        data = data[:, self.TOTAL_TS, :]

        # Harmonise and add vegetation indices
        t0 = time.time()
        data, _, _ = self.harmonise_CIMMYTMalawi_planet(data, self.BAND_NAMES, self.BAND_NAME_MAP)
        t1 = time.time()
        logger.info("Added vegetation indices")
        logger.info(f"Harmonisation took {t1 - t0:.2f} seconds")
        
        # Standard preprocessing pipeline
        t0 = time.time()
        preprocess = Preprocess(data, outlier_dict, normalise_dict, self.mode, self.model_dir)
        data, outlier_dict, normalise_dict = preprocess.prep(self.OUTLIER_PER_TIMESTAMP, self.mask_value)
        t1 = time.time()
        logger.info("Preprocessing complete")
        logger.info(f"Harmonisation took {t1 - t0:.2f} seconds")
        
        t0 = time.time()
        # logger.info(f"Resampling")
        # data = preprocess.resample_time_series(data, 10, 20)
        # data = preprocess.resample_time_series(data, 10, 20)
        # t1 = time.time()
        # logger.info(f"Resampling took {t1 - t0:.2f} seconds")

        # Select features
        data = np.expand_dims(data, axis=-1)
        data = data[:, :, self.FEATURES_IDX]

        def flatten(x: np.ndarray) -> np.ndarray:
            return x.reshape(x.shape[0], -1) if len(x.shape) > 2 else x

        # INFERENCE
        for model_type, clf_strat in model_clf_strat.items():
            if model_type == 'rfc_raw':
                infer_obj = InferRFC()
                out_raster = infer_obj.inference(os.path.join(MODEL_DIR, model_type), flatten(data), original_shape, model_type)
                out_path = Path(str(out_dir).replace(f"{self.path_id}_domain_aware", f"{self.path_id}_{model_type}"))
                os.makedirs(out_path.parent, exist_ok=True)
                infer_obj.write(out_raster, metadata, valid_mask, no_data, out_path)
            if model_type == 'cnn_raw':
                infer_obj = InferCNN()
                out_raster = infer_obj.inference(Path(os.path.join(MODEL_DIR, model_type)), np.expand_dims(flatten(data), axis=-1), original_shape, clf_strat)
                out_path = Path(str(out_dir).replace(f"{self.path_id}_domain_aware", f"{self.path_id}_{model_type}"))
                os.makedirs(out_path.parent, exist_ok=True)
                infer_obj.write(out_raster, metadata, valid_mask, no_data, out_path)
            if model_type == 'vanilla':
                infer_obj = InferSC()
                out_raster = infer_obj.inference(Path(os.path.join(MODEL_DIR, model_type)), data, original_shape, clf_strat)
                out_path = Path(str(out_dir).replace(f"{self.path_id}_domain_aware", f"{self.path_id}_{model_type}"))
                os.makedirs(out_path.parent, exist_ok=True)
                infer_obj.write(out_raster, metadata, valid_mask, no_data, out_path)
            if model_type == 'class_balanced':
                infer_obj = InferSC()
                out_raster = infer_obj.inference(Path(os.path.join(MODEL_DIR, model_type)), data, original_shape, clf_strat)
                out_path = Path(str(out_dir).replace(f"{self.path_id}_domain_aware", f"{self.path_id}_{model_type}"))
                os.makedirs(out_path.parent, exist_ok=True)
                infer_obj.write(out_raster, metadata, valid_mask, no_data, out_path)
            if model_type == 'domain_aware':
                infer_obj = InferSC()
                out_raster = infer_obj.inference(Path(os.path.join(MODEL_DIR, model_type)), data, original_shape, clf_strat)
                out_path = Path(str(out_dir))
                os.makedirs(out_path.parent, exist_ok=True)
                infer_obj.write(out_raster, metadata, valid_mask, no_data, out_path)
                    
        








       



        