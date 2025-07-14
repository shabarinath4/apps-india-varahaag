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
import time
import joblib
import sys
import yaml
import faiss
from tabulate import tabulate
from scipy.stats import mode
from pathlib import Path
from typing import Optional
from sklearn.preprocessing import LabelEncoder        
import rasterio as rio
import os
import numpy as np
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.models import load_model
from src.data.data_loader_rise import DataLoader
from src.data.preprocess_rise import Preprocess
from src.models.fsl3 import Infer


# correct doc strings everywhere
# write a small readme
# add resample code

# set unknown class
# show support set
# differeniate ft and bt runs
# inf loop

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VarahaIndia:
    def __init__(
        self,
        base_path: Optional[str] = None,
        path_id: Optional[str] = None,
        results_dir: Optional[str] = None,
        model_dir: Optional[str] = None,
        mode: Optional[str] = None,
        plot_level_train: Optional[bool] = None,
        mask_value: Optional[float] = np.nan
    ) -> None:
        """
        Initializes the class with provided configuration parameters.

        Args:
            base_path (Optional[str]): Base directory path for data or project.
            path_id (Optional[str]): Identifier for the specific data path.
            results_dir (Optional[str]): Directory to save results.
            model_dir (Optional[str]): Directory where models are saved/loaded.
            mode (Optional[str]): Operational mode, e.g., 'train', 'test', etc.
            plot_level_train (Optional[bool]): Whether to aggregate data at field level.
            mask_value (Optional[float]): Value used to mask invalid data.
        """
        self.base_path = base_path
        self.path_id = path_id
        self.results_dir = results_dir
        self.model_dir = model_dir
        self.mode = mode
        self.plot_level_train = plot_level_train
        self.mask_value = mask_value

        # Load YAML metadata for all datasets
        with open(self.base_path / "region_config.yaml", "r") as f:
            self.dataset_meta = yaml.safe_load(f)
        
        # Extract metadata for this dataset
        self.meta = self.dataset_meta[self.path_id]
        
        # Paths
        self.REGION_PATH = self.base_path / "data" / "train_ready" / self.path_id

        # Dataset specific params
        self.label_columns = self.meta["label_columns"]
        self.CROPS_TO_DROP = self.meta["crops_to_drop"]
        self.OUTLIER_PER_TIMESTAMP = self.meta["outlier_per_timestamp"]
        self.NORMALISE_STRATEGY = self.meta["normalise_strategy"]
        self.plot_level_train = self.meta["plot_level_train"]
        self.group_crops = self.meta["group_crops"]
        self.crops_to_keep = self.meta["crops_to_keep"]
        
        # Band settings
        self.remove_bands = self.meta["remove_bands"]

        # Time settings
        self.TOTAL_TS = np.arange(0, self.meta["total_ts"], 1)
        self.TS_SLICE_IDX = self.meta["ts_slice_idx"]
        self.ORIGINAL_INTERVAL_DAYS = self.meta["ORIGINAL_INTERVAL_DAYS"]
        self.TARGET_INTERVAL_DAYS = self.meta["TARGET_INTERVAL_DAYS"]

        # Satellite band mapping
        self.BAND_NAME_MAP = self.meta["band_name_map"]
        self.BAND_NAMES = list(self.BAND_NAME_MAP.keys())
        self.VIs = self.meta["VIs"]

        # Bands selected 
        self.BANDS_SELECTED = [v for v in self.BAND_NAME_MAP.values() if v not in self.remove_bands]

        # Compute band indices in data array
        band_names_values = list(self.BAND_NAME_MAP.values())
        self.band_indices = [band_names_values.index(b) for b in self.BANDS_SELECTED if b in band_names_values]
        
        # Vegetation indices indices (assuming VIs follow bands)
        vi_start_idx = len(band_names_values)
        self.vi_indices = list(range(vi_start_idx, vi_start_idx + len(self.VIs)))
        
        # Final features and indices
        self.FEATURES = self.BANDS_SELECTED + list(self.VIs.keys())
        self.FEATURES_IDX = self.band_indices + self.vi_indices

    def harmonise_VarahaIndia_S1S2(self, data: np.ndarray, BAND_NAMES: list[str], BAND_NAME_MAP: dict[str, str]) -> tuple[np.ndarray, list[str], dict[str, str]]:
        """
        Harmonises input features and creates vegetation indices.

        Args:
            data (np.ndarray): Raw data array.
            FEATURES (list[str]): List of feature names.
            BAND_NAME_MAP (dict): Mapping from raw to standardised band names.
        Returns:
            Tuple[np.ndarray, list[str], dict[str, str]]: Processed data, updated features list, and band name map.
        """
        logger.info('Harmonising features and adding vegetation indices')
        dataloader = DataLoader(list(BAND_NAME_MAP.keys()), self.mask_value)
        data, features, band_name_map = dataloader.make_VIs(data, self.VIs, BAND_NAMES, BAND_NAME_MAP)
        return data, features, band_name_map
    
    def find_s1_s2_csv_files(
            self,
            root_dir
    ):
            s1_files = []
            s2_files = []

            for dirpath, dirnames, filenames in os.walk(root_dir):
                for file in filenames:
                    if file.endswith('.csv'):
                        full_path = os.path.join(dirpath, file)
                        if file.startswith('S1'):
                            s1_files.append(full_path)
                        elif file.startswith('S2'):
                            s2_files.append(full_path)

            return s1_files, s2_files
        
    def read_and_standardize_csv(
            self, 
            file_list
    ):
        dfs = []
        replacement = 'cropname'
        for f in file_list:
            df = pd.read_csv(f)
            df.columns = [
                col if not any(term in col for term in self.label_columns)
                else next(col.replace(term, replacement) for term in self.label_columns if term in col)
                for col in df.columns
            ]
            if 'Practice' in df.columns and 'cropname' in df.columns:
                df['cropname'] = df.apply(
                    lambda row: f"{row['cropname']}_({row['Practice']})" if pd.notna(row['Practice']) else row['cropname'],
                    axis=1
                )
            dfs.append(df)
        return pd.concat(dfs, ignore_index=True)
    
    def load_varaha_india_S1_S2(self):

        UID = self.meta["UID"]
        min_count = self.meta["min_sample_per_class"]

        # read all s1 s2 files
        s1_files, s2_files = self.find_s1_s2_csv_files(self.REGION_PATH)
        s1_df = self.read_and_standardize_csv(s1_files)
        s2_df = self.read_and_standardize_csv(s2_files)

        # drop duplicates and merge
        s1_df = s1_df.drop_duplicates(subset=UID)
        s2_df = s2_df.drop_duplicates(subset=UID)
        merged_df = pd.merge(s1_df, s2_df, on=UID)
        merged_df = merged_df.sort_values(by=UID).reset_index(drop=True)
        num_samples = merged_df.shape[0]

        label_column = 'cropname_x'
        merged_df = merged_df.dropna(subset=[label_column])
        merged_df = merged_df.drop_duplicates()
        merged_df[label_column] = merged_df[label_column].str.lower().str.replace(' ', '_')

        if self.group_crops:
            merged_df[label_column] = merged_df[label_column].apply(
                lambda x: x if x in self.crops_to_keep else 'others'
            )

        label_counts = merged_df[label_column].value_counts()
        valid_labels = label_counts[label_counts > min_count].index
        mask = merged_df[label_column].isin(valid_labels)
        merged_df = merged_df[mask]

        labels = merged_df[label_column].values
        ids = merged_df[UID].values
        idx = merged_df.index.values

        num_samples = merged_df.shape[0]
        data = np.empty((num_samples, len(self.TOTAL_TS), len(self.BANDS_SELECTED)))
        for t_idx, time in enumerate(self.TOTAL_TS):
            for b_idx, band in enumerate(self.BANDS_SELECTED):
                col = f"{time}_{band}"
                data[:, t_idx, b_idx] = merged_df[col].values

        logger.info(f"Dropped labels with less than {min_count}")

        # add VIs
        data, _, _ = self.harmonise_VarahaIndia_S1S2(data, self.BAND_NAMES, self.BAND_NAME_MAP)
        logger.info("Added vegetation indices")
        
        # preprocess data
        preprocess = Preprocess(data, labels, {}, {}, self.mode, self.model_dir)
        data, outlier_dict, normalise_dict = preprocess.prep(self.OUTLIER_PER_TIMESTAMP, self.mask_value)
        logger.info("Preprocessing complete")
        
        # resampling
        t0 = time.time()
        logger.info(f"Resampling")
        data = preprocess.resample_time_series(data, self.ORIGINAL_INTERVAL_DAYS, self.TARGET_INTERVAL_DAYS)
        t1 = time.time()
        logger.info(f"Resampling took {t1 - t0:.2f} seconds")
        data = np.expand_dims(data, axis=-1)
        
        # select final bands and timestamps
        data = data[:, :, self.FEATURES_IDX]
        data = data[:, np.arange(self.TS_SLICE_IDX[0], self.TS_SLICE_IDX[1], 1), :]
        features = self.FEATURES
        logger.info("Band and VI selection done")

        if self.plot_level_train:
            unique_ids = np.unique(ids)
            data_processed = []
            labels_mode = []
            idx_processed = []
            le = LabelEncoder()
            le.fit(labels)

            for uid in unique_ids:
                mask = (ids == uid)
                field_pixels = data[mask]
                median_sample = np.nanmedian(field_pixels, axis=0)
                data_processed.append(median_sample)

                field_labels = labels[mask]
                field_labels_encoded = le.transform(field_labels)
                counts = np.bincount(field_labels_encoded)
                majority_label_encoded = np.argmax(counts)
                majority_label = le.inverse_transform([majority_label_encoded])[0]
                labels_mode.append(majority_label)

                idx_sample = idx[np.where(mask)[0][0]]
                idx_processed.append(idx_sample)

            data = np.array(data_processed)
            labels = np.array(labels_mode)
            idx = np.array(idx_processed)
            ids = unique_ids
            logger.info("Field level median aggregation done with percentile-based low-std filtering on band 6.")
        else:
            labels = labels

        return data, labels, ids, idx, features, self.FEATURES_IDX, outlier_dict, normalise_dict
    
    def deploy_Varaha_S1_S2_fsl(
            self,
            model_type,
            inf_in,
            inf_out
        ):

        UID = self.meta["UID"]

        # read and combine all csvs
        s1_files, s2_files = self.find_s1_s2_csv_files(inf_in)
        s1_df = self.read_and_standardize_csv(s1_files)
        s2_df = self.read_and_standardize_csv(s2_files)

        # drop duplicates
        s1_df = s1_df.drop_duplicates(subset=UID)
        s2_df = s2_df.drop_duplicates(subset=UID)
        merged_df = pd.merge(s1_df, s2_df, on=UID)
        merged_df = merged_df.sort_values(by=UID).reset_index(drop=True)
        num_samples = merged_df.shape[0]
        ids = merged_df[UID]

        # create datacube format
        data = np.empty((num_samples, len(self.TOTAL_TS), len(self.BANDS_SELECTED)))
        for t_idx, time in enumerate(self.TOTAL_TS):
            for b_idx, band in enumerate(self.BANDS_SELECTED):
                col = f"{time}_{band}"
                data[:, t_idx, b_idx] = merged_df[col].values

        # load model
        emb_path = Path(self.model_dir / f"{model_type}_embeddings.npy")
        labels_path = Path(self.model_dir / f"{model_type}_labels.npy")
        model_path = Path(self.model_dir / "model.h5")
        faiss_idx_path = Path(self.model_dir / f"{model_type}_index.faiss")

        # Load dict  
        with open(Path(self.model_dir / "normalise_dict.json"), 'r') as f_in:
            normalise_dict = json.load(f_in)
        with open(Path(self.model_dir / "outlier_dict.json"), 'r') as f_in:
            outlier_dict = json.load(f_in)

        # Load inference object
        infer = Infer()
        embeddings = np.load(emb_path)
        embeddings_labels = np.load(labels_path)

        # Setup for temporal pyramid pooling layer
        class TemporalPyramidPooling(tf.keras.layers.Layer):
            def __init__(self, pyramid_levels=[1, 2], **kwargs):
                super().__init__(**kwargs)
                self.pyramid_levels = pyramid_levels

            def call(self, t):
                def pool_level(t, n):
                    time_len = tf.shape(t)[1]
                    boundaries = tf.cast(tf.linspace(0., tf.cast(time_len, tf.float32), n + 1), tf.int32)
                    pooled_slices = []
                    for i in range(n):
                        start = boundaries[i]
                        end = boundaries[i + 1]
                        slice_ = t[:, start:end, ...]
                        pooled = tf.reduce_max(slice_, axis=1)
                        pooled_slices.append(pooled)
                    return tf.concat(pooled_slices, axis=-1)

                pooled_levels = [pool_level(t, n) for n in self.pyramid_levels]
                return tf.concat(pooled_levels, axis=-1)

            def get_config(self):
                config = super().get_config()
                config.update({"pyramid_levels": self.pyramid_levels})
                return config
            
        base_network = load_model(model_path, custom_objects={'TemporalPyramidPooling': TemporalPyramidPooling})
        
        try:
            index = faiss.read_index(str(faiss_idx_path))
            labels = np.load(labels_path, allow_pickle=True)
        except Exception:
            index = None
            labels = None

        import time
        t0 = time.time()
        
        # add vi's
        data, _, _ = self.harmonise_VarahaIndia_S1S2(data, self.BAND_NAMES, self.BAND_NAME_MAP)
        t1 = time.time()
        logger.info("Added vegetation indices")
        logger.info(f"Harmonisation took {t1 - t0:.2f} seconds")
        
        # preprocess
        t0 = time.time()
        preprocess = Preprocess(data, labels, outlier_dict, normalise_dict, self.mode, self.model_dir)
        data, outlier_dict, normalise_dict = preprocess.prep(self.OUTLIER_PER_TIMESTAMP, self.mask_value)
        t1 = time.time()
        logger.info("Preprocessing complete")
        logger.info(f"Preprocessing took {t1 - t0:.2f} seconds")
        
        # resampling
        t0 = time.time()
        logger.info(f"Resampling")
        data = preprocess.resample_time_series(data, self.ORIGINAL_INTERVAL_DAYS, self.TARGET_INTERVAL_DAYS)
        t1 = time.time()
        logger.info(f"Resampling took {t1 - t0:.2f} seconds")
        data = np.expand_dims(data, axis=-1)
    
        # select bands and timestamps
        data = data[:, :, self.FEATURES_IDX]
        data = data[:, np.arange(self.TS_SLICE_IDX[0], self.TS_SLICE_IDX[1], 1), :]

        # run inference
        predictions, distances = infer.infer(base_network, embeddings, embeddings_labels, index, data)
        np.save(inf_out / "predictions.npy", predictions)
        np.save(inf_out / "distances.npy", distances)
        np.save(inf_out / "ids.npy", ids)

        t1 = time.time()
        logger.info(f"Full prediction took {t1 - t0:.2f} seconds")
        logger.info(f"Model inference done : {inf_out}")
