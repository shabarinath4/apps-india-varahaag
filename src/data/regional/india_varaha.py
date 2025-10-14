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

from src.data.base.data_loader import DataLoader
from src.data.base.preprocess import Preprocess
from src.models.fsl.inference import InferSC
from src.models.cnn.inference import InferCNN
from src.models.rfc.inference import InferRFC
from src.data.base.data_loader import DataLoader
from src.data.base.preprocess import Preprocess


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IndiaVaraha:
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
        with open(self.base_path / "configs" / "india_varaha.yaml", "r") as f:
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
        self.pre_train_test_split = self.meta["pre_train_test_split"]
        
        # Band settings
        self.remove_bands = self.meta["remove_bands"]

        # Time settings
        self.TOTAL_TS = np.arange(0, self.meta["total_ts"], 1)
        self.TS_SLICE_IDX = self.meta["ts_slice_idx"]
        self.resample_ts = self.meta["resample_ts"]
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
    
    def find_csv_files(
            self,
            root_dir
    ):
            files = []

            for dirpath, dirnames, filenames in os.walk(root_dir):
                for file in filenames:
                    if file.endswith('.csv'):
                        full_path = os.path.join(dirpath, file)
                        files.append(full_path)

            return files
        
    def read_and_standardize_csv(
            self, 
            file_list
    ):
        dfs = []
        replacement = 'crop_name'
        for f in file_list:
            df = pd.read_csv(f)
            df.columns = [
                col if not any(term in col for term in self.label_columns)
                else next(col.replace(term, replacement) for term in self.label_columns if term in col)
                for col in df.columns
            ]
            if 'Practice' in df.columns and 'crop_name' in df.columns:
                df['crop_name'] = df.apply(
                    lambda row: f"{row['crop_name']}_({row['Practice']})" if pd.notna(row['Practice']) else row['crop_name'],
                    axis=1
                )
            dfs.append(df)
        return pd.concat(dfs, ignore_index=True)
    
    def load_varaha_india_S1_S2(self):

        UID = self.meta["UID"]
        min_count = self.meta["min_sample_per_class"]

        # read all s1 s2 files and standardise columns
        files = self.find_csv_files(self.REGION_PATH)
        merged_df = self.read_and_standardize_csv(files)


        label_column = 'crop_name'
        merged_df = merged_df.dropna(subset=[label_column])
        merged_df = merged_df.drop_duplicates()
        merged_df[label_column] = merged_df[label_column].str.lower().str.replace(' ', '_')

        if self.group_crops:
            merged_df[label_column] = merged_df[label_column].replace('paddy', 'paddy_(transplanting)')

            merged_df[label_column] = merged_df[label_column].apply(
                lambda x: x if x in self.crops_to_keep else 'others'
            )


        # Filter low count pixels
        label_counts = merged_df[label_column].value_counts()
        uid_counts = merged_df.groupby(label_column)[UID].transform('nunique')

        merged_df = merged_df[
            (merged_df[label_column].map(label_counts) > min_count)
            & (uid_counts >= 5)
        ]

        labels = merged_df[label_column].values
        pre_train_test_split = merged_df[self.pre_train_test_split].values
        ids = merged_df[UID].values
        idx = merged_df.index.values
        logger.info(f"Dropped labels with less than {min_count}")

        # convert to datacube format
        num_samples = merged_df.shape[0]
        data = np.empty((num_samples, len(self.TOTAL_TS), len(self.BANDS_SELECTED)))

        for t_idx, time in enumerate(self.TOTAL_TS):
            for b_idx, band in enumerate(self.BANDS_SELECTED):
                col = f"{time}_{band}"
                data[:, t_idx, b_idx] = merged_df[col].values

        # add VIs
        data, _, _ = self.harmonise_VarahaIndia_S1S2(data, self.BAND_NAMES, self.BAND_NAME_MAP)
        logger.info("Added vegetation indices")

        # preprocess data
        preprocess = Preprocess(data, {}, {}, self.mode, self.model_dir)
        data, outlier_dict, normalise_dict = preprocess.prep(self.OUTLIER_PER_TIMESTAMP, self.mask_value)
        logger.info("Preprocessing complete")
        
        # select final bands and timestamps
        data = data[:, :, self.FEATURES_IDX]
        data = data[:, np.arange(self.TS_SLICE_IDX[0], self.TS_SLICE_IDX[1], 1), :]
        features = self.FEATURES
        logger.info("Band and VI selection done")

        # resampling
        if self.resample_ts:
            import time
            t0 = time.time()
            logger.info(f"Resampling")
            data = preprocess.resample_time_series(data, self.ORIGINAL_INTERVAL_DAYS, self.TARGET_INTERVAL_DAYS)
            t1 = time.time()
            logger.info(f"Resampling took {t1 - t0:.2f} seconds")
        
        # train at plot level or pixel level
        if self.plot_level_train:
            unique_ids = np.unique(ids)
            data_processed = []
            labels_mode = []
            idx_processed = []
            pre_train_test_split_s = []
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
                
                field_split_flags = pre_train_test_split[mask]
                majority_flag = np.bincount(field_split_flags.astype(int)).argmax().astype(bool)
                pre_train_test_split_s.append(majority_flag)

                idx_sample = idx[np.where(mask)[0][0]]
                idx_processed.append(idx_sample)

            data = np.array(data_processed)
            labels = np.array(labels_mode)
            idx = np.array(idx_processed)
            ids = unique_ids
            logger.info("Field level median aggregation done with percentile-based low-std filtering on band 6.")
        else:
            labels = labels

        # define domains
        domains = None
        knw_emb = None

        return data, labels, ids, idx, features, self.FEATURES_IDX, outlier_dict, normalise_dict, domains, knw_emb, pre_train_test_split_s
    
    def deploy_Varaha_S1_S2_fsl(
            self,
            MODEL_DIR,
            in_path,
            out_dir,
            model_clf_strat
        ):

        UID = self.meta["UID"]
        # read all s1 s2 files and standardise columns
        files = self.find_csv_files(in_path)
        merged_df = self.read_and_standardize_csv(files)

        label_column = 'crop_name'
        merged_df = merged_df.dropna(subset=[label_column])
        merged_df = merged_df.drop_duplicates()
        merged_df[label_column] = merged_df[label_column].str.lower().str.replace(' ', '_')

        # read all s1 s2 files and standardise columns
        files = self.find_csv_files(self.REGION_PATH)
        merged_df = self.read_and_standardize_csv(files)

        num_samples = merged_df.shape[0]
        ids = merged_df[UID]

        # create datacube format
        data = np.empty((num_samples, len(self.TOTAL_TS), len(self.BANDS_SELECTED)))
        for t_idx, time in enumerate(self.TOTAL_TS):
            for b_idx, band in enumerate(self.BANDS_SELECTED):
                col = f"{time}_{band}"
                data[:, t_idx, b_idx] = merged_df[col].values

        # Load dict  
        with open(Path(self.model_dir / "normalise_dict.json"), 'r') as f_in:
            normalise_dict = json.load(f_in)
        with open(Path(self.model_dir / "outlier_dict.json"), 'r') as f_in:
            outlier_dict = json.load(f_in)

        import time
        t0 = time.time()
        original_shape = np.expand_dims(data.transpose(1, 0, 2), axis=2).shape

        # add vi's
        data, _, _ = self.harmonise_VarahaIndia_S1S2(data, self.BAND_NAMES, self.BAND_NAME_MAP)
        t1 = time.time()
        logger.info("Added vegetation indices")
        logger.info(f"Harmonisation took {t1 - t0:.2f} seconds")
        
        # preprocess
        t0 = time.time()
        preprocess = Preprocess(data, outlier_dict, normalise_dict, self.mode, self.model_dir)
        data, outlier_dict, normalise_dict = preprocess.prep(self.OUTLIER_PER_TIMESTAMP, self.mask_value)
        t1 = time.time()
        logger.info("Preprocessing complete")
        logger.info(f"Preprocessing took {t1 - t0:.2f} seconds")
        
        # select final bands and timestamps
        data = data[:, :, self.FEATURES_IDX]
        data = data[:, np.arange(self.TS_SLICE_IDX[0], self.TS_SLICE_IDX[1], 1), :]
        features = self.FEATURES
        logger.info("Band and VI selection done")

        # resampling
        if self.resample_ts:
            t0 = time.time()
            logger.info(f"Resampling")
            data = preprocess.resample_time_series(data, self.ORIGINAL_INTERVAL_DAYS, self.TARGET_INTERVAL_DAYS)
            t1 = time.time()
            logger.info(f"Resampling took {t1 - t0:.2f} seconds")
            data = np.expand_dims(data, axis=-1)

        def flatten(x: np.ndarray) -> np.ndarray:
            return x.reshape(x.shape[0], -1) if len(x.shape) > 2 else x

        def process_and_save_output(output, out_path, model_type, MODEL_DIR, ids, le_bt_required=True):
            out_df = pd.DataFrame(output.transpose(1, 0, 2)[:, :, 0])
            new_cols = ['predicted'] + list(range(len(out_df.columns) - 1))
            out_df.columns = new_cols
            if le_bt_required:
                le_bt = joblib.load(os.path.join(MODEL_DIR, model_type, 'le_bt.pkl'))
                out_df['predicted'] = out_df['predicted'].astype(int)
                out_df['predicted'] = le_bt.inverse_transform(out_df['predicted'].values)
            out_df['ids'] = ids
            out_df.to_csv(out_path / f"{self.path_id}_{model_type}.csv")


        for model_type, clf_strat in model_clf_strat.items():
            if model_type == 'rfc_raw':
                infer_obj = InferRFC()
                output = infer_obj.inference(os.path.join(MODEL_DIR, model_type), flatten(data), original_shape, model_type)
                out_path = Path(str(out_dir).replace(self.path_id, f"{self.path_id}_{model_type}"))
                process_and_save_output(output, out_path, model_type, MODEL_DIR, ids)

            if model_type == 'cnn_raw':
                infer_obj = InferCNN()
                output = infer_obj.inference(Path(os.path.join(MODEL_DIR, model_type)), np.expand_dims(flatten(data), axis=-1), original_shape, clf_strat)
                out_path = Path(str(out_dir).replace(self.path_id, f"{self.path_id}_{model_type}"))
                process_and_save_output(output, out_path, model_type, MODEL_DIR, ids)

            if model_type == 'vanilla':
                infer_obj = InferSC()
                output = infer_obj.inference(Path(os.path.join(MODEL_DIR, model_type)), data, original_shape, clf_strat)
                out_path = Path(str(out_dir).replace(self.path_id, f"{self.path_id}_{model_type}"))
                process_and_save_output(output, out_path, model_type, MODEL_DIR, ids)

            if model_type == 'class_balanced':
                infer_obj = InferSC()
                output = infer_obj.inference(Path(os.path.join(MODEL_DIR, model_type)), data, original_shape, clf_strat)
                out_path = Path(str(out_dir).replace(self.path_id, f"{self.path_id}_{model_type}"))
                process_and_save_output(output, out_path, model_type, MODEL_DIR, ids)

            if model_type == 'domain_aware':
                infer_obj = InferSC()
                output = infer_obj.inference(Path(os.path.join(MODEL_DIR, model_type)), data, original_shape, clf_strat)
                out_path = Path(str(out_dir).replace(self.path_id, f"{self.path_id}_{model_type}"))
                process_and_save_output(output, out_path, model_type, MODEL_DIR, ids)
