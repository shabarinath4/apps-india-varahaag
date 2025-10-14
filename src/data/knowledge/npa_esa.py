import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

"""Perform cropland mapping inference using a local execution of presto.

Make sure you test this script on the Python version 3.9+, and have worldcereal
dependencies installed with the presto wheel file installed with it's dependencies.
"""

from pathlib import Path
import requests
import xarray as xr
from openeo_gfmap.features.feature_extractor import (
    EPSG_HARMONIZED_NAME,
    apply_feature_extractor_local,
)
from openeo_gfmap.inference.model_inference import apply_model_inference_local
from worldcereal.openeo.feature_extractor import PrestoFeatureExtractor
from worldcereal.openeo.inference import CropClassifier

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

from tensorflow.keras.models import load_model
from src.data.data_loader_rise import DataLoader
from src.data.preprocess_rise import Preprocess
from src.models.fsl3 import Infer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

base_path = Path(__file__).resolve().parents[2]
sys.path.append('C:\\Users\\shaba\\Desktop\\1.1 Crop ID - Domain Adaptation\\algos-crop_identification-deCIFR')

class ESA:
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
        self.mask_value = 9999
        self.mode = 'train'
        self.model_dir ='models'
        self.OUTLIER_PER_TIMESTAMP = False

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        # Load YAML metadata for all datasets
        with open(self.base_path / "region_config_npa.yaml", "r") as f:
            self.dataset_meta = yaml.safe_load(f)

        # Extract metadata for this dataset
        self.meta = self.dataset_meta[self.path_id]
        self.VIs = self.meta["VIs"]
        self.TS_SLICE_WIDTH = self.meta["ts_slice_width"]
        self.TS_COMPOSITE_MODE = self.meta["ts_composite_mode"]
        self.ALGORITHM = self.meta["algorithm"]
        self.CLUSTER_ASSIGN = self.meta["cluster_assign"]
        self.CLASS_ASSIGN = self.meta["class_assign"]
        self.BAND_NAME_MAP = self.meta["band_name_map"]
        self.BAND_NAMES = list(self.BAND_NAME_MAP.keys())

    def harmonise_ESA(self, data: np.ndarray, BAND_NAMES: list[str], BAND_NAME_MAP: dict[str, str]) -> tuple[np.ndarray, list[str], dict[str, str]]:
        logger.info('Harmonising features and adding vegetation indices')
        dataloader = DataLoader(list(BAND_NAME_MAP.keys()), self.mask_value)
        data, features, band_name_map = dataloader.make_VIs(data, self.VIs, BAND_NAMES, BAND_NAME_MAP)
        return data, features, band_name_map

    def load(self, data):
        data, _, feature_names_map = self.harmonise_ESA(data, self.BAND_NAMES, self.BAND_NAME_MAP)
        logger.info("Added vegetation indices")
        
        preprocess = Preprocess(data, None, {}, {}, self.mode, self.model_dir)
        data, outlier_dict, normalise_dict = preprocess.prep(self.OUTLIER_PER_TIMESTAMP, self.mask_value)
        logger.info("Preprocessing complete")

        return data, outlier_dict, normalise_dict
    
    def deploy_CIMMYTMalawi_planet_fsl(
            self, 
            MODEL_DIR,
            in_path,
            out_dir
        ):
        
        no_data = 255
        cloud_classes = [0, 3, 5, 9]
        data = np.load(Path(in_path))
        metadata = np.load(Path(in_path).parent / "metadata.pickle.npy", allow_pickle=True).item()
        cld_msk = np.load(Path(in_path).parent / "cloudmask.npy")
        emb_path = Path(MODEL_DIR / "base_embeddings.npy")
        labels_path = Path(MODEL_DIR / "base_labels.npy")
        model_path = Path(MODEL_DIR / "model.h5")
        faiss_idx_path = Path(MODEL_DIR / "base_index.faiss")
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

        infer = Infer()
        embeddings = np.load(emb_path)
        embeddings_labels = np.load(labels_path)

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

        original_shape = data.shape
        data = data.transpose(1, 2, 0, 3).reshape(-1, original_shape[0], original_shape[3])

        t0 = time.time()
        data, _, _ = self.harmonise_CIMMYTMalawi_planet(data, self.BAND_NAMES, self.BAND_NAME_MAP)
        t1 = time.time()
        logger.info("Added vegetation indices")
        logger.info(f"Harmonisation took {t1 - t0:.2f} seconds")
        
        t0 = time.time()
        preprocess = Preprocess(data, labels, outlier_dict, normalise_dict, self.mode, self.model_dir)
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
        data = np.expand_dims(data, axis=-1)
        data = data[:, :, self.FEATURES_IDX]
        predictions, distances = infer.infer(base_network, embeddings, embeddings_labels, index, data)
        t1 = time.time()
        logger.info(f"Full prediction took {t1 - t0:.2f} seconds")
        logger.info("Model inference done")

        predictions_reshaped = predictions.reshape(original_shape[1], original_shape[2], -1)
        distances_reshaped = distances.reshape(original_shape[1], original_shape[2], -1)
        if predictions_reshaped.ndim == 3:
            pred_to_write = predictions_reshaped.transpose(2, 0, 1)  
        else:
            pred_to_write = np.expand_dims(predictions_reshaped, axis=0)  

        distances_reshaped = None
        if distances_reshaped is not None:
            if distances_reshaped.ndim == 3:
                dist_to_write = distances_reshaped.transpose(2, 0, 1)
            else:
                dist_to_write = np.expand_dims(distances_reshaped, axis=0)
            dist_to_write = dist_to_write * 10000
            out_raster = np.concatenate([pred_to_write, dist_to_write], axis=0)
        else:
            out_raster = pred_to_write
        logger.info("Reshaping done")

        invalid_mask = ~valid_mask
        for band in range(out_raster.shape[0]):
            out_raster[band, invalid_mask[0, :, :]] = no_data
        logger.info("No data masking done")

        transform = metadata['geotiff_metadata']['transform']
        crs = metadata['geotiff_metadata']['crs']
        meta = {
            "driver": "GTiff",
            "height": out_raster.shape[1],
            "width": out_raster.shape[2],
            "count": out_raster.shape[0],
            "dtype": out_raster.dtype,
            "crs": crs,
            "transform": transform,
            "nodata": no_data,
            "compress": "lzw"
        }
        try:
            out_raster_int = np.round(out_raster).astype(rio.uint8)
            meta.update(dtype='uint8')
            with rio.open(out_dir, 'w', **meta) as dst:
                dst.write(out_raster_int)
            logger.info(f"Written: {out_dir}")
        except Exception as e:
            logger.error(f"Failed to write raster: {e}")
        logger.info("Prediction Done")
