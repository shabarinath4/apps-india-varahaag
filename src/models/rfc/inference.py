import joblib
import numpy as np
import os

import numpy as np
import sys
from pathlib import Path
import logging
import faiss
import time
import joblib
import rasterio as rio
import os
from tensorflow.keras.models import load_model
import tensorflow as tf

from src.models.rfc.predict import PredictRFC

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InferRFC():
    pass

    def write(
            self,
            out_raster,
            metadata,
            valid_mask,
            no_data,
            out_path
    ):  
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
            with rio.open(out_path, 'w', **meta) as dst:
                dst.write(out_raster_int)
            logger.info(f"Written: {out_path}")
        except Exception as e:
            logger.error(f"Failed to write raster: {e}")
        logger.info("Prediction Done")

    def inference(
        self,
        MODEL_DIR,
        data,
        original_shape,
        model_type,
    ):
        rfc = joblib.load(os.path.join(MODEL_DIR, 'rfc_raw.pkl'))
        pred_obj = PredictRFC()
        predictions, probs = pred_obj.predict_rfc(rfc, data)
        # Adding distances
        predictions_reshaped = predictions.reshape(original_shape[1], original_shape[2], -1)
        probs = probs.reshape(original_shape[1], original_shape[2], -1)
        if predictions_reshaped.ndim == 3:
            pred_to_write = predictions_reshaped.transpose(2, 0, 1)  
        else:
            pred_to_write = np.expand_dims(predictions_reshaped, axis=0)  

        if probs is not None:
            if probs.ndim == 3:
                dist_to_write = probs.transpose(2, 0, 1)
            else:
                dist_to_write = np.expand_dims(probs, axis=0)
            dist_to_write = dist_to_write * 100
            out_raster = np.concatenate([pred_to_write, dist_to_write], axis=0)
        else:
            out_raster = pred_to_write

        return out_raster