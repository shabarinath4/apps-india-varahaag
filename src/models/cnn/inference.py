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

from src.models.cnn.predict import PredictCNN

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InferCNN():
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

    def inference(self, MODEL_DIR, data, original_shape, model_type):
        
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
        
        model_path = os.path.join(MODEL_DIR, 'cnn_raw.h5')
        
        cnn = load_model(
            model_path,
            compile=False,
             custom_objects={'TemporalPyramidPooling': TemporalPyramidPooling}
        )
        pred_obj = PredictCNN()
        predictions, probs = pred_obj.predict_cnn(cnn, data)
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
