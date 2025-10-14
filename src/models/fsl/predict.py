import numpy as np
import rasterio as rio
from pathlib import Path
import time
import faiss
import logging
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

base_path = Path(__file__).resolve().parents[2]
sys.path.append('C:\\Users\\shaba\\Desktop\\1.1 Crop ID - Domain Adaptation\\algos-crop_identification-deCIFR')


class Predict:
    def __init__(self):
        pass

    def make_faiss_index(
        self, 
        embeddings: np.ndarray, 
        labels: np.ndarray
    ):
        d = embeddings.shape[1]
        index = faiss.IndexFlatL2(d)
        index.add(embeddings.astype('float32'))
        return index, labels

    def predict_using_faiss(
        self, 
        index: faiss.IndexFlatL2, 
        labels: np.ndarray, 
        query_embeddings: np.ndarray, 
        k: int = 1
    ):
        t0 = time.time() 
        distances, indices = index.search(query_embeddings.astype('float32'), k)
        labels = np.array(labels)
        t1 = time.time() 
        print(f"Search took {t1 - t0:.2f} seconds")
        
        t0 = time.time() 
        if k > 1:
            predicted_labels = labels[indices]
            predicted_labels = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=1, arr=predicted_labels)
        else:
            indices = np.clip(indices.flatten(), 0, len(labels) - 1)
            predicted_labels = labels[indices]
        t1 = time.time() 
        print(f"Prediction took {t1 - t0:.2f} seconds")

        return predicted_labels, distances
    
    def predict_class(
        self, 
        ref_emb: np.ndarray, 
        y_ref: np.ndarray, 
        eval_emb: np.ndarray, 
        y_eval: None, 
        ref_index: faiss.IndexFlatL2
    ):
        if ref_index == None:
            ref_index, y_train = self.make_faiss_index(ref_emb, y_ref)
        y_pred, distances = self.predict_using_faiss(ref_index, y_ref, eval_emb)
        if y_eval is not None:
            accuracy = np.mean(y_pred == y_eval)
            return accuracy, y_pred, y_eval
        else:
            return y_pred, distances
    