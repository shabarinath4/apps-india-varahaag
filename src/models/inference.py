import numpy as np
import faiss
import time
import tensorflow as tf

class Infer:
    def __init__(self):
        pass

    def make_faiss_index(
        self, 
        embeddings: np.ndarray, 
        labels: np.ndarray
    ):
        """
        Make a faiss index for faster similarity search

        Parameters
        ----------
        embeddings : np.ndarray
            embedding vector
        labels : np.ndarray
            label array
        
        Returns
        -------
        index : faiss.IndexFlatL2
        labels : np.ndarray
        """
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
        """
        Make a prediction by searching the query in the index database

        Parameters
        ----------
        index : faiss.IndexFlatL2
            index to search in 
        labels : np.ndarray
            label array
        query_embeddings : np.ndarray
            embedding vector
        k : int
            number of nearest neighbors to search for a match
        
        Returns
        -------
        predicted_labels : np.ndarray
        distances : np.ndarray
        """
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
    
    def infer(
        self, 
        base_network: tf.keras.Model, 
        support_emb: np.ndarray, 
        support_y: np.ndarray, 
        support_index: faiss.IndexFlatL2 , 
        x_infer: np.ndarray
    ):
        """
        Make a prediction by searching the query in the index database

        Parameters
        ----------
        base_network : tf.keras.Model
            The base embedding model to predict using
        support_emb : np.ndarray
            support embeddings
        support_y : np.ndarray
            support labels
        support_index : faiss.IndexFlatL2
            support index database
        x_infer : np.ndarray
            query array for inference

        Returns
        -------
        predicted_labels : np.ndarray
        distances : np.ndarray
        """
        query_emb = base_network.predict(x_infer, verbose=0)
        if support_index == None:
            support_index, support_y = Infer().make_faiss_index(support_emb, support_y)
        predicted_labels, distances = self.predict_using_faiss(support_index, support_y, query_emb)
        return predicted_labels, distances

