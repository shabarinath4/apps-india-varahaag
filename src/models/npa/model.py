import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture

import numpy as np
from sklearn.cluster import KMeans
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnx
from onnx import StringStringEntryProto

class Models:
    def __init__(
            self,
            data: np.ndarray,
            random_state: int=1,
            n_clusters: int=5
    ):
        self.data = data
        self.random_state = random_state
        self.n_clusters = n_clusters

    def kmeans(self):
        kmeans_model = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        kmeans_labels = kmeans_model.fit_predict(self.data)
        return kmeans_model, kmeans_labels

    def gmm(self):
        gmm_model = GaussianMixture(n_components=self.n_clusters, random_state=self.random_state)
        gmm_labels = gmm_model.fit_predict(self.data)
        return gmm_model, gmm_labels
    
    def agglomerative(self):
        agg_model = AgglomerativeClustering(n_clusters=self.n_clusters)
        agg_labels = agg_model.fit_predict(self.data)
        return agg_model, agg_labels
    
    def convert_to_onxx(
            self,
            model,
            shape
        ):

        initial_type = [('features', FloatTensorType([None, shape[1]]))]
        onx = convert_sklearn(model, initial_types=initial_type, target_opset=9)

        onnx_path = "model.onnx"
        with open(onnx_path, "wb") as f:
            f.write(onx.SerializeToString())

        model = onnx.load(onnx_path)

        class_params = {
            "class_names": list(range(5)),
            "class_to_label": list(range(5)),
            "description": "clustering model with n clusters"
        }
        entry = StringStringEntryProto(key="class_params", value=str(class_params))
        model.metadata_props.append(entry)
        patched_path = "model.onnx"
        onnx.save(model, patched_path)

        print(f"Patched ONNX model saved to: {patched_path}")