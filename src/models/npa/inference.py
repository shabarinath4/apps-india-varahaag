import numpy as np

class Inference:
    def __init__(
            self,
            data,
            model
    ):
        self.data = data
        self.model = model

    def assign(
            self,
            assign_logic
    ):
        centroids = self.model.cluster_centers_
        centroid_scores = centroids.sum(axis=1)
        rapseed_score = centroid_scores
        if assign_logic == 'max':
            cluster_mapped = np.argmax(rapseed_score)
            print('assigning max cluster')
        if assign_logic == 'min':
            cluster_mapped = np.argmin(rapseed_score)
            print('assigning max cluster')
        return cluster_mapped

    # map_optical
    # map_radar
    # create_score
    # combine_score
    # set_labels

    def set_labels(
            self,
            inferred,
            cluster_mapped,
            set_logic
    ):
        assigned = (inferred == cluster_mapped).astype(int)
    
        if set_logic == 'any':
            classified = np.any(assigned == 1, axis=0).astype(int)
            print('setting crop')

        return classified