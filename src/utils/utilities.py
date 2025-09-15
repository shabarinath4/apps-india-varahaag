import os
import pacmap
import numpy as np
import random
import numpy as np
import colorsys
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import colorsys
import pacmap
import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
import pacmap
from sklearn.metrics import silhouette_score, silhouette_samples

from shapely.geometry import shape
from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples


class FileOperations(object):
    def __init__(self) -> None:
        pass

    def list_files_in_subdirectories(self, directory):
        file_list = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                full_path = os.path.join(root, file)
                file_list.append(full_path)
        return file_list
    
class VisualiseModelEmbeddings(object):
    def __init__(self, RESULTS_DIR, X_train, trained_embeddings, X_test, test_embeddings, label_mapping, y_train, y_test, y_pred):
        self.RESULTS_DIR = RESULTS_DIR
        self.X_train = X_train
        self.trained_embeddings = trained_embeddings
        self.X_test = X_test
        self.test_embeddings = test_embeddings
        self.label_mapping = label_mapping
        self.y_train = y_train
        self.y_test = y_test
        self.y_pred = y_pred

    def pacplots(self, plot_folder):
        def generate_color_dict(n, seed=4):
            np.random.seed(seed)
            hues = np.linspace(0, 1, n)
            saturations = np.linspace(0.7, 1.0, n)
            values = np.linspace(0.5, 1.0, n)
            colors = [colorsys.hsv_to_rgb(hues[i], saturations[i], values[i]) for i in range(n)]
            return {i: colors[i] for i in range(n)}

        categories = pd.DataFrame(self.y_train).value_counts().index.tolist()
        categories = [int(c[0]) for c in categories]
        color_dict = generate_color_dict(len(set(self.y_train)))

        param_combinations = list(itertools.product([5, 10], [0.5, 1.0], [2.0, 3.0]))[:6]
        total_samples = len(self.X_train)

        for i, (n_neighbors, mn_ratio, fp_ratio) in enumerate(param_combinations):
            datasets = [
                (self.X_train, self.y_train, os.path.join(plot_folder, f'xtrain_{i+1}.pdf')),
                (self.trained_embeddings, self.y_train, os.path.join(plot_folder, f'train_embeddings_{i+1}.pdf')),
                (self.X_test.reshape(self.X_test.shape[0], -1), self.y_test, os.path.join(plot_folder, f'xtest_{i+1}.pdf')),
                (self.test_embeddings, self.y_test, os.path.join(plot_folder, f'test_embeddings_{i+1}.pdf')),
                (self.test_embeddings, self.y_pred, os.path.join(plot_folder, f'pred_embeddings_{i+1}.pdf'))
            ]
            
            for data, labels, fname in datasets:
                embedding = pacmap.PaCMAP(
                    n_components=2,
                    n_neighbors=n_neighbors,
                    MN_ratio=mn_ratio,
                    FP_ratio=fp_ratio
                )
                X_transformed = embedding.fit_transform(data, init="pca")
                plt.figure(figsize=(6, 6))
                for category in categories:
                    plt.scatter(
                        X_transformed[labels == category, 0],
                        X_transformed[labels == category, 1],
                        color=color_dict[category],
                        label=self.label_mapping[category],
                        s=4,
                        alpha=0.7
                    )
                plt.title(f'PaCMAP (n={n_neighbors}, MN={mn_ratio}, FP={fp_ratio})\nTotal Samples: {total_samples}')
                plt.legend()
                plt.savefig(fname, format='pdf', bbox_inches="tight")
                plt.close()

    def pacmap(self):
        plot_folder = os.path.join(self.RESULTS_DIR, "plots")
        if not os.path.exists(plot_folder):
            os.makedirs(plot_folder)
        self.pacplots(plot_folder)

class SilhouetteMetric(object):
    def __init__(self, label_mapping):
        self.label_mapping = label_mapping
    
    def silhouette_overall(self, x, y):
        sil_coeff = silhouette_score(x, y)
        print(f"Silhouette Coefficient: {round(sil_coeff, 3)}")

    def silhouette_per_class(self, x, y):
        sil_samples = silhouette_samples(x, y)
        per_cluster_sil = {}
        for c in set(np.array(y)):
            per_cluster_sil[self.label_mapping[c]] = round(np.mean(sil_samples[y == c]), 3)
        print(f"Silhouette Per Class: {per_cluster_sil}")
        return per_cluster_sil
    
    def save_silhouette_per_class_csv(self, x, y, csv_file):
        per_cluster_sil = self.silhouette_per_class(x, y)
        df = pd.DataFrame(list(per_cluster_sil.items()), columns=['Class', 'Silhouette'])
        df.to_csv(csv_file, index=False)

class PlotCnfClf(object):
    def __init__(self, clf_rep, cnf_mat, label_mapping, RESULTS_DIR, suffix):
        self.cnf_mat = cnf_mat
        self.clf_rep = clf_rep
        self.label_mapping = label_mapping
        self.RESULTS_DIR = RESULTS_DIR
        self.suffix = suffix
        
    def save_cnf_clf(self):
        if not os.path.exists(self.RESULTS_DIR + '\\performance\\'):
            os.makedirs(self.RESULTS_DIR + '\\performance\\')

        if self.label_mapping:
            plt.figure(figsize=(8, 6))
            sns.heatmap(self.cnf_mat, annot=True, fmt="d", cmap="Blues", 
                        xticklabels=list(self.label_mapping.values()), yticklabels=list(self.label_mapping.values()))
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title('Confusion Matrix with Original Labels')
            plt.savefig(self.RESULTS_DIR + '\\performance\\classification_report'+ str(self.suffix) +'.jpg')
        with open(self.RESULTS_DIR + '\\performance\\classification_report'+ str(self.suffix) +'.txt', "w") as f:
            f.write(self.clf_rep)

class Geometry(object):
    def get_coordinates(self, geojson_str):
        geom = shape(eval(geojson_str))  
        if geom.geom_type == 'Point':
            return geom.y, geom.x  
        elif geom.geom_type == 'Polygon':
            centroid = geom.centroid
            return centroid.y, centroid.x  
        else:
            return None  

    def get_lat_lon(self, df, GEOM_COLNAME):
        df[['latitude', 'longitude']] = df[GEOM_COLNAME].apply(self.get_coordinates).apply(pd.Series)
        return df


