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
    def __init__(self, RESULTS_DIR, FIG_NAME_IN, FIG_NAME_OUT, model, X_train, y_train, label_mapping, trained_embeddings):
        self.RESULTS_DIR = RESULTS_DIR
        self.FIG_NAME_IN = FIG_NAME_IN
        self.FIG_NAME_OUT = FIG_NAME_OUT
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.label_mapping = label_mapping
        self.trained_embeddings = trained_embeddings

    def pacplots(self, data, file_name, title):
        def generate_color_dict(n, seed=4):
            np.random.seed(seed)
            hues = np.linspace(0, 1, n)
            saturations = np.linspace(0.7, 1.0, n)
            values = np.linspace(0.5, 1.0, n)
            colors = [colorsys.hsv_to_rgb(hues[i], saturations[i], values[i]) for i in range(n)]
            color_dict = {i: colors[i] for i in range(n)}
            return color_dict

        categories = pd.DataFrame(self.y_train).value_counts().index.tolist()
        categories = [int(c[0]) for c in categories]
        color_dict = generate_color_dict(len(set(self.y_train)))

        embedding = pacmap.PaCMAP(n_components=2, n_neighbors=5, MN_ratio=0.5, FP_ratio=3.0)
        X_transformed = embedding.fit_transform(data, init="pca")

        plt.figure(figsize=(6, 6))
        for category in categories:
            plt.scatter(
                X_transformed[self.y_train == category, 0],
                X_transformed[self.y_train == category, 1],
                color=color_dict[category],
                label=f'{self.label_mapping[category]}',
                s=4,
                alpha=0.7
            )
        plt.title(title)
        plt.legend()
        plt.savefig(file_name, format='pdf', bbox_inches="tight")
        plt.close()

        fig_legend = plt.figure(figsize=(2, 2))
        ax = fig_legend.add_subplot(111)
        ax.axis('off')
        handles, labels = plt.gca().get_legend_handles_labels()
        legend = fig_legend.legend(handles, labels, loc='center', frameon=False)
        legend_file_name = str(file_name).replace('.pdf', '_legend.pdf')
        fig_legend.savefig(legend_file_name, bbox_inches='tight')
        plt.close(fig_legend)

    def pacmap(self):
        if not os.path.exists(os.path.join(self.RESULTS_DIR, "plots\\")):
            os.makedirs(os.path.join(self.RESULTS_DIR, "plots\\"))
        self.pacplots(self.X_train, file_name=self.FIG_NAME_IN, title='PaCMAP Visualisation of training data')
        self.pacplots(self.trained_embeddings, file_name=self.FIG_NAME_OUT, title='PaCMAP Visualisation of trained embeddings')

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


