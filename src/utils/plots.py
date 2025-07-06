import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from typing import List, Union
import os

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from typing import Union

import numpy as np
import os
import matplotlib.pyplot as plt
from typing import List, Union
from sklearn.cluster import KMeans
import pandas as pd

class Explainability():
    def __init__(self):
        pass

    def feature_importances(
        self,
        rfc: RandomForestClassifier,
        X_train: Union[pd.DataFrame, np.ndarray],
        RESULTS_DIR: str
    ) -> None:
        """
        Plots the top 20 most important features from a trained Random Forest classifier.

        Parameters
        ----------
        rfc : RandomForestClassifier
            A trained random forest classifier.
        X_train : pd.DataFrame or np.ndarray
            Training features used to extract feature names. If not a DataFrame, indices are used.
        MODEL_DIR : str
            Directory path to save the output plot.

        Returns
        -------
        None
        """
        importances = rfc.feature_importances_
        feature_names = (
            X_train.columns if isinstance(X_train, pd.DataFrame) else np.arange(len(importances))
        )
        sorted_indices = np.argsort(importances)[::-1][:20]

        plt.figure(figsize=(12, 6))
        plt.bar(range(len(sorted_indices)), importances[sorted_indices], align="center", color="skyblue")
        plt.xticks(range(len(sorted_indices)), np.array(feature_names)[sorted_indices], 
                rotation=90, fontsize=9)
        plt.xlabel("Feature", fontsize=12)
        plt.ylabel("Importance", fontsize=12)
        plt.title("Top 20 Feature Importances from Random Forest", fontsize=14, weight='bold')
        plt.tight_layout()
        plt.savefig(f"{RESULTS_DIR}/feature_importances.png", dpi=300)

class Plots():
    def __init__(self):
        pass

    def plot_clusters(
        self,
        X: np.ndarray,  # shape (samples, time, bands)
        y: Union[np.ndarray, List],
        label_mapping,
        band_names: List[str],
        FEATURES_IDX: List[int],
        mode: str = "paper",
        RESULTS_DIR: str = 'results',
        train_mode: str = 'base'
    ) -> None:

        rc_params_paper = {
            'font.family': 'serif',
            'font.size': 11,
            'axes.titlesize': 14,
            'axes.titleweight': 'bold',
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'legend.title_fontsize': 12,
            'lines.linewidth': 2,
            'lines.markersize': 6,
            'figure.figsize': (6, 4),
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'figure.autolayout': True
        }

        rc_params_presentation = {
            'font.family': 'sans-serif',
            'font.size': 14,
            'axes.titlesize': 18,
            'axes.titleweight': 'bold',
            'axes.labelsize': 16,
            'xtick.labelsize': 14,
            'ytick.labelsize': 14,
            'legend.fontsize': 13,
            'legend.title_fontsize': 14,
            'lines.linewidth': 2.5,
            'lines.markersize': 8,
            'figure.figsize': (8, 6),
            'figure.dpi': 150,
            'savefig.dpi': 150,
            'figure.autolayout': True
        }

        plt.rcParams.update(rc_params_paper if mode == "paper" else rc_params_presentation)

        X = np.array(X)
        y = np.array(y)
        unique_classes = np.unique(y)
        T = X.shape[1]

        import seaborn as sns
        import datetime
        num_crops = len(unique_classes)
        palette = sns.color_palette("tab10", n_colors=num_crops)  # or "Set2", "husl", etc.
        crop_colors = {class_label: palette[i] for i, class_label in enumerate(unique_classes)}
        start_date = datetime.datetime(2025, 10, 4)  # Starting date (Oct 4)
        interval_days = 14

        # Plot 1: Cluster Medians
        fig, axes = plt.subplots(len(FEATURES_IDX), len(unique_classes), figsize=(16, 10), sharex=True, sharey=True)
        for j, band_idx in enumerate(np.arange(len(FEATURES_IDX))):
            for i, class_label in enumerate(unique_classes):
                ax = axes[j, i] if len(unique_classes) > 1 else axes[j]
                class_data = X[y == class_label, :, band_idx]
                n_clusters = min(3, len(class_data))
                if n_clusters < 2:
                    continue
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=3)
                clusters = kmeans.fit_predict(class_data)
                cluster_medians = np.array([class_data[clusters == k].mean(axis=0) for k in range(n_clusters)])
                ax.plot(cluster_medians.T, alpha=0.4, color='blue')
                ax.plot(np.median(class_data, axis=0), color='red', linestyle='--', label='Median')
                ax.set_title(f'Class {class_label} - {band_names[band_idx]}')
                x_vals = np.arange(T)
                step = 3
                dates = [start_date + datetime.timedelta(days=ti*interval_days) for ti in range(T)]
                ax.set_xticks(x_vals[::step])
                # ax.set_xticklabels([d.strftime('%d %b') for d in dates][::step], rotation=45)
                ax.grid(True, linestyle='--', alpha=0.5)

        plt.tight_layout()
        os.makedirs(os.path.join(RESULTS_DIR, train_mode), exist_ok=True)
        plt.savefig(f"{RESULTS_DIR}/{train_mode}/clusters.png", dpi=300)
        plt.close()

        # Plot 2: IQR Fill Per Class
        fig, axes = plt.subplots(len(FEATURES_IDX), len(unique_classes), figsize=(16, 10), sharex=True, sharey=True)
        for j, band_idx in enumerate(np.arange(len(FEATURES_IDX))):
            for i, class_label in enumerate(unique_classes):
                crop_name = label_mapping[class_label]
                color = crop_colors[class_label]
                ax = axes[j, i] if len(unique_classes) > 1 else axes[j]
                class_data = X[y == class_label, :, band_idx]
                q1 = np.quantile(class_data, 0.25, axis=0)
                q3 = np.quantile(class_data, 0.75, axis=0)
                median = np.median(class_data, axis=0)
                x_vals = np.arange(T)
                ax.plot(x_vals, median, color=color, label=crop_name)
                ax.fill_between(x_vals, q1, q3, alpha=0.3, color='gray', label='IQR')
                ax.set_title(f'Class {crop_name} - {band_names[band_idx]}')
                step = 3
                dates = [start_date + datetime.timedelta(days=ti*interval_days) for ti in range(T)]
                ax.set_xticks(x_vals[::step])
                # ax.set_xticklabels([d.strftime('%d %b') for d in dates][::step], rotation=45)
                ax.grid(True, linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.savefig(f"{RESULTS_DIR}/{train_mode}/iqr_per_class.png", dpi=300)
        plt.close()

        # Plot 3: All Classes in One Plot per Band
        for _, band_idx in enumerate(np.arange(len(FEATURES_IDX))):
            fig, ax = plt.subplots(figsize=(10, 6))
            for i, class_label in enumerate(unique_classes):
                crop_name = label_mapping[class_label]
                color = crop_colors[class_label]
                class_data = X[y == class_label, :, band_idx]
                q1 = np.quantile(class_data, 0.25, axis=0)
                q3 = np.quantile(class_data, 0.75, axis=0)
                median = np.median(class_data, axis=0)
                x_vals = np.arange(T)
                ax.plot(x_vals, median, label=crop_name, color=color)
                ax.fill_between(x_vals, q1, q3, alpha=0.2)
            ax.set_title(f'All Classes - {band_names[band_idx]}')
            step = 3
            dates = [start_date + datetime.timedelta(days=ti*interval_days) for ti in range(T)]
            ax.set_xticks(x_vals[::step])
            # ax.set_xticklabels([d.strftime('%d %b') for d in dates][::step], rotation=45)
            ax.set_xlabel("Time")
            ax.set_ylabel("Reflectance")
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.5)
            plt.tight_layout()
            plt.savefig(f"{RESULTS_DIR}/{train_mode}/all_crops_{band_names[band_idx]}.png", dpi=300)
            plt.close()

        for class_label in unique_classes:
            for i, band_idx in enumerate(np.arange(len(FEATURES_IDX))):
                if i == 14:
                    crop_name = label_mapping[class_label]
                    color = crop_colors[class_label]
                    fig, ax = plt.subplots(figsize=(10, 6))
                    class_data = X[y == class_label, :, band_idx]  # (samples, time)
                    x_vals = np.arange(T)

                    min_vals = np.min(class_data, axis=0)
                    max_vals = np.max(class_data, axis=0)
                    median = np.median(class_data, axis=0)
                    q1 = np.quantile(class_data, 0.25, axis=0)
                    q3 = np.quantile(class_data, 0.75, axis=0)

                    ax.fill_between(x_vals, q1, q3, alpha=0.2, color=color)
                    ax.plot(x_vals, median, color=color, label=crop_name)
                    
                    ax.set_title(f'Crop {crop_name} - {band_names[band_idx]}')
                    step = 3
                    dates = [start_date + datetime.timedelta(days=ti*interval_days) for ti in range(T)]
                    ax.set_xticks(x_vals[::step])
                    # ax.set_xticklabels([d.strftime('%d %b') for d in dates][::step], rotation=45)

                    ax.set_xlabel("Time")
                    ax.set_ylabel("Reflectance")
                    ax.grid(True, linestyle='--', alpha=0.5)
                    ax.legend()

                    plt.tight_layout()
                    fname = f"{RESULTS_DIR}/{train_mode}/crop_fill_{crop_name}_{band_names[band_idx]}.png"
                    plt.savefig(fname, dpi=300)
                    plt.close()

        for class_label in unique_classes:
            for i, band_idx in enumerate(np.arange(len(FEATURES_IDX))):
                if i == 7:  # You can change or remove this condition as needed
                    crop_name = label_mapping[class_label]
                    color = crop_colors[class_label]
                    fig, ax = plt.subplots(figsize=(10, 6))

                    class_data = X[y == class_label, :, band_idx]  # shape: (samples, time)
                    x_vals = np.arange(T)

                    # Plot individual sample lines
                    for sample in class_data:
                        ax.plot(x_vals, sample, color=color, alpha=0.3, linewidth=0.8)

                    # Optionally add median line
                    median = np.median(class_data, axis=0)
                    ax.plot(x_vals, median, color=color, label=crop_name, linewidth=2)

                    ax.set_title(f'Crop {crop_name} - {band_names[band_idx]}')
                    step = 3
                    dates = [start_date + datetime.timedelta(days=ti*interval_days) for ti in range(T)]
                    ax.set_xticks(x_vals[::step])
                    # ax.set_xticklabels([d.strftime('%d %b') for d in dates][::step], rotation=45)
                    ax.set_xlabel("Time")
                    ax.set_ylabel("Reflectance")
                    ax.grid(True, linestyle='--', alpha=0.5)
                    ax.legend()

                    plt.tight_layout()
                    fname = f"{RESULTS_DIR}/{train_mode}/crop_line_{crop_name}_{band_names[band_idx]}.png"
                    plt.savefig(fname, dpi=300)
                    plt.close()