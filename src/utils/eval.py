from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns

def save_classification_report(output_dir: str, name: str, y_true, y_pred, label_mapping: dict) -> None:

    labels = list(label_mapping.keys())
    target_names = [label_mapping[i] for i in labels]

    report_dict = classification_report(y_true, y_pred, labels=labels, target_names=target_names, output_dict=True)
    df_report = pd.DataFrame(report_dict).transpose()
    
    path = os.path.join(output_dir, f"{name}_classification_report.csv")
    df_report.to_csv(path)

def save_confusion_matrix(output_dir: str, name: str, cm: np.ndarray, label_mapping: dict) -> None:

    labels = [label_mapping[i] for i in range(len(label_mapping))]
    df_cm = pd.DataFrame(cm, index=labels, columns=labels)

    # Save CSV
    csv_path = os.path.join(output_dir, f"{name}_confusion_matrix.csv")
    df_cm.to_csv(csv_path)

    # Save image
    plt.figure(figsize=(10, 8))
    sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues', cbar=True, xticklabels=True, yticklabels=True)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    img_path = os.path.join(output_dir, f"{name}_confusion_matrix.png")
    plt.savefig(img_path, format='png')
    plt.close()

def analyze_feature_importances(rfc, X_shape, band_names=None, time_labels=None, results_dir=None):
    
    n_samples, n_time, n_bands = X_shape
    
    importances = rfc.feature_importances_
    importances_reshaped = importances.reshape(n_time, n_bands)

    band_importance = importances_reshaped.sum(axis=0)
    time_importance = importances_reshaped.sum(axis=1)

    band_importance /= band_importance.sum()
    time_importance /= time_importance.sum()

    if band_names is None:
        band_names = [f"Band {i}" for i in range(n_bands)]
    if time_labels is None:
        time_labels = [f"T{i}" for i in range(n_time)]

    fig, axes = plt.subplots(3, 1, figsize=(12, 14))

    im = axes[0].imshow(importances_reshaped, cmap="viridis", aspect="auto")
    axes[0].set_xticks(range(n_bands))
    axes[0].set_xticklabels(band_names, rotation=45, ha="right")
    axes[0].set_yticks(range(n_time))
    axes[0].set_yticklabels(time_labels)
    axes[0].set_title("Feature Importance (Time × Band)")
    fig.colorbar(im, ax=axes[0], orientation="vertical", label="Importance")

    axes[1].bar(band_names, band_importance)
    axes[1].set_xticklabels(band_names, rotation=45, ha="right")
    axes[1].set_title("Aggregated Band Importance")
    axes[1].set_ylabel("Normalized Importance")

    axes[2].bar(time_labels, time_importance)
    axes[2].set_title("Aggregated Time Importance")
    axes[2].set_ylabel("Normalized Importance")
    
    plt.tight_layout()
    if results_dir:
        plt.savefig(f"{results_dir}/feature_importances.png", dpi=300)
    else:
        plt.show()

    return importances_reshaped, band_importance, time_importance
