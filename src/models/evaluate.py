import os
import numpy as np
import os
import numpy as np
import pandas as pd
from typing import Any, Dict, Optional
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
from sklearn.metrics import classification_report

from src.models.fsl3 import Evaluate
from src.utils.utilities import (VisualiseModelEmbeddings, SilhouetteMetric)

import logging
log = logging.getLogger(__name__)

def flatten_data(X: np.ndarray) -> np.ndarray:
    """
    Flattens the input data if it has more than 2 dimensions.

    Parameters:
        X (np.ndarray): Input data array.

    Returns:
        np.ndarray: Flattened data.
    """
    return X.reshape(X.shape[0], -1) if len(X.shape) > 2 else X

def save_classification_report(output_dir: str, name: str, y_true, y_pred, label_mapping: dict) -> None:
    """
    Generates and saves a classification report as a CSV file with original labels.

    Parameters:
        output_dir (str): Directory to save the report.
        name (str): Base filename for the report.
        y_true (array-like): True class labels (numeric).
        y_pred (array-like): Predicted class labels (numeric).
        label_mapping (dict): Dict mapping numeric labels to class names.

    Returns:
        None
    """
    labels = list(label_mapping.keys())
    target_names = [label_mapping[i] for i in labels]

    report_dict = classification_report(y_true, y_pred, labels=labels, target_names=target_names, output_dict=True)
    df_report = pd.DataFrame(report_dict).transpose()
    
    path = os.path.join(output_dir, f"{name}_classification_report.csv")
    df_report.to_csv(path)

def save_classification_report(output_dir: str, name: str, y_true, y_pred, label_mapping: dict) -> None:
    """
    Generates and saves a classification report as a CSV file with original labels.

    Parameters:
        output_dir (str): Directory to save the report.
        name (str): Base filename for the report.
        y_true (array-like): True class labels (numeric).
        y_pred (array-like): Predicted class labels (numeric).
        label_mapping (dict): Dict mapping numeric labels to class names.

    Returns:
        None
    """
    labels = list(label_mapping.keys())
    target_names = [label_mapping[i] for i in labels]

    report_dict = classification_report(y_true, y_pred, labels=labels, target_names=target_names, output_dict=True)
    df_report = pd.DataFrame(report_dict).transpose()
    
    path = os.path.join(output_dir, f"{name}_classification_report.csv")
    df_report.to_csv(path)

def save_confusion_matrix(output_dir: str, name: str, cm: np.ndarray, label_mapping: dict) -> None:
    """
    Saves a confusion matrix as a CSV file and an image with labels from label_mapping.

    Parameters:
        output_dir (str): Directory to save the matrix.
        name (str): Base filename for the matrix.
        cm (np.ndarray): Confusion matrix array.
        label_mapping (dict): Dict mapping class indices to label names.

    Returns:
        None
    """
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

# =======================
# Evaluate Base Model
# =======================
def evaluate_base_model(
    base_network: Any,
    X_test_input: np.ndarray,
    X_test_model: np.ndarray,
    y_test: np.ndarray,
    base_embeddings: np.ndarray,
    base_labels: np.ndarray,
    base_index: Any,
    X_train_input: Optional[np.ndarray] = None,
    y_train: Optional[np.ndarray] = None,
    results_dir: str = 'results',
    le: Dict = None
) -> Dict[str, Any]:
    label_mapping = dict(enumerate(le.classes_))

    print(X_train_input.shape)
    unique_X_train, unique_indices = np.unique(X_train_input, axis=0, return_index=True)
    print(unique_X_train.shape)
    unique_y_train = y_train[unique_indices]
    unique_base_embeddings = base_embeddings[unique_indices]
    num_samples = unique_X_train.shape[0]
    flattened_unique_X_train = unique_X_train.reshape(num_samples, -1)

    visualise_model_embeddings = VisualiseModelEmbeddings(
        results_dir,
        results_dir / 'pac_base_in.pdf',
        results_dir / 'pac_base_out.pdf',
        base_network,
        flattened_unique_X_train,
        unique_y_train,
        label_mapping,
        unique_base_embeddings
    )
    
    visualise_model_embeddings.pacmap()
    log.info("Base Network Embeddings visualised")
    silhouette_metric = SilhouetteMetric(label_mapping)
    silhouette_metric.silhouette_overall(flatten_data(X_test_input), y_test)
    silhouette_metric.silhouette_overall(base_embeddings, base_labels)
    silhouette_metric.save_silhouette_per_class_csv(flatten_data(X_test_input), y_test, results_dir / 'base_input.csv')
    silhouette_metric.save_silhouette_per_class_csv(base_embeddings, base_labels, results_dir / 'base_embed.csv')
    log.info("Base Network Silhouette calculated")

    evaluate = Evaluate()
    test_embeddings = base_network.predict(X_test_model)

    fsl_acc, fsl_pred, fsl_labels = evaluate.predict_class(
        base_embeddings,
        base_labels,
        test_embeddings,
        y_test,
        base_index
    )

    fsl_cm = confusion_matrix(y_test, fsl_pred)
    fsl_report = classification_report(y_test, fsl_pred)
    log.info(f"Base FSL Accuracy: {100 * fsl_acc:.2f}%")
    log.info(f"Base FSL Classification Report:\n{fsl_report}")
    log.info(f"Base FSL Confusion Matrix:\n{fsl_cm}")

    save_confusion_matrix(results_dir, 'base_fsl_cm', fsl_cm, label_mapping)
    save_classification_report(results_dir, 'base_fsl_report', y_test, fsl_pred, label_mapping)

    from collections import defaultdict
    from scipy.spatial.distance import cdist

    all_dists = cdist(test_embeddings, base_embeddings, metric='euclidean')
    max_correct_dists_per_class = defaultdict(float)

    for i, (true_label, pred_label) in enumerate(zip(y_test, fsl_pred)):
        if pred_label == true_label:
            matching_indices = np.where(base_labels == pred_label)[0]
            if matching_indices.size == 0:
                continue
            dists_to_class = all_dists[i][matching_indices]
            max_dist = np.max(dists_to_class)
            key = str(int(pred_label))
            max_correct_dists_per_class[key] = max(max_correct_dists_per_class[key], max_dist)

    thresholds = dict(max_correct_dists_per_class)

    rfc_emb = RandomForestClassifier(random_state=42)
    rfc_emb.fit(base_embeddings, base_labels)
    pred_rfc_emb = rfc_emb.predict(test_embeddings)
    rfc_emb_acc = accuracy_score(y_test, pred_rfc_emb)
    rfc_emb_report = classification_report(y_test, pred_rfc_emb)
    rfc_emb_cm = confusion_matrix(y_test, pred_rfc_emb)
    log.info(f"Embeddings RFC Accuracy: {100 * rfc_emb_acc:.2f}%")
    log.info(f"Embeddings RFC Classification Report:\n{rfc_emb_report}")
    log.info(f"Embeddings RFC Confusion Matrix:\n{rfc_emb_cm}")

    save_confusion_matrix(results_dir, 'rfc_base_embed_cm', rfc_emb_cm, label_mapping)
    save_classification_report(results_dir, 'rfc_base_embed_report', y_test, pred_rfc_emb, label_mapping)

    if X_train_input is not None and y_train is not None:
        X_train_flat = flatten_data(X_train_input)
        X_test_flat = flatten_data(X_test_input)

        rfc_raw = RandomForestClassifier(random_state=42)
        rfc_raw.fit(X_train_flat, y_train)
        pred_rfc_raw = rfc_raw.predict(X_test_flat)
        rfc_raw_acc = accuracy_score(y_test, pred_rfc_raw)
        rfc_raw_report = classification_report(y_test, pred_rfc_raw)
        rfc_raw_cm = confusion_matrix(y_test, pred_rfc_raw)
        log.info(f"Base RFC Accuracy: {100 * rfc_raw_acc:.2f}%")
        log.info(f"Base RFC Classification Report:\n{rfc_raw_report}")
        log.info(f"Base RFC Confusion Matrix:\n{rfc_raw_cm}")

        save_confusion_matrix(results_dir, 'rfc_base_raw_cm', rfc_raw_cm, label_mapping)
        save_classification_report(results_dir, 'rfc_base_raw_report', y_test, pred_rfc_raw, label_mapping)
    else:
        rfc_raw_acc = rfc_raw_report = rfc_raw_cm = rfc_raw = None

    return {
        'max_correct_dists_per_class': thresholds,
        'fsl_acc': fsl_acc,
        'fsl_report': fsl_report,
        'fsl_cm': fsl_cm,
        'rfc_emb_acc': rfc_emb_acc,
        'rfc_emb_report': rfc_emb_report,
        'rfc_emb_cm': rfc_emb_cm,
        'rfc_raw_acc': rfc_raw_acc,
        'rfc_raw_report': rfc_raw_report,
        'rfc_raw_cm': rfc_raw_cm,
    }

def evaluate_finetuned_model(
    ft_network: Any, 
    support_x: np.ndarray, 
    support_y: np.ndarray, 
    query_x: np.ndarray, 
    query_y: np.ndarray, 
    ft_embeddings: np.ndarray, 
    ft_labels: np.ndarray, 
    ft_index: Any,
    results_dir : str,
    le: Dict
) -> Dict[str, Any]:
    
    label_mapping = dict(enumerate(le.classes_))
    silhouette_metric = SilhouetteMetric(label_mapping)
    silhouette_metric.silhouette_overall(flatten_data(support_x), support_y)
    silhouette_metric.silhouette_overall(ft_embeddings, ft_labels)
    silhouette_metric.save_silhouette_per_class_csv(flatten_data(support_x), support_y, results_dir / 'ft_input.csv')
    silhouette_metric.save_silhouette_per_class_csv(ft_embeddings, ft_labels, results_dir / 'ft_embed.csv')
    log.info("Base Network Silhouette calculated")

    label_mapping = dict(enumerate(le.classes_))
    query_x_flat = flatten_data(query_x)
    query_embeddings = ft_network.predict(query_x)
    support_x_flat = flatten_data(support_x)
    support_embeddings = ft_network.predict(support_x)

    evaluate = Evaluate()
    query_acc, query_pred, query_labels = evaluate.predict_class(
        ft_embeddings, ft_labels, query_embeddings, query_y, ft_index
    )

    unique_X_train, unique_indices = np.unique(support_x, axis=0, return_index=True)
    valid_unique_indices = [i for i in unique_indices if i < len(ft_embeddings)]
    unique_X_train = unique_X_train[:len(valid_unique_indices)]
    unique_y_train = support_y[valid_unique_indices]
    unique_base_embeddings = ft_embeddings[valid_unique_indices]
    num_samples = unique_X_train.shape[0]
    flattened_unique_X_train = unique_X_train.reshape(num_samples, -1)

    visualise_model_embeddings = VisualiseModelEmbeddings(
        results_dir,
        results_dir / 'pac_ft_in.pdf',
        results_dir / 'pac_ft_out.pdf',
        ft_network,
        flattened_unique_X_train,
        unique_y_train,
        label_mapping,
        unique_base_embeddings
    )
    visualise_model_embeddings.pacmap()
    log.info("Ft Network Embeddings visualised")

    query_cm = confusion_matrix(query_y, query_pred)
    query_report = classification_report(query_y, query_pred)
    log.info(f"Finetune FSL Query Accuracy: {100 * query_acc:.2f}%")
    log.info(f"Finetune FSL Query Classification Report:\n{query_report}")
    log.info(f"Finetune FSL Query Confusion Matrix:\n{query_cm}")

    save_confusion_matrix(results_dir, 'rfc_ft_fsl_cm', query_cm, label_mapping)
    save_classification_report(results_dir, 'rfc_ft_fsl_cm', query_y, query_pred, label_mapping)
    
    from collections import defaultdict
    from scipy.spatial.distance import cdist

    all_dists = cdist(query_embeddings, ft_embeddings, metric='euclidean')
    max_correct_dists_per_class = defaultdict(float)

    for i, (true_label, pred_label) in enumerate(zip(query_y, query_pred)):
        if pred_label == true_label:
            matching_indices = np.where(ft_labels == pred_label)[0]
            if matching_indices.size == 0:
                continue
            dists_to_class = all_dists[i][matching_indices]
            max_dist = np.max(dists_to_class)
            key = str(int(pred_label))
            max_correct_dists_per_class[key] = max(max_correct_dists_per_class[key], max_dist)

    thresholds = dict(max_correct_dists_per_class)

    rfc_emb = RandomForestClassifier(random_state=42)
    rfc_emb.fit(ft_embeddings, ft_labels)
    pred_rfc_emb = rfc_emb.predict(query_embeddings)
    rfc_emb_acc = accuracy_score(query_y, pred_rfc_emb)
    rfc_emb_report = classification_report(query_y, pred_rfc_emb)
    rfc_emb_cm = confusion_matrix(query_y, pred_rfc_emb)

    save_confusion_matrix(results_dir, 'rfc_ft_emb_cm', rfc_emb_cm, label_mapping)
    save_classification_report(results_dir, 'rfc_ft_emb_cm', query_y, pred_rfc_emb, label_mapping)

    log.info(f"Finetune RFC Accuracy: {100 * rfc_emb_acc:.2f}%")
    log.info(f"Finetune RFC Report:\n{rfc_emb_report}")
    log.info(f"Finetune RFC Confusion Matrix:\n{rfc_emb_cm}")

    if support_x is not None and support_y is not None and query_x is not None and query_y is not None:
        support_x_flat = flatten_data(support_x)
        query_x_flat = flatten_data(query_x)

        rfc_raw = RandomForestClassifier(random_state=42)
        rfc_raw.fit(support_x_flat, support_y)
        pred_rfc_raw = rfc_raw.predict(query_x_flat)
        rfc_raw_acc = accuracy_score(query_y, pred_rfc_raw)
        rfc_raw_report = classification_report(query_y, pred_rfc_raw)
        rfc_raw_cm = confusion_matrix(query_y, pred_rfc_raw)

        save_confusion_matrix(results_dir, 'rfc_ft_raw_cm', rfc_raw_cm, label_mapping)
        save_classification_report(results_dir, 'rfc_ft_raw_cm', query_y, pred_rfc_raw, label_mapping)

        log.info(f"Finetune RFC (Raw Data) Accuracy: {100 * rfc_raw_acc:.2f}%")
        log.info(f"Finetune RFC (Raw Data) Classification Report:\n{rfc_raw_report}")
        log.info(f"Finetune RFC (Raw Data) Confusion Matrix:\n{rfc_raw_cm}")
    else:
        rfc_raw_acc = None
        rfc_raw_report = None
        rfc_raw_cm = None
        rfc_raw = None

    return {
        'max_correct_dists_per_class': thresholds,
        'label_mapping': label_mapping,
        'query_acc': query_acc,
        'query_pred': query_pred,
        'query_report': query_report,
        'query_cm': query_cm,
        'rfc_emb_acc': rfc_emb_acc,
        'rfc_emb_report': rfc_emb_report,
        'rfc_emb_cm': rfc_emb_cm,
        'rfc_emb_classifier': rfc_emb,
        'rfc_raw_acc': rfc_raw_acc,
        'rfc_raw_report': rfc_raw_report,
        'rfc_raw_cm': rfc_raw_cm,
        'rfc_raw_classifier': rfc_raw
    }
