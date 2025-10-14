import os
import numpy as np
import pandas as pd
import joblib
import logging
from pathlib import Path
from typing import Any, Dict, Optional
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from collections import defaultdict
from scipy.spatial.distance import cdist

from src.models.fsl.predict import Predict
from src.utils.utilities import VisualiseModelEmbeddings, SilhouetteMetric
from src.models.rfc.model import TrainRFC
from src.models.cnn.model import TrainCNN
from src.utils.eval import save_confusion_matrix, save_classification_report

log = logging.getLogger(__name__)

def flatten(x: np.ndarray) -> np.ndarray:
    return x.reshape(x.shape[0], -1) if len(x.shape) > 2 else x

def eval_fsl(
    base_model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    train_embeddings: np.ndarray,
    base_index: Any,
    X_train: Optional[np.ndarray] = None,
    y_train: Optional[np.ndarray] = None,
    results_dir: str = 'results',
    model_dir: str = 'saved_models',
    label_encoder: Dict = None
) -> Dict[str, Any]:
    label_map = dict(enumerate(label_encoder.classes_))

    X_train_unique, unique_idx = np.unique(X_train, axis=0, return_index=True)
    y_train_unique = y_train[unique_idx]
    train_emb_unique = train_embeddings[unique_idx]
    X_train_flat = X_train_unique.reshape(X_train_unique.shape[0], -1)

    predictor = Predict()
    test_emb = base_model.predict(X_test)

    # accuracy
    acc, preds, _ = predictor.predict_class(
        train_embeddings,
        y_train,
        test_emb,
        y_test,
        base_index
    )

    # cnf and clf
    cm = confusion_matrix(y_test, preds)
    report = classification_report(y_test, preds)
    log.info(f"FSL Accuracy: {100 * acc:.2f}%")
    log.info(f"FSL Classification Report:\n{report}")
    log.info(f"FSL Confusion Matrix:\n{cm}")
    save_confusion_matrix(results_dir, 'fsl_cm', cm, label_map)
    save_classification_report(results_dir, 'fsl_report', y_test, preds, label_map)

    # silhouette
    silhouette = SilhouetteMetric(label_map)
    silhouette.silhouette_overall(flatten(X_test), y_test)
    silhouette.silhouette_overall(train_embeddings, y_train)
    silhouette.save_silhouette_per_class_csv(flatten(X_test), y_test, Path(results_dir) / 'silhouette_base_input.csv')
    silhouette.save_silhouette_per_class_csv(train_embeddings, y_train, Path(results_dir) / 'silhouette_base_embed.csv')

    # visualise embeddings
    vis = VisualiseModelEmbeddings(
        results_dir,
        X_train_flat,
        train_emb_unique,
        X_test,
        test_emb,
        label_map,
        y_train_unique,
        y_test,
        preds
    )
    vis.pacmap()

    all_dists = cdist(test_emb, train_embeddings, metric='euclidean')
    max_class_dists = defaultdict(float)

    for i, (true, pred) in enumerate(zip(y_test, preds)):
        if pred == true:
            class_idx = np.where(y_train == pred)[0]
            if class_idx.size == 0:
                continue
            dists = all_dists[i][class_idx]
            max_class_dists[str(int(pred))] = max(max_class_dists[str(int(pred))], np.max(dists))

    thresholds = dict(max_class_dists)

    return {
        "accuracy": acc,
        "report": report,
        "confusion_matrix": cm,
        "thresholds": thresholds
    }