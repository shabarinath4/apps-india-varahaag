import os
import json
import numpy as np

from typing import Any, Dict, Tuple
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model

from src.data.varaha_india_gee import VarahaIndia

import logging
log = logging.getLogger(__name__)

def load_region_data(
    base_path: str,
    path_id: str, 
    results_dir: str, 
    model_dir: str, 
    mode: str, 
    plot_level_train: Any,
    mask_value: int
) -> Any:
    """
    Loads the dataset for the specified region.

    Parameters:
        path_id (str): Identifier for the region ('malawi', 'india', 'ethiopia').
        results_dir (str): Directory path for results.
        model_dir (str): Directory path for the model.
        mode (str): Mode of operation.
        PLOT_LEVEL_TRAIN (Any): Parameter used for training plot level in 'india' region.

    Returns:
        Any: Loaded dataset object or data for the specified region.
    """

    if 'malawi' in path_id:
        load_region = CIMMYTMalawi(
            base_path,
            path_id, 
            results_dir, 
            model_dir, 
            mode, 
            plot_level_train, 
            mask_value
        )
        return load_region.load_CIMMYTMalawi_planet()
    
    elif 'india' in path_id:
        load_region = VarahaIndia(
            base_path,
            path_id, 
            results_dir, 
            model_dir, 
            mode, 
            plot_level_train, 
            mask_value
        )
        return load_region.load_varaha_india_S1_S2()

    elif 'ethiopia' in path_id:
        load_region = CIMMYTEthiopia(
            base_path,
            path_id, 
            results_dir, 
            model_dir, 
            mode,
            plot_level_train, 
            mask_value
        )
        return load_region.load_cimmyt_ethiopia_S1_S2()

# # =======================
# # RUN DATA DRIVEN OR KNOWLEDGE DRIVEN MODELS
# # =======================
# def load_and_concat(
#         base_path,
#         path_ids, 
#         results_dir, 
#         model_dir, 
#         mode, 
#         plot_level_train, 
#         mask_value
# ):
#     from collections import defaultdict
#     x_dict_all = defaultdict(list)
#     y_list, idx_list, ids_list = [], [], []
#     outlier_dicts = []
#     normalise_dicts = []
#     features = None

#     import numpy as np
#     for i, pid in enumerate(path_ids):
#         X, y, ids, idx, features_tmp, features_idx, outlier_dict, normalise_dict = load_region_data(
#             base_path,
#             pid, 
#             results_dir, 
#             model_dir, 
#             mode, 
#             plot_level_train, 
#             mask_value
#         )

#         for k,v in X.items():
#             x_dict_all[k].append(v)

#         y_list.append(y)
#         unique_idx = np.array([f"{i}_{x}" for x in idx])
#         idx_list.append(unique_idx)
#         ids_list.append(ids)
#         outlier_dicts.append(outlier_dict)
#         normalise_dicts.append(normalise_dict)

#         if features is None:
#             features = features_tmp

#     for k in x_dict_all:
#         x_dict_all[k] = np.concatenate(x_dict_all[k], axis=0)

#     y_all = np.concatenate(y_list, axis=0)
#     idx_all = np.concatenate(idx_list, axis=0)
#     ids_all = np.concatenate(ids_list, axis=0)

#     le = LabelEncoder()
#     y_all = le.fit_transform(y_all)

#     return x_dict_all, y_all, ids_all, features, features_idx, idx_all, le, outlier_dicts, normalise_dicts

# =======================
# RUN DATA DRIVEN OR KNOWLEDGE DRIVEN MODELS
# =======================
def load_and_concat(
        base_path,
        path_ids, 
        results_dir, 
        model_dir, 
        mode, 
        plot_level_train, 
        mask_value
):
    from collections import defaultdict
    x_list, y_list, idx_list, ids_list = [], [], [], []
    outlier_dicts = []
    normalise_dicts = []
    features = None

    import numpy as np
    for i, pid in enumerate(path_ids):
        X, y, ids, idx, domains, features_tmp, features_idx, outlier_dict, normalise_dict = load_region_data(
            base_path,
            pid, 
            results_dir, 
            model_dir, 
            mode, 
            plot_level_train, 
            mask_value
        )

        x_list.append(X)
        y_list.append(y)
        unique_idx = np.array([f"{i}_{x}" for x in idx])
        idx_list.append(unique_idx)
        ids_list.append(ids)
        outlier_dicts.append(outlier_dict)
        normalise_dicts.append(normalise_dict)

        if features is None:
            features = features_tmp

    x_all = np.concatenate(x_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)
    idx_all = np.concatenate(idx_list, axis=0)
    ids_all = np.concatenate(ids_list, axis=0)

    # le = LabelEncoder()
    # y_all = le.fit_transform(y_all)

    return x_all, y_all, ids_all, features, features_idx, domains, idx_all, outlier_dicts, normalise_dicts


def load_existing_model(
    MODEL_DIR: str
) -> Tuple[Any, np.ndarray, np.ndarray, Dict, Dict, Dict]:
    """
    Loads a pre-trained model and associated metadata from disk.

    Parameters:
        MODEL_DIR (str): Directory containing the model files.

    Returns:
        Tuple containing:
            - base_network: Loaded Keras model.
            - embeddings (np.ndarray): Embeddings array.
            - embeddings_labels (np.ndarray): Labels for embeddings.
            - outlier_dict (dict): Dictionary with outlier information.
            - normalise_dict (dict): Normalisation parameters.
            - runtime_params (dict): Training parameters and runtime config.
    """
    embeddings = np.load(os.path.join(MODEL_DIR, 'embeddings.npy'))
    embeddings_labels = np.load(os.path.join(MODEL_DIR, 'embeddings_labels.npy'))
    base_network = load_model(os.path.join(MODEL_DIR, 'model.keras'))

    with open(os.path.join(MODEL_DIR, 'outlier_dict.json'), 'r') as file:
        outlier_dict = json.load(file)
    with open(os.path.join(MODEL_DIR, 'normalise_dict.json'), 'r') as file:
        normalise_dict = json.load(file)
    with open(os.path.join(MODEL_DIR, 'training_parameters.json'), 'r') as file:
        runtime_params = json.load(file)
    
    return base_network, embeddings, embeddings_labels, outlier_dict, normalise_dict, runtime_params 

def make_ml_ready(
    X, y, ids_all, features, use_3d,
    link_to_bt_test_idx=None,
    idx=None
):
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.utils import shuffle
    import logging

    log = logging.getLogger(__name__)
    if idx is None:
        idx = np.arange(len(y))

    log.info(f"Original value counts {np.unique(y, return_counts=True)}")

    if link_to_bt_test_idx is None:
        # Grouped split: ensure same id does not go to both train and test
        unique_ids = np.unique(ids_all)

        # One label per id (e.g., majority or first label)
        id_to_label = {}
        for uid in unique_ids:
            id_indices = np.where(ids_all == uid)[0]
            id_labels = y[id_indices]
            if len(np.unique(id_labels)) > 1:
                label = np.bincount(id_labels).argmax()
            else:
                label = id_labels[0]
            id_to_label[uid] = label

        id_list = np.array(list(id_to_label.keys()))
        label_list = np.array([id_to_label[uid] for uid in id_list])

        id_train, id_test = train_test_split(
            id_list,
            test_size=0.3,
            random_state=1,
            stratify=label_list if len(np.unique(label_list)) > 1 else None
        )

        train_mask = np.isin(ids_all, id_train)
        test_mask = np.isin(ids_all, id_test)

        X_train, y_train, idx_train = X[train_mask], y[train_mask], idx[train_mask]
        X_test, y_test, idx_test = X[test_mask], y[test_mask], idx[test_mask]

    else:
        test_mask = np.isin(idx, link_to_bt_test_idx)
        train_mask = ~test_mask
        X_train, y_train, idx_train = X[train_mask], y[train_mask], idx[train_mask]
        X_test, y_test, idx_test = X[test_mask], y[test_mask], idx[test_mask]

    # Remove NaNs
    train_mask_nan = ~np.isnan(X_train).any(axis=tuple(range(1, X_train.ndim)))
    X_train, y_train, idx_train = X_train[train_mask_nan], y_train[train_mask_nan], idx_train[train_mask_nan]

    test_mask_nan = ~np.isnan(X_test).any(axis=tuple(range(1, X_test.ndim)))
    X_test, y_test, idx_test = X_test[test_mask_nan], y_test[test_mask_nan], idx_test[test_mask_nan]

    # Shuffle training data
    X_train, y_train, idx_train = shuffle(X_train, y_train, idx_train, random_state=42)

    def to_3d(X, features):
        if X.ndim == 2:
            timestamps = sorted(set(int(f.split('_')[0]) for f in features))
            bands = sorted(set(f.split('_')[1] for f in features))
            return X.reshape(-1, len(timestamps), len(bands))
        elif X.ndim == 3:
            return X
        else:
            raise ValueError(f"Unexpected X ndim={X.ndim}")

    X_train_3d = to_3d(X_train, features)
    X_test_3d = to_3d(X_test, features)

    X_train_2d = X_train_3d.reshape(X_train_3d.shape[0], -1)
    X_test_2d = X_test_3d.reshape(X_test_3d.shape[0], -1)

    if use_3d:
        X_train_input = np.expand_dims(X_train_3d, axis=-1)
        X_test_input = np.expand_dims(X_test_3d, axis=-1)
        X_train_model = X_train_input
        X_test_model = X_test_input
    else:
        X_train_input = X_train_2d
        X_test_input = X_test_2d
        X_train_model = X_train_2d
        X_test_model = X_test_2d

    log.info(f"X_train_3d: {X_train_3d.shape}")
    log.info(f"y_train: {y_train.shape}")
    log.info(f"X_test_3d: {X_test_3d.shape}")
    log.info(f"Original value counts {np.unique(y, return_counts=True)}")

    return (
        X_train_3d, X_test_3d,
        X_train_input, X_test_input,
        X_train_model, X_test_model,
        y_train, y_test,
        idx_train, idx_test
    )

