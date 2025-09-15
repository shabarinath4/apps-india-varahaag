import os
import json
import logging
import numpy as np
import pandas as pd

import numpy as np
import faiss
import joblib

import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
from tensorflow.keras.models import load_model

from sklearn.utils import resample
from sklearn.preprocessing import LabelEncoder

from src.models.fsl3 import Train
from src.loader import load_and_concat, make_ml_ready
from src.models.evaluate import evaluate_base_model, evaluate_finetuned_model
from src.utils.plots import Plots
from sklearn.utils import resample
import numpy as np

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Setup for temporal pyramid pooling layer
class TemporalPyramidPooling(tf.keras.layers.Layer):
    def __init__(self, pyramid_levels=[1, 2], **kwargs):
        super().__init__(**kwargs)
        self.pyramid_levels = pyramid_levels

    def call(self, t):
        def pool_level(t, n):
            time_len = tf.shape(t)[1]
            boundaries = tf.cast(tf.linspace(0., tf.cast(time_len, tf.float32), n + 1), tf.int32)
            pooled_slices = []
            for i in range(n):
                start = boundaries[i]
                end = boundaries[i + 1]
                slice_ = t[:, start:end, ...]
                pooled = tf.reduce_max(slice_, axis=1)
                pooled_slices.append(pooled)
            return tf.concat(pooled_slices, axis=-1)

        pooled_levels = [pool_level(t, n) for n in self.pyramid_levels]
        return tf.concat(pooled_levels, axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({"pyramid_levels": self.pyramid_levels})
        return config
            
def sample_multiple(X_list, y, sampling_technique, sampling_mode, n=None, random_state=42):
    """
    Resample multiple X arrays and one y array consistently using the 'Resample' technique.
    Assumes all X arrays have the same number of rows (samples) as y.
    """

    if sampling_technique != 'Resample':
        raise ValueError("Currently sample_multiple supports only 'Resample' technique")

    unique, counts = np.unique(y, return_counts=True)
    class_counts = dict(zip(unique, counts))

    resampled_Xs = [[] for _ in X_list]
    resampled_y = []

    for cls in unique:
        mask = (y == cls)
        cls_count = class_counts[cls]
        cls_indices = np.where(mask)[0]

        if sampling_mode == 'majority':
            n_samples = max(class_counts.values())
        elif sampling_mode == 'minority':
            n_samples = min(class_counts.values())
        elif sampling_mode == 'median':
            n_samples = int(np.median(list(class_counts.values())))
        elif sampling_mode == 'fixed':
            if n is None:
                raise ValueError("Specify a fixed number to sample")
            n_samples = n
        elif sampling_mode == 'min200':
            # If a crop/class has fewer than 200 samples, resample to 200
            # Otherwise, leave it as-is
            n_samples = cls_count if cls_count >= 200 else 200
        else:
            raise ValueError("Invalid sampling_mode")

        # Resample indices for this class
        resampled_indices = resample(
            cls_indices,
            n_samples=n_samples,
            replace=True if n_samples > cls_count else False,  # only replace if upsampling
            random_state=random_state
        )

        for i, X in enumerate(X_list):
            resampled_Xs[i].append(X[resampled_indices])
        resampled_y.append(np.full(shape=n_samples, fill_value=cls))

    resampled_Xs = [np.vstack(parts) for parts in resampled_Xs]
    resampled_y = np.hstack(resampled_y)

    return (*resampled_Xs, resampled_y)


def test_no_leakage(idx_train_bt, idx_test_ft):
    intersection_train = set(idx_train_bt).intersection(set(idx_test_ft))
    if intersection_train:
        print(f"Leakage detected: finetune TEST indices overlap with base TRAIN indices: {intersection_train}")
    else:
        print("No leakage: finetune TEST and base TRAIN sets are disjoint.")

def concat_dict(data_dict):
    shapes = {k: data_dict[k].shape for k in data_dict.keys()}
    knw_broadcast = np.repeat(data_dict['knw'][:, None, :], data_dict['train'].shape[1], axis=1)
    merged = np.concatenate([data_dict['train'], knw_broadcast], axis=-1)

def train_model(**kwargs):
    
    use_3d = True
    mode = 'train'
    training_params = {}

    # base path
    from pathlib import Path
    base_path = Path(os.getcwd())

    # experiment details
    experiment = kwargs.get('experiment', {})
    exp_id = experiment.get('exp_id')
    model_id = experiment.get('model_id')

    # data
    data = kwargs.get('data', {})
    resample_data = data.get('resample_data')
    plot_level_train = data.get('plot_level_train')
    ORIGINAL_INTERVAL_DAYS = data.get('original_interval_days')
    TARGET_INTERVAL_DAYS = data.get('target_interval_days')
    comb_per_sample = data.get('comb_per_sample')
    MASK_VALUE = data.get('mask_value')

    # model
    model = kwargs.get('model', {})
    use_pretrained_model = model.get('use_pretrained_model')
    batch_size = model.get('batch_size')
    bt_epochs = model.get('bt_epochs')
    ft_epochs = model.get('ft_epochs')
    min_delta = model.get('min_delta')
    patience = model.get('patience')
    k_shot_value = model.get('k_shot_value')
   
    # in paths
    in_paths = kwargs.get('in_paths', {})
    bt_path_ids = in_paths.get('bt_path_ids')
    ft_path_ids = in_paths.get('ft_path_ids')
    fs_path_ids = in_paths.get('fs_path_ids')
    model_dir_bt = in_paths.get('model_dir_bt')

    # out paths
    out_paths = kwargs.get('out_paths', {})
    model_dir = out_paths.get('model_dir')
    results_dir = out_paths.get('results_dir')

    log_folder = f"logs/{exp_id}"
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)

    if bt_path_ids is not None:
        # =======================
        # Base Training
        # =======================  
        # Load basetraining data
        X_bt, y_bt, ids_all, features_bt, features_idx, features_bt_idx, domains, outlier_bt_list, normalise_bt_list = load_and_concat(
            base_path, bt_path_ids, results_dir, model_dir, mode, plot_level_train, MASK_VALUE
        )

        X_train_3d_bt, X_test_3d_bt, X_train_input_bt, X_test_input_bt, \
        X_train_model_bt, X_test_model_bt, y_train_bt, y_test_bt, \
        idx_train_bt, idx_test_bt = make_ml_ready(X_bt, y_bt, ids_all, features_bt, use_3d)

        le_bt = LabelEncoder()
        le_bt.fit(y_train_bt)
        y_train_bt = le_bt.transform(y_train_bt)
        y_test_bt  = le_bt.transform(y_test_bt)

        # Sample data
        if resample_data:
            log.info(f"Before sampling:")
            log.info(f"{pd.Series(y_train_bt).value_counts()}")
            X_train_input_bt, X_train_3d_bt, X_train_model_bt, y_train_bt = sample_multiple(
                [X_train_input_bt, X_train_3d_bt, X_train_model_bt],
                y_train_bt,
                sampling_technique='Resample',
                sampling_mode='min200',
                n=500
            )
            log.info(f"Post sampling:")
            log.info(f"{pd.Series(y_train_bt).value_counts()}")
            
        label_mapping = dict(zip(le_bt.transform(le_bt.classes_), le_bt.classes_))
 
        # Plot signatures
        plot_obj = Plots()
        plot_obj.plot_clusters(
            X_train_3d_bt,
            y_train_bt,
            label_mapping,
            features_bt, 
            features_idx,
            mode="paper",
            RESULTS_DIR=results_dir,
            train_mode="base_train",
            )
        
        # Plot signatures
        plot_obj = Plots()
        plot_obj.plot_clusters(
            X_test_3d_bt,
            y_test_bt,
            label_mapping,
            features_bt, 
            features_idx,
            mode="paper",
            RESULTS_DIR=results_dir,
            train_mode="base_test",
            )
        
        # Start training
        log.info("----Start DECIFR BASE Training----")
        train = Train(
            X_train_model_bt, 
            y_train_bt, 
            X_test_model_bt, 
            y_test_bt, 
            comb_per_sample, 
            training_params
        )
        base_network, prototypes, prototypes_labels, \
        base_embeddings, base_params, base_labels, base_index, _ = train.train_model(
            loss_type='triplet',
            triplet_strategy='standard',
            domain_array=domains,
            episodic=False,
            episodes=3,
            k_way=5,
            n_shot=100,
            n_query=50,
            n_epochs=bt_epochs,
            patience=patience
        )
        
        # evaluate
        base_eval_results = evaluate_base_model(
            base_network, 
            X_test_input_bt, 
            X_test_model_bt, 
            y_test_bt, 
            base_embeddings, 
            base_labels, 
            base_index,
            X_train_input=X_train_input_bt, 
            y_train=y_train_bt,
            results_dir=results_dir,
            model_dir=model_dir,
            le=le_bt
        )
        log.info(f"Xtrain : {X_train_model_bt.shape}")
        log.info(f"ytrain: {y_train_bt.shape}")
        log.info(f"{label_mapping}")
        
        # save models
        with open(model_dir / "bt_max_correct_dists_per_class.json", "w") as f:
            json.dump(base_eval_results["max_correct_dists_per_class"], f, indent=4)
        base_network.save(os.path.join(model_dir, 'model.keras'))
        base_network.save(os.path.join(model_dir, 'model.h5'))
        np.save(os.path.join(model_dir, 'base_embeddings.npy'), base_embeddings)
        np.save(os.path.join(model_dir, 'base_labels.npy'), base_labels)
        faiss.write_index(base_index, os.path.join(model_dir, 'base_index.faiss'))

        joblib.dump(le_bt, model_dir / "le_bt.pkl")
        joblib.dump(features_bt, model_dir / "features_bt.pkl")

    if ft_path_ids is not None:
        # =======================
        # Fine Tune Training
        # =======================  
        # Load fine-tuning data
        X_ft, y_ft, ids_all, features_ft, features_idx, features_ft_idx, le_ft, outlier_ft_list, normalise_ft_list = load_and_concat(
            base_path, ft_path_ids, results_dir, model_dir, mode, plot_level_train, MASK_VALUE
        )

        try:
            idx_test_bt
        except NameError:
            idx_test_bt = None

        # Prepare ML-ready data
        X_train_3d_ft, X_test_3d_ft, X_train_input_ft, X_test_input_ft, \
        X_train_model_ft, X_test_model_ft, y_train_ft, y_test_ft, \
        idx_train_ft, idx_test_ft = make_ml_ready(
            X_ft, y_ft, ids_all, features_ft, use_3d,
            link_to_bt_test_idx=idx_test_bt,
            idx=None
        )
        # Sample data
        if resample_data:
            log.info(f"Before sampling:")
            log.info(f"{pd.Series(y_train_ft).value_counts()}")
            X_train_input_ft, X_train_3d_ft, X_train_model_ft, y_train_ft = sample_multiple(
                [X_train_input_ft, X_train_3d_ft, X_train_model_ft],
                y_train_ft,
                sampling_technique='Resample',
                sampling_mode='fixed',
                n=500
            )
            log.info(f"Post sampling:")
            log.info(f"{pd.Series(y_train_ft).value_counts()}")
        label_mapping = dict(zip(le_ft.transform(le_ft.classes_), le_ft.classes_))
        
        # Plot clusters for visualization
        plot_obj = Plots()
        plot_obj.plot_clusters(
            X_train_3d_ft,
            y_train_ft,
            label_mapping,
            features_ft,
            features_idx,
            mode="paper",
            RESULTS_DIR=results_dir,
            train_mode="ft",
        )

        # Train (fine-tune) model using all training data
        train = Train(
            X_train_model_ft, 
            y_train_ft, 
            X_test_model_ft, 
            y_test_ft, 
            comb_per_sample, 
            training_params
        )

        try:
            if model_dir_bt:
                model_path = Path(model_dir_bt) / "model.h5"
                base_network = load_model(
                    model_path,
                    custom_objects={'TemporalPyramidPooling': TemporalPyramidPooling}
                )
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")

        if 'base_network' not in locals():
            raise RuntimeError("base_network is not initialized or check if model_dir_bt is set correctly.")

        ft_model, _, _, X_train_used_ft, ft_embeddings, ft_params, ft_labels, ft_index, _ = train.finetune(
            base_network=base_network,
            support_x=X_train_model_ft,
            support_y=y_train_ft,
            loss_type='triplet',
            n_epochs=ft_epochs,
            batch_size=batch_size,
            freeze_base=False
        )

        # Evaluate on held-out test data
        finetune_eval_results = evaluate_finetuned_model(
            ft_model,
            X_train_model_ft,
            y_train_ft,
            X_test_model_ft,
            y_test_ft,
            ft_embeddings,
            ft_labels,
            ft_index,
            results_dir,
            model_dir,
            le_ft
        )


        # save models
        with open(model_dir / "ft_max_correct_dists_per_class.json", "w") as f:
            json.dump(finetune_eval_results["max_correct_dists_per_class"], f, indent=4)
        ft_model.save(os.path.join(model_dir, 'finetune_model.keras'))
        ft_model.save(os.path.join(model_dir, 'finetune_model.h5'))
        np.save(os.path.join(model_dir, 'finetune_embeddings.npy'), ft_embeddings)
        np.save(os.path.join(model_dir, 'finetune_labels.npy'), ft_labels)
        faiss.write_index(ft_index, os.path.join(model_dir, 'finetune_index.faiss'))

        joblib.dump(le_ft, model_dir / "le_ft.pkl")
        joblib.dump(features_ft, model_dir / "features_ft.pkl")
    
    if fs_path_ids is not None:
        # =======================
        # Few-Shot Inference
        # =======================
        # Load few-shot dataset
        X_fs, y_fs, ids_all_fs, features_fs, features_idx_fs, features_fs_idx, le_fs, outlier_fs_list, normalise_fs_list = load_and_concat(
            base_path, fs_path_ids, results_dir, model_dir, mode, plot_level_train, MASK_VALUE
        )

        # Prepare ML-ready data
        X_train_3d_fs, X_test_3d_fs, X_input_train_fs, X_input_test_fs, \
        X_model_train_fs, X_model_test_fs, y_train_fs, y_test_fs, \
        idx_train_fs, idx_test_fs = make_ml_ready(
            X_fs, y_fs, ids_all_fs, features_fs, use_3d,
            link_to_bt_test_idx=None,
            idx=None
        )

        # Run few-shot evaluation
        eval_fsl_obj = Evaluate()
        fs_model, _, _, X_used_fs, fs_embeddings, fs_params, fs_labels, fs_index, _ = eval_fsl_obj.few_shot_predict(
            model=ft_model,
            x_data=X_model_train_fs,
            y_data=y_train_fs,
            support_sizes=k_shot_value,
            episodes_per_setting=10,
            verbose=True
        )

        # Evaluate few-shot performance
        fewshot_eval_results = evaluate_finetuned_model(
            fs_model,
            X_used_fs,
            y_train_fs,
            X_model_test_fs,
            y_test_fs,
            fs_embeddings,
            fs_labels,
            fs_index,
            results_dir,
            le_fs
        )

        # Save few-shot outputs
        np.save(os.path.join(model_dir, 'fewshot_embeddings.npy'), fs_embeddings)
        np.save(os.path.join(model_dir, 'fewshot_labels.npy'), fs_labels)
        faiss.write_index(fs_index, os.path.join(model_dir, 'fewshot_index.faiss'))
        joblib.dump(le_fs, model_dir / "le_fs.pkl")
        joblib.dump(features_fs, model_dir / "features_fs.pkl")
