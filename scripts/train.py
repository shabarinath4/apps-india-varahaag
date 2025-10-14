import os
import json
import logging
import numpy as np
import pandas as pd
import pickle
import faiss
import joblib
from pathlib import Path

import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
from tensorflow.keras.models import load_model
from tensorflow.keras.models import save_model


from sklearn.utils import resample
from sklearn.preprocessing import LabelEncoder

from src.data.base.preprocess import Preprocess

from src.loader import load_and_concat, make_ml_ready

from src.models.fsl.model import TrainFSL
from src.models.fsl.predict import Predict
from src.models.fsl.evaluate import eval_fsl

from src.models.rfc.model import TrainRFC
from src.models.rfc.predict import PredictRFC
from src.models.rfc.evaluate import eval_rfc

from src.models.cnn.model import TrainCNN
from src.models.cnn.predict import PredictCNN
from src.models.cnn.evaluate import eval_cnn

from src.models.supcon.model import TrainSupCon

from src.utils.plots import Plots
from sklearn.utils import resample

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

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

            resampled_indices = resample(
                cls_indices,
                n_samples=n_samples,
                replace=True if n_samples > cls_count else False,
                random_state=random_state
            )

            for i, X in enumerate(X_list):
                resampled_Xs[i].append(X[resampled_indices])
            resampled_y.append(np.full(shape=n_samples, fill_value=cls))

        resampled_Xs = [np.vstack(parts) for parts in resampled_Xs]
        resampled_y = np.hstack(resampled_y)

        return (*resampled_Xs, resampled_y)
 
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

    if bt_path_ids is not None:
        
        #--------------------------------------------
        # Data Processing
        #--------------------------------------------
        # Load basetraining data
        X_bt, y_bt, ids_all, features_bt, features_idx, features_bt_idx, outlier_bt_list, normalise_bt_list, domains, knw_emb, pre_train_test_split = load_and_concat(
            base_path, bt_path_ids, results_dir, model_dir, mode, plot_level_train, MASK_VALUE
        )

        X_train_3d_bt, X_test_3d_bt, X_train_input_bt, X_test_input_bt, \
        X_train_model_bt, X_test_model_bt, y_train_bt, y_test_bt, \
        idx_train_bt, idx_test_bt = make_ml_ready(X_bt, y_bt, ids_all, features_bt, use_3d, pre_train_test_split=pre_train_test_split)

        le_bt = LabelEncoder()
        le_bt.fit(y_train_bt)
        y_train_bt = le_bt.transform(y_train_bt)
        y_test_bt  = le_bt.transform(y_test_bt)

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

        # --------------------------------------------
        # Plot functions
        # --------------------------------------------
        # Plot signatures
        plot_obj = Plots()
        plot_obj.plot_clusters(X_train_3d_bt, y_train_bt, label_mapping, features_bt, features_idx, mode="paper", RESULTS_DIR=results_dir, train_mode="base_train")
        
        # Plot signatures
        plot_obj = Plots()
        plot_obj.plot_clusters(X_test_3d_bt, y_test_bt, label_mapping, features_bt, features_idx, mode="paper", RESULTS_DIR=results_dir, train_mode="base_test")

        def flatten(x: np.ndarray) -> np.ndarray:
            return x.reshape(x.shape[0], -1) if len(x.shape) > 2 else x

        #--------------------------------------------
        # Make ML Ready
        #--------------------------------------------
        X_train_3d_bt, X_test_3d_bt, X_train_input_bt, X_test_input_bt, \
        X_train_model_bt, X_test_model_bt, y_train_btl, y_test_bt, \
        idx_train_bt, idx_test_bt = make_ml_ready(X_bt, y_bt, ids_all, features_bt, use_3d)

        le_bt = LabelEncoder()
        le_bt.fit(y_train_btl)
        y_train_bt = le_bt.transform(y_train_btl)
        y_test_bt  = le_bt.transform(y_test_bt)
        label_mapping = {i: label for i, label in enumerate(le_bt.classes_)}

        #--------------------------------------------
        # Sup Con
        #--------------------------------------------
        def train_and_save_supcon_model(
            mode_name,
            X_train_model_bt, y_train_bt,
            X_test_model_bt, y_test_bt,
            X_train_input_bt, X_test_input_bt,
            domains, training_params, le_bt,
            model_dir, bt_epochs, patience,
            results_dir, min_samples_per_class_auto=None
        ):
            """
            Train SupCon for a single mode and save model, embeddings, and FAISS index.
            """
            os.makedirs(model_dir, exist_ok=True)
            mode_dir = os.path.join(model_dir, mode_name)
            os.makedirs(mode_dir, exist_ok=True)
            results_dir = os.path.join(results_dir, mode_name)
            os.makedirs(results_dir, exist_ok=True)

            # Compute automatic min_samples_per_class if not provided
            unique_classes, class_counts = np.unique(y_train_bt, return_counts=True)
            if min_samples_per_class_auto is None:
                min_class_count = class_counts.min()
                min_samples_per_class_auto = max(5, min(25, min_class_count))

            batch_size_auto = min_samples_per_class_auto * len(unique_classes)
            batch_size_auto = max(batch_size_auto, 64)
            batch_size_auto = min(batch_size_auto, 512)

            # Mode-specific parameters
            if mode_name == "vanilla":
                balanced_sampling = False
                min_samples_per_class = None
                min_samples_per_domain = None
            elif mode_name == "class_balanced":
                balanced_sampling = True
                min_samples_per_class = min_samples_per_class_auto
                min_samples_per_domain = None
            elif mode_name == "domain_aware":
                balanced_sampling = True
                min_samples_per_class = min_samples_per_class_auto
                min_samples_per_domain = 5
            else:
                raise ValueError(f"Unknown mode: {mode_name}")

            train_supcon = TrainSupCon(
                X_train_model_bt, y_train_bt,
                X_test_model_bt, y_test_bt,
                training_params
            )

            base_network, prototypes, prototypes_labels, \
            base_embeddings, base_params, base_labels, base_index, train_labels = train_supcon.train_model(
                batch_size=batch_size_auto,
                min_epochs=25,                  
                max_epochs=bt_epochs,           
                val_split=0.2,
                patience=patience,
                domain_array=domains,
                min_samples_per_class=min_samples_per_class or 5,
                min_samples_per_domain=min_samples_per_domain or 5,
                batch_mode="domain_aware",
                temperature=0.05
            )


            predict_obj = Predict()
            test_embeddings = base_network.predict(X_test_model_bt)
            
            # Evaluate
            base_eval_results = eval_fsl(
                base_network,
                X_test_input_bt,
                y_test_bt,
                base_embeddings,
                base_index,
                X_train=X_train_input_bt,
                y_train=y_train_bt,
                results_dir=results_dir,
                model_dir=mode_dir,
                label_encoder=le_bt
            )

            # Save models and embeddings
            with open(os.path.join(mode_dir, "thresholds.json"), "w") as f:
                json.dump(base_eval_results["thresholds"], f, indent=4)
            base_network.save(os.path.join(mode_dir, 'model.keras'))
            base_network.save(os.path.join(mode_dir, 'model.h5'))
            np.save(os.path.join(mode_dir, 'base_embeddings.npy'), base_embeddings)
            np.save(os.path.join(mode_dir, 'base_labels.npy'), base_labels)
            np.save(os.path.join(mode_dir, 'test_embeddings.npy'), test_embeddings)
            np.save(os.path.join(mode_dir, 'y_test_bt.npy'), y_test_bt)
            joblib.dump(le_bt, os.path.join(mode_dir, 'le_bt.pkl'))

            faiss.write_index(base_index, os.path.join(mode_dir, 'base_index.faiss'))

            np.save(os.path.join(model_dir, 'X_train_model_bt.npy'), X_train_model_bt)
            np.save(os.path.join(model_dir, 'y_train_bt.npy'), y_train_bt)
            np.save(os.path.join(model_dir, 'X_test_model_bt.npy'), X_test_model_bt)
            np.save(os.path.join(model_dir, 'y_test_bt.npy'), y_test_bt)
            print(f"Saved SupCon {mode_name} model and embeddings to {mode_dir}")

            #--------------------------------------------
            # RFC EMB
            #--------------------------------------------
            # train
            rfc = TrainRFC().rfc(base_embeddings, base_labels)

            # predict
            predict_rfc_obj = PredictRFC()
            y_pred, _ = predict_rfc_obj.predict_rfc(rfc, test_embeddings)

            # eval
            eval_rfc(y_pred, y_test_bt, results_dir, le_bt, identifier='rfc_emb')

            # save
            model_path = os.path.join(mode_dir, "rfc_embeddings.pkl")
            joblib.dump(rfc, model_path)


        train_and_save_supcon_model(
            "vanilla",
            X_train_model_bt, y_train_bt,
            X_test_model_bt, y_test_bt,
            X_train_input_bt, X_test_input_bt,
            domains, training_params, le_bt,
            model_dir, bt_epochs, patience, Path(results_dir)
        )

        train_and_save_supcon_model(
            "class_balanced",
            X_train_model_bt, y_train_bt,
            X_test_model_bt, y_test_bt,
            X_train_input_bt, X_test_input_bt,
            domains, training_params, le_bt,
            model_dir, bt_epochs, patience, Path(results_dir)
        )

        train_and_save_supcon_model(
            "domain_aware",
            X_train_model_bt, y_train_bt,
            X_test_model_bt, y_test_bt,
            X_train_input_bt, X_test_input_bt,
            domains, training_params, le_bt,
            model_dir, bt_epochs, patience, Path(results_dir)
        )

        #--------------------------------------------
        # RFC RAW
        #--------------------------------------------
        # train
        rfc = TrainRFC().rfc(flatten(X_train_model_bt), y_train_bt)

        # predict
        predict_rfc_obj = PredictRFC()
        y_pred, _ = predict_rfc_obj.predict_rfc(rfc, flatten(X_test_model_bt))

        # eval
        eval_rfc(y_pred, y_test_bt, results_dir, le_bt, identifier='rfc_raw')

        # save
        mode_name = 'rfc_raw'
        os.makedirs(model_dir, exist_ok=True)
        mode_dir = os.path.join(model_dir, mode_name)
        os.makedirs(mode_dir, exist_ok=True)
        model_path = os.path.join(mode_dir, "rfc_raw.pkl")
        joblib.dump(rfc, model_path)

        #--------------------------------------------
        # CNN RAW
        #--------------------------------------------
        # train
        cnn = TrainCNN(len(np.unique(y_train_bt)), 50, 64).cnn(np.expand_dims(flatten(X_train_model_bt), axis=-1), y_train_bt)

        # predict
        predict_cnn_obj = PredictCNN()
        y_pred, _ = predict_cnn_obj.predict_cnn(cnn, np.expand_dims(flatten(X_test_model_bt), axis=-1))

        # eval
        eval_cnn(y_pred, y_test_bt, results_dir, le_bt, identifier='cnn_raw')

        # save
        mode_name = 'cnn_raw'
        os.makedirs(model_dir, exist_ok=True)
        mode_dir = os.path.join(model_dir, mode_name)
        os.makedirs(mode_dir, exist_ok=True)

        model_path = os.path.join(mode_dir, "cnn_raw.h5")  
        cnn.save(model_path)  # or save_model(cnn, model_path)

        # #--------------------------------------------
        # # FSL
        # #--------------------------------------------
        # train_fsl = TrainFSL(X_train_model_bt, y_train_bt, X_test_model_bt, y_test_bt, comb_per_sample, training_params)
        # base_network, prototypes, prototypes_labels, \
        # base_embeddings, base_params, base_labels, base_index, _ = train_fsl.train_model(
        #     loss_type='triplet',
        #     triplet_strategy='batch_domainaware',
        #     domain_array=domains,
        #     episodic=False,
        #     episodes=3,
        #     k_way=5,
        #     n_shot=100,
        #     n_query=50,
        #     n_epochs=bt_epochs,
        #     patience=patience)

        # predict_obj = Predict()
        # test_embeddings = base_network.predict(X_test_model_bt)

        # # Evaluate
        # base_eval_results = eval_fsl(
        #     base_network, 
        #     X_test_input_bt, 
        #     y_test_bt, 
        #     base_embeddings, 
        #     base_index,
        #     X_train=X_train_input_bt, 
        #     y_train=y_train_bt,
        #     results_dir=results_dir,
        #     model_dir=model_dir,
        #     label_encoder=le_bt
        # )

        # # save models
        # with open(model_dir / "thresholds.json", "w") as f:
        #     json.dump(base_eval_results["thresholds"], f, indent=4)
        # base_network.save(os.path.join(model_dir, 'model.keras'))
        # base_network.save(os.path.join(model_dir, 'model.h5'))
        # np.save(os.path.join(model_dir, 'base_embeddings.npy'), base_embeddings)
        # np.save(os.path.join(model_dir, 'base_labels.npy'), base_labels)
        # faiss.write_index(base_index, os.path.join(model_dir, 'base_index.faiss'))
