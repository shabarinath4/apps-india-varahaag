# import tensorflow as tf
# tf.config.set_visible_devices([], 'GPU')
import os
import argparse
import numpy as np
import pandas as pd
from functools import partial
import warnings
import uuid
from tqdm import tqdm
import datetime
import uuid
import yaml
import argparse
import uuid

from train import train_model
from src.utils.utilities import (FileOperations, VisualiseModelEmbeddings, SilhouetteMetric, PlotCnfClf)

# Add relevant paths
warnings.filterwarnings('ignore')

# Convert string to boolean
def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', 'no', 'n', '0'}:
        return False
    elif value.lower() in {'true', 't', 'yes', 'y', '1'}:
        return True
    else:
        raise argparse.ArgumentTypeError(f"Boolean value expected. Got {value}")

def create_directories(model_dir, results_dir):
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(os.path.join(results_dir, 'params'), exist_ok=True)
    os.makedirs(os.path.join(results_dir, 'performance'), exist_ok=True)
    
def str_to_bool(v):
    return v.lower() in ('yes', 'true', 't', '1')

def create_directories(*dirs):
    for directory in dirs:
        directory.mkdir(parents=True, exist_ok=True)

def parse_arguments():
    parser = argparse.ArgumentParser(prog='DeCIFR', description='Crop Type Mapper')
    parser.add_argument('--bt_path_ids', type=str, default=None)
    parser.add_argument('--ft_path_ids', type=str, default=None)
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    # in
    bt_path_ids = getattr(args, 'bt_path_ids', None)
    ft_path_ids = getattr(args, 'ft_path_ids', None)

    # Load YAML metadata for all datasets
    from pathlib import Path
    base_path = Path(os.getcwd())
    with open(base_path / "region_config.yaml", "r") as f:
        dataset_meta = yaml.safe_load(f)
    meta = None
    print(dataset_meta)
    if bt_path_ids:
        try:
            bt_path_ids = [item.strip() for item in bt_path_ids.split(",")]
            meta = dataset_meta[bt_path_ids[0]]
        except (KeyError, IndexError):
            pass

    if ft_path_ids:
        try:
            ft_path_ids = [item.strip() for item in ft_path_ids.split(",")]
            meta = dataset_meta[ft_path_ids[0]]
        except (KeyError, IndexError):
            pass

    if not meta:
        raise ValueError("No valid path ID found in bt_path_ids or ft_path_ids.")

    # in
    model_dir_bt = meta["model_dir_bt"]
    
    # data and preprocess
    ORIGINAL_INTERVAL_DAYS = meta["ORIGINAL_INTERVAL_DAYS"]
    TARGET_INTERVAL_DAYS = meta["TARGET_INTERVAL_DAYS"]
    MASK_VALUE = meta["MASK_VALUE"]
    plot_level_train = meta["plot_level_train"]
    comb_per_sample = meta["comb_per_sample"]
    resample_data = meta["resample_data"]

    # model
    EXP_VER = meta["EXP_VER"]
    MODEL_TYPE = meta["MODEL_TYPE"]
    batch_size = meta["batch_size"]
    bt_epochs = meta["bt_epochs"]
    ft_epochs = meta["ft_epochs"]
    k_shot_value = meta["k_shot_value"]
    min_delta = meta["min_delta"]
    patience = meta["patience"]
    use_pretrained_model = meta["use_pretrained_model"]

    # out
    saved_models_root = Path("saved_models") / "trials" / MODEL_TYPE
    results_root = Path("results") / "trials" / MODEL_TYPE

    # Generate one unique experiment ID string 
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    unique_id = uuid.uuid4()
    exp_id = f"{EXP_VER}_{MODEL_TYPE}_{timestamp}_{unique_id}"
    
    model_dir = saved_models_root / exp_id
    results_dir = results_root / exp_id

    # Create necessary directories 
    create_directories(model_dir, results_dir)

    train_params = {
        'experiment': {
            'exp_id': exp_id,
            'model_id': exp_id,
        },

        'data': {
            'resample_data': resample_data,
            'plot_level_train': plot_level_train,
            'original_interval_days': ORIGINAL_INTERVAL_DAYS,
            'target_interval_days': TARGET_INTERVAL_DAYS,
            'comb_per_sample': comb_per_sample,
            'mask_value': MASK_VALUE
        },

        'model': {
            'use_pretrained_model': use_pretrained_model,
            'batch_size': batch_size,
            'bt_epochs': bt_epochs,
            'ft_epochs': ft_epochs,
            'min_delta': min_delta,
            'patience': patience,
            'k_shot_value': k_shot_value,
        },

        'in_paths': {
            'bt_path_ids': bt_path_ids,
            'ft_path_ids': ft_path_ids,
            'model_dir_bt': model_dir_bt,
        },

        'out_paths': {
            'model_dir': model_dir,
            'results_dir': results_dir
        },
    }
    
    train_model(**train_params)
        
if __name__ == "__main__":
    
    main()
    