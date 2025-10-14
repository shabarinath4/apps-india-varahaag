import os
from pathlib import Path
import yaml
import argparse

from src.data.regional.india_varaha import IndiaVaraha
from pathlib import Path

plot_level_train = False
base_path = Path(os.getcwd())
model_clf_strat={'rfc_raw':None, 'cnn_raw':None, 'vanilla':'rfc', 'class_balanced':'rfc', 'domain_aware':'rfc'}

def parse_arguments():
    parser = argparse.ArgumentParser(prog='DeCIFR', description='Crop Type Mapper')
    parser.add_argument('--inf_in', type=str, default='india_250909_inf_in')
    parser.add_argument('--inf_out', type=str, default='india_250909_inf_out')
    return parser.parse_args()

args = parse_arguments()
inf_in = getattr(args, 'inf_in', 'india_250909_inf_in')
inf_out = getattr(args, 'inf_out', 'india_250909_inf_out')
mode = 'inference'

with open(base_path / "configs" /"india_varaha.yaml", "r") as f:
    dataset_meta = yaml.safe_load(f)

meta = dataset_meta[inf_in]
model_dir = base_path / meta["model_dir"]
results_dir = base_path / meta["results_dir"]
MASK_VALUE = meta["MASK_VALUE"]
inf_in_path = base_path / "data" / "train_ready" / inf_in
inf_out_path = base_path / "data" / "train_ready" / inf_out

if not os.path.exists(inf_out_path):
    os.makedirs(inf_out_path)

# =======================
# Fine Tune Training
# =======================  
inf_obj = IndiaVaraha(
        base_path,
        inf_in,
        results_dir, 
        model_dir, 
        mode, 
        plot_level_train, 
        MASK_VALUE
    )

inf_obj.deploy_Varaha_S1_S2_fsl(
    model_dir,
    inf_in_path,
    inf_out_path,
    model_clf_strat
)