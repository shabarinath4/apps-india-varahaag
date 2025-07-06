import os
from pathlib import Path
import yaml
import argparse

from src.data.varaha_india_gee import VarahaIndia

mode = 'inference'
plot_level_train = False
base_path = Path(os.getcwd())

def parse_arguments():
    parser = argparse.ArgumentParser(prog='DeCIFR', description='Crop Type Mapper')
    parser.add_argument('--path_ids', type=str, default=['india_250620_ft'])
    parser.add_argument('--inf_in', type=str, default='india_250620_inf_in')
    parser.add_argument('--inf_out', type=str, default='india_250620_inf_out')
    return parser.parse_args()

args = parse_arguments()
inf_in = getattr(args, 'inf_in', 'india_250620_inf_in')
inf_out = getattr(args, 'inf_out', 'india_250620_inf_out')
mode = 'inference'

with open(base_path / "region_config.yaml", "r") as f:
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
inf_obj = VarahaIndia(
        base_path,
        inf_in,
        results_dir, 
        model_dir, 
        mode, 
        plot_level_train, 
        MASK_VALUE
    )

inf_obj.deploy_Varaha_S1_S2_fsl(
    inf_in_path,
    inf_out_path
)