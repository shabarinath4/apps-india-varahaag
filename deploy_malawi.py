from src.data.regional.malawi_cimmyt import MalawiCIMMYT
from pathlib import Path

base_path = Path(r"D:\Users\shaba\Desktop\1.1 Crop ID - Domain Adaptation\decifr-ag")
in_dir = Path(r"D:\Users\shaba\Desktop\1.1 Crop ID - Domain Adaptation\decifr-ag\data\train_ready\malawi_250812_v4_ft\inference_test\192053c\datacube.npy")
out_dir = Path(r"D:\Users\shaba\Desktop\1.1 Crop ID - Domain Adaptation\decifr-ag\data\train_ready\malawi_250812_v4_ft\inference_test\192053c\output_rfc.tif")
MODEL_DIR = Path(r"D:\Users\shaba\Desktop\1.1 Crop ID - Domain Adaptation\decifr-ag\saved_models\production\fsl\99_fsl_20251005-193827_465bcc73-f723-447e-8797-8401c522c02c")
model_clf_strat={'rfc_raw':None, 'cnn_raw':None, 'vanilla':'rfc', 'class_balanced':'rfc', 'domain_aware':'rfc'}

# base_path = Path(r"/gpfs/data1/cmongp1/nair/decifr")
# MODEL_DIR = Path(r"/gpfs/data1/cmongp1/nair/decifr/saved_models/production/fsl/...")

load_region = MalawiCIMMYT(
    base_path=base_path,
    mode='inference',
    path_id='malawi_250812_v4_ft',
    mask_value=65535)

data = load_region.deploy_MalawiCIMMYT_planet_fsl(
    MODEL_DIR,
    in_dir,
    out_dir,
    model_clf_strat
)