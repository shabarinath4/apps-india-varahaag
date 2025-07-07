shared_base_config: &base_config
  UID: '.geo'
  label_columns: ['crop_name', 'cropname']
  plot_level_train: true
  total_ts: 16
  ts_slice_idx: [8, 16]
  remove_bands: ["B1", "B10"]
  crops_to_drop: []
  min_sample_per_class: 5
  group_crops: true
  crops_to_keep: ["paddy_(transplanting)", "paddy_(dsr)", "cotton"]
  outlier_per_timestamp: false
  normalise_strategy: "IQR"
  band_name_map:
    'VH_dB': 'VH_dB'
    'VV_dB': 'VV_dB'
    'B2': 'B2' 
    'B3': 'B3' 
    'B4': 'B4' 
    'B5': 'B5' 
    'B6': 'B6' 
    'B7': 'B7' 
    'B8': 'B8' 
    'B8A': 'B8A' 
    'B9': 'B9' 
    'B11': 'B11' 
    'B12': 'B12'
    'REP': 'REP'
  VIs: {"NDVI":[8, 4]}
  EXP_VER: "99"
  MODEL_TYPE: "fsl"
  model_dir_bt: null
  batch_size: 64
  bt_epochs: 100
  ft_epochs: 100
  min_delta: 0.0001
  patience: 10
  use_pretrained_model: false
  k_shot_value: [1, 5, 10, 'all']
  ORIGINAL_INTERVAL_DAYS: 20
  TARGET_INTERVAL_DAYS: 10
  MASK_VALUE: !!float .nan
  comb_per_sample: 25
  resample_data: true

india_250620_bt:
  <<: *base_config
  path: "data/train_ready/india_250620_bt"

india_250620_ft:
  <<: *base_config
  path: "data/train_ready/india_250620_ft"

india_250620_inf_in:
  <<: *base_config
  path: "data/train_ready/india_250620_inf_in"
  results_dir: 'results/trials/fsl/99_fsl_20250706-233404_554d0145-3798-41c3-93f6-90a3ec1db4f3'
  model_dir: 'saved_models/trials/fsl/99_fsl_20250706-233404_554d0145-3798-41c3-93f6-90a3ec1db4f3'
