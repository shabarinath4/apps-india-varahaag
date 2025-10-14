import pandas as pd
import numpy as np
import os

csv_path = r"D:\Users\shaba\Desktop\1.1 Crop ID - Domain Adaptation\decifr-ag\data\knowledge\spam\spam2020V2r0_global_harvested_area.csv\spam2020V2r0_global_harvested_area\spam2020V2r0_global_H_TA.csv"
df = pd.read_csv(csv_path)

crop_cols = {
    'wheat': 'WHEA_A',
    'maize': 'MAIZ_A',
    'rapeseed': 'RAPE_A',
    'millet': 'MILL_A',
    'sorghum': 'SORG_A',
    'rice': 'RICE_A',
    'soybean': 'SOYB_A',
    'sugarcane': 'SUGB_A'
}

sample_sizes = {1: 20, 2: 30, 3: 50}

df = df[df['ADM0_NAME'].isin([
    'France', 'Ukraine', 'Canada', 'United States of America',
    'China', 'India', 'Zambia', 'Brazil'
])].reset_index(drop=True)

out_dir = r"D:\Users\shaba\Desktop\1.1 Crop ID - Domain Adaptation\decifr-ag\data\knowledge\spam\crop_samples"
os.makedirs(out_dir, exist_ok=True)

for crop, col in crop_cols.items():
    all_samples = []
    cols_to_keep = ['ADM0_NAME', 'x', 'y', 'grid_code', col]
    crop_df = df[cols_to_keep].dropna()
    for country, country_df in crop_df.groupby('ADM0_NAME'):
        country_df = country_df[country_df[col] > 0].copy()
        if country_df.empty:
            continue
        try:
            country_df['group'] = pd.qcut(
                country_df[col],
                q=3,
                labels=[1, 2, 3],
                duplicates='drop'
            )
        except ValueError:
            continue
        for group_label, n_samples in sample_sizes.items():
            group_df = country_df[country_df['group'] == group_label]
            if len(group_df) > n_samples:
                group_sample = group_df.sample(n_samples, random_state=42)
            else:
                group_sample = group_df
            group_sample['crop'] = crop
            all_samples.append(group_sample)
    if all_samples:
        final_samples = pd.concat(all_samples, axis=0)
        out_path = os.path.join(out_dir, f"{crop}_samples.csv")
        final_samples.to_csv(out_path, index=False)
        print(f"Saved {crop} samples → {out_path}")
