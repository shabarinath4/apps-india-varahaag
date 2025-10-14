import geopandas as gpd
import numpy as np
import pandas as pd

meta = gpd.read_file(r'D:\Users\shaba\Desktop\1.1 Crop ID - Domain Adaptation\decifr-ag\data\train_ready\malawi_250812_v4_ft\harmonized_malawi_3857_v4.geojson')
v4_ids = pd.read_csv(r"D:\Users\shaba\Desktop\1.1 Crop ID - Domain Adaptation\decifr-ag\data\train_ready\malawi_250812_v4_ft\v4_duplicate_field_ids.csv")
meta = meta[meta['Field_Id'].isin(list(v4_ids['Field_Id_2']))]
print(meta[meta['Intercrop']=='Soybean']['Year'])
