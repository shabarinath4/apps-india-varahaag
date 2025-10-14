import geopandas as gpd
import rasterio
from rasterio.mask import mask
import numpy as np
from collections import Counter
import pandas as pd
from multiprocessing import Pool, cpu_count

def process_geometry(args):
    idx, geometry, raster_path, nodata = args
    try:
        with rasterio.open(raster_path) as src:
            out_image, _ = mask(src, [geometry], crop=True, nodata=nodata)
            data = out_image[0]
            data = data[data != nodata]
            class_counts = dict(Counter(data.flatten()))
            class_counts = {str(k): float(v) for k, v in class_counts.items()}
    except Exception as e:
        print(f"Failed at index {idx}: {e}")
        class_counts = {}
    return idx, class_counts

def count_classes_per_geometry_mp(shapefile_path, raster_path, output_path):
    gdf = gpd.read_file(shapefile_path)
    gdf = gdf.to_crs(epsg=3857)
    with rasterio.open(raster_path) as src:
        nodata = src.nodata

    args_list = [(idx, row['geometry'], raster_path, nodata) for idx, row in gdf.iterrows()]

    results = []
    with Pool(processes=cpu_count()) as pool:
        for i, res in enumerate(pool.imap_unordered(process_geometry, args_list), 1):
            results.append(res)
            if i % 10 == 0 or i == len(args_list):
                print(f"Processed {i} / {len(args_list)} geometries")

    results.sort(key=lambda x: x[0])
    class_counts_list = [r[1] for r in results]
    class_set = sorted(set().union(*[d.keys() for d in class_counts_list]))

    counts_df = pd.DataFrame([{cls: counts.get(cls, 0.0) for cls in class_set} for counts in class_counts_list])
    national_totals = counts_df.sum(axis=0).to_dict()

    for cls in class_set:
        total = national_totals[cls]
        if total > 0:
            counts_df[cls + '_share'] = counts_df[cls] / total
            print("counts_df[cls] / total" , counts_df[cls] / total)
        else:
            counts_df[cls + '_share'] = 0.0

    gdf = pd.concat([gdf.reset_index(drop=True), counts_df], axis=1)
    gdf.to_file(output_path)
    print(f"Saved output to: {output_path}")

if __name__ == "__main__":
    shapefile_path = r"C:\Users\shaba\Desktop\1.1 Crop ID - Domain Adaptation\decifr-ag\data\train_ready\malawi_250617_bt\gadm41_MWI_shp\gadm41_MWI_2.shp"
    raster_path = r"C:\Users\shaba\Desktop\1.1 Crop ID - Domain Adaptation\inference\malawi_250617_bt\masked_outputs\20211004_20221031_output_masked.tif"
    output_path = r"C:\Users\shaba\Desktop\1.1 Crop ID - Domain Adaptation\inference\malawi_250617_bt\masked_outputs\20211004_20221031_output_crop_share_prop.shp"

    count_classes_per_geometry_mp(shapefile_path, raster_path, output_path)
