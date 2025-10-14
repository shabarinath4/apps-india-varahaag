import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling

input_tif = r'C:\Users\shaba\Desktop\1.1 Crop ID - Domain Adaptation\inference\malawi_250617_bt\masked_outputs\20221003_20231030_output_masked.tif'
output_tif = r'C:\Users\shaba\Desktop\1.1 Crop ID - Domain Adaptation\inference\malawi_250617_bt\masked_outputs\20221003_20231030_output_masked_equal_area.tif'

dst_crs = (
    "+proj=aea +lat_1=-15 +lat_2=-10 +lat_0=-13.5 +lon_0=34 "
    "+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
)

with rasterio.open(input_tif) as src:
    resolution = src.res[0]

    transform, width, height = calculate_default_transform(
        src.crs, dst_crs, src.width, src.height, *src.bounds, resolution=resolution)

    kwargs = src.meta.copy()
    kwargs.update({
        'crs': dst_crs,
        'transform': transform,
        'width': width,
        'height': height,
        'dtype': src.dtypes[0],
        'compress': 'lzw'
    })

    with rasterio.open(output_tif, 'w', **kwargs) as dst:
        for i in range(1, src.count + 1):
            reproject(
                source=rasterio.band(src, i),
                destination=rasterio.band(dst, i),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=dst_crs,
                resampling=Resampling.nearest)
