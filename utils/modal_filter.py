import rasterio
from rasterio.windows import Window
from skimage.filters.rank import modal
from skimage.morphology import square
from skimage.util import img_as_ubyte
import numpy as np

def apply_modal_filter_skimage_chunked(input_raster_path, output_raster_path, window_size=3, block_size=1024):
    with rasterio.open(input_raster_path) as src:
        profile = src.profile.copy()
        width, height = src.width, src.height
        dtype = src.dtypes[0]
        nodata = src.nodata

        if dtype != 'uint8':
            raise ValueError("Input raster must be of type uint8 to use skimage.filters.rank.modal.")

        profile.update({
            'compress': 'lzw',
            'tiled': True,
            'blockxsize': block_size,
            'blockysize': block_size,
            'dtype': 'uint8'
        })

        with rasterio.open(output_raster_path, 'w', **profile) as dst:
            for y in range(0, height, block_size):
                for x in range(0, width, block_size):
                    print(x, y)
                    w = min(block_size, width - x)
                    h = min(block_size, height - y)

                    window = Window(x, y, w, h)
                    data = src.read(1, window=window, boundless=True, fill_value=nodata)
                    mask = data != nodata if nodata is not None else np.ones_like(data, dtype=bool)

                    data[data == nodata] = 0
                    data_ubyte = img_as_ubyte(data.astype(np.uint8))
                    mask_ubyte = mask.astype(np.uint8)

                    filtered = modal(data_ubyte, selem=square(window_size), mask=mask_ubyte)

                    if nodata is not None:
                        filtered[~mask] = nodata

                    dst.write(filtered.astype(np.uint8), 1, window=window)

    print(f"Saved modal-filtered raster to: {output_raster_path}")



if __name__ == "__main__":
    raster_path = r"C:\Users\91814\v4_1\20221003_20231030.tiff"  # must be uint8
    output_path = r"C:\Users\91814\v4_1\20221003_20231030_modal_skimage.tiff"

    apply_modal_filter_skimage_chunked(
        raster_path,
        output_path,
        window_size=5,       # e.g., 5x5 modal window
        block_size=10032      # tile size, multiple of 16
    )

