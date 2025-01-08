import os
import math
import pprint
import pyproj
import rasterio
import numpy as np
from rasterio.mask import mask
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, mapping
from rasterio.transform import Affine
from rasterio.enums import Resampling
from PIL import Image
import rasterio
from rasterio.enums import Resampling

def representRaster(filePath):
    with rasterio.open(filePath) as src:
        data = src.read(1)
        fig = plt.figure(figsize=[12,8])
        # Plot the raster data using matplotlib
        ax = fig.add_axes([0, 0, 1, 1])
        raster_image=ax.imshow(data)
        plt.show()
# representRaster("band_1.tif")



def extract_and_save_bands(input_path, output_dir):
  """
  Extracts specified bands from a raster image and saves them as individual GeoTIFF files.

  Args:
      input_path: Path to the input raster image.
      output_dir: Path to the output directory.
  """

  with rasterio.open(input_path) as src:
    num_bands = src.count
    band_descriptions = src.descriptions 
    print(band_descriptions)

# Example usage
input_path = "F:/ice-wheat/data/dataForProcess/RGB/RGB_202406111255/53/normal53.tif"
output_dir = "."
extract_and_save_bands(input_path, output_dir) 