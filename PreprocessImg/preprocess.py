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
from rasterio.plot import show

def representRaster(filePath):
    with rasterio.open(filePath) as src:
        data = src.read(1)
        fig = plt.figure(figsize=[12,8])
        # Plot the raster data using matplotlib
        ax = fig.add_axes([0, 0, 1, 1])
        raster_image=ax.imshow(data)
        plt.show()
representRaster("rotated_image.tif")

# Open the raster .tif file
with rasterio.open("F:/ice-wheat/data/dataForProcess/RGB/RGB_202406111255/53/normal53.tif") as src:
    # Read the data from the raster (shape will be (bands, height, width))
    data = src.read()

    # Get the metadata and affine transformation
    profile = src.profile
    transform = src.transform

    # Rotate the image by 90 degrees (along height and width)
    rotated_data = np.rot90(data, k=2, axes=(1, 2))

    # Update the affine transformation to account for the rotated image
    # When rotating 90 degrees, the width and height swap
    new_transform = Affine(transform.a, transform.b, transform.c, transform.d, transform.e, transform.f)
    new_transform = new_transform * Affine.rotation(180)

    # Update the profile with the new dimensions and transformation
    profile.update(
        transform=new_transform,
        width=rotated_data.shape[2],  # New width (previous height)
        height=rotated_data.shape[1]  # New height (previous width)
    )

    # Save the rotated image
    with rasterio.open('rotated_image.tif', 'w', **profile) as dst:
        dst.write(rotated_data)

    # Optionally, show the rotated image (select one band for display)
    show(rotated_data[0])  # Display the first band, or modify as needed





# "F:/ice-wheat/data/dataForProcess/RGB/RGB_202406111255/53/normal53.tif"