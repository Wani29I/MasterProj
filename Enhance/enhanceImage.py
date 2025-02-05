import os
import cv2
import rasterio
import numpy as np
from PIL import Image
import geopandas as gpd
from rasterio.mask import mask
from rasterio.plot import show
import matplotlib.pyplot as plt
from shapely.geometry import box
from rasterio.windows import Window
# from sklearn.decomposition import PCA

def display_rgb_raster(raster_path):
    with rasterio.open(raster_path) as src:
        # Read Red, Green, Blue bands (assuming they are in 1, 2, 3)
        red = src.read(1).astype(np.float32)
        green = src.read(2).astype(np.float32)
        blue = src.read(3).astype(np.float32)

        # Normalize bands to 0-1 for visualization
        red = (red - red.min()) / (red.max() - red.min())
        green = (green - green.min()) / (green.max() - green.min())
        blue = (blue - blue.min()) / (blue.max() - blue.min())

        # Stack bands to create an RGB image
        rgb_image = np.dstack((red, green, blue))

        # Show the image
        plt.figure(figsize=(10, 10))
        plt.imshow(rgb_image)
        plt.axis('off')
        plt.title("RGB Raster Visualization")
        plt.show()

def normalize_rgbdsm(input_raster, output_raster):
    with rasterio.open(input_raster) as src:
        profile = src.profile.copy()
        
        # Read R, G, B, DSM (ignoring Alpha for now)
        red, green, blue, alpha, dsm = src.read()

        # Normalize each RGB band to 0-255
        red_norm = cv2.normalize(red, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        green_norm = cv2.normalize(green, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        blue_norm = cv2.normalize(blue, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Save normalized raster (Preserve all bands)
        with rasterio.open(output_raster, "w", **profile) as dest:
            dest.write(red_norm, 1)
            dest.write(green_norm, 2)
            dest.write(blue_norm, 3)
            dest.write(alpha, 4)  # Alpha remains unchanged
            dest.write(dsm, 5)

def apply_gamma_rgbdsm(input_raster, output_raster, gamma=1.2):
    with rasterio.open(input_raster) as src:
        profile = src.profile.copy()
        
        # Read all bands (R, G, B, Alpha, DSM)
        red, green, blue, alpha, dsm = src.read()

        # Normalize to 0-1 range
        red = red / 255.0
        green = green / 255.0
        blue = blue / 255.0

        # Apply gamma correction
        red_corr = np.power(red, gamma) * 255.0
        green_corr = np.power(green, gamma) * 255.0
        blue_corr = np.power(blue, gamma) * 255.0

        # Convert back to uint8
        red_corr = np.clip(red_corr, 0, 255).astype(np.uint8)
        green_corr = np.clip(green_corr, 0, 255).astype(np.uint8)
        blue_corr = np.clip(blue_corr, 0, 255).astype(np.uint8)

        # Save enhanced raster
        with rasterio.open(output_raster, "w", **profile) as dest:
            dest.write(red_corr, 1)
            dest.write(green_corr, 2)
            dest.write(blue_corr, 3)
            dest.write(alpha, 4)  # Alpha remains unchanged
            dest.write(dsm, 5)  # DSM remains unchanged

def adjust_contrast(input_raster, output_raster):
    with rasterio.open(input_raster) as src:
        profile = src.profile.copy()
        
        # Read RGB bands (assuming 1=Red, 2=Green, 3=Blue)
        red, green, blue, alpha, dsm = src.read(1), src.read(2), src.read(3), src.read(4), src.read(5)
        # Apply CLAHE (Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        red_adj = clahe.apply(cv2.normalize(red, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))
        green_adj = clahe.apply(cv2.normalize(green, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))
        blue_adj = clahe.apply(cv2.normalize(blue, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))

        # Save enhanced raster
        with rasterio.open(output_raster, "w", **profile) as dest:
            dest.write(red_adj, 1)
            dest.write(green_adj, 2)
            dest.write(blue_adj, 3)
            dest.write(alpha, 4)
            dest.write(dsm, 5)

def adjust_saturation(input_raster, output_raster, saturation_scale=1.3):
    with rasterio.open(input_raster) as src:
        profile = src.profile.copy()

        # Read RGB bands
        red, green, blue = src.read(1), src.read(2), src.read(3)

        # Stack RGB into an image
        img_rgb = np.dstack((red, green, blue)).astype(np.uint8)

        # Convert to HSV
        img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

        # Scale saturation channel
        img_hsv[:, :, 1] = np.clip(img_hsv[:, :, 1] * saturation_scale, 0, 255)

        # Convert back to RGB
        img_adjusted = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)

        # Save enhanced raster
        with rasterio.open(output_raster, "w", **profile) as dest:
            dest.write(img_adjusted[:, :, 0], 1)
            dest.write(img_adjusted[:, :, 1], 2)
            dest.write(img_adjusted[:, :, 2], 3)
            dest.write(src.read(4), 4)
            dest.write(src.read(5), 5)

def unsharp_mask(input_raster, output_raster, sigma=1.0, strength=1.3):
    """
    Apply Unsharp Masking for better sharpening while maintaining natural colors.
    
    Parameters:
    - input_raster: Path to input raster
    - output_raster: Path to save the sharpened image
    - sigma: The blur intensity (higher = less sharpening)
    - strength: Sharpening intensity (higher = more sharpening)
    """
    with rasterio.open(input_raster) as src:
        profile = src.profile.copy()

        # Read RGB bands
        red, green, blue = src.read(1), src.read(2), src.read(3)

        # Convert to float for better precision
        red, green, blue = red.astype(np.float32), green.astype(np.float32), blue.astype(np.float32)

        # Apply Gaussian blur to create a "softened" version of the image
        red_blur = cv2.GaussianBlur(red, (0, 0), sigma)
        green_blur = cv2.GaussianBlur(green, (0, 0), sigma)
        blue_blur = cv2.GaussianBlur(blue, (0, 0), sigma)

        # Compute the sharpened image: original + (original - blurred) * strength
        red_sharp = np.clip(red + (red - red_blur) * strength, 0, 255).astype(np.uint8)
        green_sharp = np.clip(green + (green - green_blur) * strength, 0, 255).astype(np.uint8)
        blue_sharp = np.clip(blue + (blue - blue_blur) * strength, 0, 255).astype(np.uint8)

        # Save enhanced raster
        with rasterio.open(output_raster, "w", **profile) as dest:
            dest.write(red_sharp, 1)
            dest.write(green_sharp, 2)
            dest.write(blue_sharp, 3)
            dest.write(src.read(4), 4)
            dest.write(src.read(5), 5)


def normalize_multispectral(input_raster, output_raster):
    with rasterio.open(input_raster) as src:
        profile = src.profile.copy()

        # Read all 4 bands
        bands = [src.read(i) for i in range(1, 5)]

        # Normalize each band (0-1 range)
        bands_norm = [(band - np.min(band)) / (np.max(band) - np.min(band)) for band in bands]

        # Convert to uint8 (0-255) for saving
        bands_scaled = [np.clip(band * 255, 0, 255).astype(np.uint8) for band in bands_norm]

        # Save the normalized raster
        with rasterio.open(output_raster, "w", **profile) as dest:
            for i, band in enumerate(bands_scaled, start=1):
                dest.write(band, i)

def compute_ndvi_ndre(input_raster, output_ndvi, output_ndre):
    with rasterio.open(input_raster) as src:
        profile = src.profile.copy()

        # Read NIR, Red, and RedEdge bands
        nir = src.read(4).astype(np.float32)  # NIR band
        red = src.read(2).astype(np.float32)  # Red band
        red_edge = src.read(3).astype(np.float32)  # RedEdge band

        # Compute NDVI
        ndvi = (nir - red) / (nir + red + 1e-10)  # Avoid division by zero
        ndvi_scaled = ((ndvi + 1) / 2 * 255).astype(np.uint8)

        # Compute NDRE
        ndre = (nir - red_edge) / (nir + red_edge + 1e-10)
        ndre_scaled = ((ndre + 1) / 2 * 255).astype(np.uint8)

        # Save NDVI
        profile.update(dtype=rasterio.uint8, count=1)
        with rasterio.open(output_ndvi, "w", **profile) as dest:
            dest.write(ndvi_scaled, 1)

        # Save NDRE
        with rasterio.open(output_ndre, "w", **profile) as dest:
            dest.write(ndre_scaled, 1)

def apply_clahe_to_multispectral(input_raster, output_raster):
    with rasterio.open(input_raster) as src:
        profile = src.profile.copy()

        # Read all 4 bands
        bands = [src.read(i) for i in range(1, 5)]

        # Apply CLAHE to each band
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        bands_enhanced = [clahe.apply(cv2.normalize(band, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)) for band in bands]

        # Save the enhanced raster
        with rasterio.open(output_raster, "w", **profile) as dest:
            for i, band in enumerate(bands_enhanced, start=1):
                dest.write(band, i)
