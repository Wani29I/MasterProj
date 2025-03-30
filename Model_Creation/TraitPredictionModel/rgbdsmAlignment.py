import os
import cv2  
import torch
import rasterio
import numpy as np
from PIL import Image
import torchvision.transforms as T
from rasterio.enums import Resampling

def openAndSplitData(dataFilePath):
    # open data path file
    returnData = []
    with open(dataFilePath, "r") as file:
        allData = file.read().splitlines()

    for data in allData:
        returnData.append(data.split(','))

    return returnData

def checkSizergbdsm(rgbPath, dsmPath):
    # Check RGB
    rgb = Image.open(rgbPath)
    rgb_width, rgb_height = rgb.size
    print(f"RGB Size: {rgb_width} x {rgb_height}")
    print(rgbPath)

    # Check DSM
    with rasterio.open(dsmPath) as dsm:
        dsm_width, dsm_height = dsm.width, dsm.height
        print(f"DSM Size: {dsm_width} x {dsm_height}")
        print(dsmPath)

    # # Check if same size
    # if (rgb_width, rgb_height) == (dsm_width, dsm_height):
    #     print("✅ RGB and DSM are the same size!")
    # else:
    #     print("❌ RGB and DSM are NOT the same size!")
    return rgb_width, rgb_height, dsm_width, dsm_height


def loopCheckSize(mainPath, dataList):
    ''' 
    loop and print max, min size of both rgb and dsm image
    '''
    max_rgb_width = 0
    max_rgb_height = 0
    max_dsm_width = 0
    max_dsm_height = 0
    min_rgb_width = 9999
    min_rgb_height = 9999
    min_dsm_width = 9999
    min_dsm_height = 9999

    for line in dataList:
        rgbPath = mainPath + line[0]
        dsmPath = mainPath + line[1]
        rgb_width, rgb_height, dsm_width, dsm_height = checkSizergbdsm(rgbPath,dsmPath)

        if(rgb_width > max_rgb_width):
            max_rgb_width = rgb_width
        if(rgb_width < min_rgb_width):
            min_rgb_width = rgb_width

        if(rgb_height > max_rgb_height):
            max_rgb_height = rgb_height
        if(rgb_height < min_rgb_height):
            min_rgb_height = rgb_height

        if(dsm_width > max_dsm_width):
            max_dsm_width = dsm_width
        if(dsm_width < min_dsm_width):
            min_dsm_width = dsm_width

        if(dsm_height > max_dsm_height):
            max_dsm_height = dsm_height
        if(dsm_height < min_dsm_height):
            min_dsm_height = dsm_height

    print("max_rgb_width: ", max_rgb_width)
    print("max_rgb_height: ", max_rgb_height)
    print("max_dsm_width: ", max_dsm_width)
    print("max_dsm_height: ", max_dsm_height)
    print("min_rgb_width: ", min_rgb_width)
    print("min_rgb_height: ", min_rgb_height)
    print("min_dsm_width: ", min_dsm_width)
    print("min_dsm_height: ", min_dsm_height)



def resize_rgb(input_path, output_path, target_size=(512, 256)):
    """
    Resize RGB image (JPG) to target size and save.

    Args:
        input_path (str): Path to input RGB image (JPG).
        output_path (str): Path to save resized RGB image.
        target_size (tuple): (width, height) to resize.
    """
    img = Image.open(input_path).convert("RGB")
    img_resized = img.resize(target_size, Image.BILINEAR)  # Smooth resize
    img_resized.save(output_path, "JPEG", quality=95)  # Save as high-quality JPG
    print(f"Saved resized RGB to: {output_path}")


def resize_and_save_dsm(input_dsm_path, output_dsm_path, target_width=512, target_height=256):
    """
    Resize DSM (GeoTIFF) and save to a new file for checking.

    Args:
        input_dsm_path (str): Path to original DSM file (.tif).
        output_dsm_path (str): Path to save resized DSM file (.tif).
        target_width (int): Desired width in pixels.
        target_height (int): Desired height in pixels.
    """
    with rasterio.open(input_dsm_path) as src:
        dsm_data = src.read(1)  # Read DSM as 2D array (band 1)
        profile = src.profile.copy()  # Copy profile to keep metadata

        # Compute scaling factor
        scale_w = target_width / src.width
        scale_h = target_height / src.height

        # Resample DSM to new size using bilinear interpolation
        dsm_resized = src.read(
            out_shape=(src.count, target_height, target_width),
            resampling=Resampling.bilinear
        )[0]  # Take band 1 after resampling

        # Update profile for new size
        profile.update({
            'height': target_height,
            'width': target_width,
            'transform': rasterio.transform.from_origin(
                src.bounds.left, src.bounds.top,
                (src.bounds.right - src.bounds.left) / target_width,
                (src.bounds.top - src.bounds.bottom) / target_height
            )
        })

        # Save resized DSM
        with rasterio.open(output_dsm_path, 'w', **profile) as dst:
            dst.write(dsm_resized, 1)  # Write band 1

        print(f"✅ Resized DSM saved to: {output_dsm_path}")
        print(f"New size: {target_width} x {target_height}")

def checkRGBRange(rgbPath):
    # Load RGB image
    rgb = Image.open(rgbPath).convert("RGB")

    # Convert to NumPy array
    rgb_array = np.array(rgb)

    # Check range
    print("RGB Min:", rgb_array.min())
    print("RGB Max:", rgb_array.max())

def checkDSMRange(dsmPath):
    with rasterio.open(dsmPath) as src:
        dsmData = src.read(1).astype(np.float32)
        nodata = src.nodata  # Get NoData value

    # Mask NoData values
    dsm_masked = np.ma.masked_equal(dsmData, nodata)
    # Check correct min and max (ignoring NoData)
    print("DSM Min (masked):", dsm_masked.min())
    print("DSM Max (masked):", dsm_masked.max())

def normalize_dsm(dsm_array, fixed_min=200.0, fixed_max=230.0, nodata_value=-10000.0, fill_value=0.0):
    """
    Normalize DSM data to [0, 1] with fixed min/max and handle NoData.
    
    Args:
        dsm_array (np.ndarray): The DSM array (H, W), raw elevation data.
        fixed_min (float): Fixed minimum value for normalization.
        fixed_max (float): Fixed maximum value for normalization.
        nodata_value (float): Value representing NoData in DSM.
        fill_value (float): Value to fill for NoData areas (default 0.0).
    
    Returns:
        np.ndarray: Normalized DSM in range [0, 1] (H, W), with NoData handled.
    """
    # Mask NoData values
    dsm_masked = np.ma.masked_equal(dsm_array, nodata_value)
    
    # Clip values to fixed range to avoid outliers
    dsm_clipped = np.clip(dsm_masked, fixed_min, fixed_max)
    
    # Normalize to [0, 1] using fixed min/max
    dsm_normalized = (dsm_clipped - fixed_min) / (fixed_max - fixed_min)
    
    # Fill NoData with specified fill_value (e.g., 0.0)
    dsm_normalized_filled = dsm_normalized.filled(fill_value)
    
    return dsm_normalized_filled.astype(np.float32)  # Ensure float32 for model input

def normalizeDSM(dsmPath):
    # Load DSM file
    with rasterio.open(dsmPath) as src:
        dsm = src.read(1).astype(np.float32)

    # Normalize DSM
    normalized_dsm = normalize_dsm(dsm)

    # Check result
    print("Normalized DSM Min:", normalized_dsm.min())
    print("Normalized DSM Max:", normalized_dsm.max())
    print("Normalized DSM Shape:", normalized_dsm.shape)

def resize_rgb_return_array(input_path, target_size=(512, 256)):
    """
    Resize RGB image and return as NumPy array (no saving).

    Args:
        input_path (str): Path to input RGB image.
        target_size (tuple): (width, height) to resize.

    Returns:
        np.ndarray: Resized RGB image as array.
    """
    img = Image.open(input_path).convert("RGB")
    img_resized = img.resize(target_size, Image.BILINEAR)  # Smooth resize
    return np.array(img_resized)


def resize_dsm_return_array(input_dsm_path, target_width=512, target_height=256):
    """
    Resize DSM and return as NumPy array (no saving).

    Args:
        input_dsm_path (str): Path to DSM file.
        target_width (int): Desired width.
        target_height (int): Desired height.

    Returns:
        np.ndarray: Resized DSM array.
    """
    with rasterio.open(input_dsm_path) as src:
        dsm_resized = src.read(
            out_shape=(src.count, target_height, target_width),
            resampling=Resampling.bilinear
        )[0]  # Return band 1

    return dsm_resized


if __name__ == '__main__':
    mainPath = "/Volumes/HD-PCFSU3-A/ice-wheat/data/dataForProcess/mainData"
    dataList = openAndSplitData("RGB_DSM_totEarNum.csv")
    # loopCheckSize(mainPath, dataList)

    for line in dataList:
        normalizeDSM(mainPath +line[1])
