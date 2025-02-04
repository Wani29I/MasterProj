import os
import rasterio
import numpy as np
from PIL import Image
import geopandas as gpd
from rasterio.mask import mask
from rasterio.plot import show
import matplotlib.pyplot as plt
from shapely.geometry import box
from rasterio.windows import Window
# from rio_color.operations import parse_operations

def representRaster(filePath):
    with rasterio.open(filePath) as src:
        data = src.read(5)
        fig = plt.figure(figsize=[12,8])
        # Plot the raster data using matplotlib
        ax = fig.add_axes([0, 0, 1, 1])
        raster_image=ax.imshow(data)
        plt.show()

def rotateAndDeleteEmptySpace(fileName, outputName, degreeOfRotation):
    # Open the raster .tif file
    with rasterio.open(fileName) as src:
        # Read the data from the raster (shape will be (bands, height, width))
        data = src.read()

        # Get the metadata and affine transformation
        profile = src.profile
        transform = src.transform

        cropped_bands = []
        for i in range(data.shape[0]):  # Loop over each band
            band = data[i]  # Get the i-th band
            pil_band = Image.fromarray(band)
            rotated_band = pil_band.rotate(degreeOfRotation, expand=True)
            rotated_band_data = np.array(rotated_band)
            
            # Crop empty space
            non_empty_pixels = rotated_band_data > 0
            non_empty_rows = np.any(non_empty_pixels, axis=1)
            non_empty_cols = np.any(non_empty_pixels, axis=0)
            
            min_row, max_row = np.where(non_empty_rows)[0][[0, -1]]
            min_col, max_col = np.where(non_empty_cols)[0][[0, -1]]
            
            cropped_band_data = rotated_band_data[min_row:max_row+1, min_col:max_col+1]
            cropped_bands.append(cropped_band_data)

        # Stack the cropped bands into a multi-band image
        cropped_data = np.stack(cropped_bands)

        # Update the profile and save the multi-band image
        profile.update(width=cropped_data.shape[2], height=cropped_data.shape[1])

        # Save the cropped image
        with rasterio.open(outputName, 'w', **profile) as dst:
            dst.write(cropped_data)

        # # Optionally, show the cropped rotated image
        # show(cropped_data[0])  # Display the cropped image (first band)

def removeNotFileType(fileType, path):
    for file in os.listdir(path):
        if(file[-len(fileType):] != fileType):
            os.remove(path + "/" + file)
            print(f"removed {file[-len(fileType):]} file")

def removeFileName(fileName, path):
    for file in os.listdir(path):
        if(file[:len(fileName)] == fileName):
            os.remove(path + "/" + file)
            print(f"removed {file} file")

def checkFileName(file, fileName):
    if(file[:len(fileName)] == fileName):
        return True
    else:
        return False
    
def loopCheckFile(path, numFile):
    for dayFolder in os.listdir(path):
        dayFolderPath = path + "/" + dayFolder
        for eachClippedFolder in os.listdir(dayFolderPath):
            eachClippedFolderPath = dayFolderPath + "/" + eachClippedFolder
            if(len(os.listdir(eachClippedFolderPath)) != numFile):
                print(len(os.listdir(eachClippedFolderPath)))
                print(eachClippedFolderPath)

def cropImage(input_raster, output_raster, offsetRatio):
    """
    Crop a multilayer raster file to the specified bounds.
    
    Parameters:
    - input_raster: Path to the input raster file.
    - output_raster: Path to save the cropped raster file.
    - offsetRatio: crop ratio
    """
    with rasterio.open(input_raster) as src:
        # Create a geometry for the crop bounds
        bounds = src.bounds
        xOffset = (bounds.top - bounds.bottom) * offsetRatio
        yOffset = (bounds.right - bounds.left) * offsetRatio
        crop_bounds = (bounds.left + yOffset, bounds.bottom + xOffset, 
                       bounds.right - yOffset, bounds.top - xOffset)
        crop_geom = [box(*crop_bounds)]

        # Crop the raster using the geometry
        out_image, out_transform = mask(src, crop_geom, crop=True)

        # Update metadata for the new cropped raster
        out_meta = src.meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform,
            "dtype": src.dtypes[0],  # Preserve original dtype
            "nodata": src.nodata,  # Preserve nodata value
            "compress": src.profile.get("compress", "LZW"),  # Preserve compression
            "photometric": src.tags().get("photometric", "RGB"),  # Preserve color interpretation
        })

        # Write the cropped raster to a new file
        with rasterio.open(output_raster, "w", **out_meta) as dest:
            dest.write(out_image)


def flip_raster(input_raster, output_raster):
    """
    Perform vertical and horizontal flips on a multi-band raster.

    Parameters:
    - input_raster: Path to the input raster file.
    - output_raster_horizontal: Path to save the horizontally flipped raster.
    """
    with rasterio.open(input_raster) as src:
        # Read the entire raster data as a numpy array
        data = src.read()  # Shape: (bands, height, width)

        # Perform horizontal flips
        horizontal_flipped = np.fliplr(data)  # Flip horizontally

        # Update metadata for the flipped rasters
        meta = src.meta.copy()

        # Save the horizontally flipped raster
        with rasterio.open(output_raster, 'w', **meta) as dest_horizontal:
            dest_horizontal.write(horizontal_flipped)

def loopPathRotate(path, type, inputFileName, outputFileName, degree):
    countday = 1
    for dayFolder in os.listdir(path):
        dayFolderPath = path + "/" + dayFolder
        coultClipped = 1
        for eachClippedFolder in os.listdir(dayFolderPath):
            eachClippedFolderPath = dayFolderPath + "/" + eachClippedFolder
            countImage = 1
            removeNotFileType(".tif", eachClippedFolderPath)
            removeFileName(outputFileName, eachClippedFolderPath)
            for imageFile in os.listdir(eachClippedFolderPath):
                if(checkFileName(imageFile, inputFileName)):
                    inputPath = eachClippedFolderPath + "/" + imageFile
                    outputPath = eachClippedFolderPath + "/" + outputFileName + "_" + eachClippedFolder +".tif"
                    rotateAndDeleteEmptySpace(inputPath, outputPath, degree)
                    print(f"{type}: -------------------- mainLoop: - {dayFolder} - {countday} / {len(os.listdir(path))} -------------------- subLoop: - {eachClippedFolder} - {coultClipped} / {len(os.listdir(dayFolderPath))}----------------------------------------------------------------------------------------------- ")
                    countImage+=1
                    break
            coultClipped += 1
        countday += 1

def loopPathCrop(path, type, inputFileName, outputFileName, cropRatio):
    countday = 1
    for dayFolder in os.listdir(path):
        dayFolderPath = path + "/" + dayFolder
        coultClipped = 1
        for eachClippedFolder in os.listdir(dayFolderPath):
            eachClippedFolderPath = dayFolderPath + "/" + eachClippedFolder
            countImage = 1
            removeNotFileType(".tif", eachClippedFolderPath)
            removeFileName(outputFileName, eachClippedFolderPath)
            for imageFile in os.listdir(eachClippedFolderPath):
                if(checkFileName(imageFile, inputFileName)):
                    inputPath = eachClippedFolderPath + "/" + imageFile
                    outputPath = eachClippedFolderPath + "/" + outputFileName + "_" + eachClippedFolder +".tif"
                    cropImage(inputPath, outputPath, cropRatio)
                    print(f"{type}: -------------------- mainLoop: - {dayFolder} - {countday} / {len(os.listdir(path))} -------------------- subLoop: - {eachClippedFolder} - {coultClipped} / {len(os.listdir(dayFolderPath))}----------------------------------------------------------------------------------------------- ")
                    countImage+=1
                    break
            coultClipped += 1
        countday += 1

def loopPathFlip(path, type, inputFileName, outputFileName):
    countday = 1
    for dayFolder in os.listdir(path):
        dayFolderPath = path + "/" + dayFolder
        coultClipped = 1
        for eachClippedFolder in os.listdir(dayFolderPath):
            eachClippedFolderPath = dayFolderPath + "/" + eachClippedFolder
            countImage = 1
            removeNotFileType(".tif", eachClippedFolderPath)
            removeFileName(outputFileName, eachClippedFolderPath)
            for imageFile in os.listdir(eachClippedFolderPath):
                if(checkFileName(imageFile, inputFileName)):
                    inputPath = eachClippedFolderPath + "/" + imageFile
                    outputPath = eachClippedFolderPath + "/" + outputFileName + "_" + eachClippedFolder +".tif"
                    # flip_raster(inputPath, outputPath)
                    print(f"{type}: -------------------- mainLoop: - {dayFolder} - {countday} / {len(os.listdir(path))} -------------------- subLoop: - {eachClippedFolder} - {coultClipped} / {len(os.listdir(dayFolderPath))}----------------------------------------------------------------------------------------------- ")
                    countImage+=1
                    break
            coultClipped += 1
        countday += 1

def loopPathChangeName(path):
    countday = 1
    for dayFolder in os.listdir(path):
        dayFolderPath = path + "/" + dayFolder
        coultClipped = 1
        for eachClippedFolder in os.listdir(dayFolderPath):
            eachClippedFolderPath = dayFolderPath + "/" + eachClippedFolder
            countImage = 1
            removeNotFileType(".tif", eachClippedFolderPath)
            for imageFile in os.listdir(eachClippedFolderPath):
                oldPath = eachClippedFolderPath + "/" + imageFile
                newPath = oldPath[:(-len(eachClippedFolder)-4)] + "_" + oldPath[(-len(eachClippedFolder)-4):]
                print(f"----- mainLoop: - {dayFolder} - {countday} / {len(os.listdir(path))} -------------------- subLoop: - {eachClippedFolder} - {coultClipped} / {len(os.listdir(dayFolderPath))}------------------------------------------------------------------------------- ")
                print(eachClippedFolder)
                if(oldPath[(-len(eachClippedFolder)-5)] != "_"):
                    # os.rename(oldPath, newPath)
                    print("nope!")
                countImage+=1
            coultClipped += 1
        countday += 1

def loopRemoveNotFileType(path):
    countday = 1
    for dayFolder in os.listdir(path):
        dayFolderPath = path + "/" + dayFolder
        coultClipped = 1
        for eachClippedFolder in os.listdir(dayFolderPath):
            eachClippedFolderPath = dayFolderPath + "/" + eachClippedFolder
            removeNotFileType(".tif", eachClippedFolderPath)
            print(eachClippedFolderPath)
            coultClipped += 1
        countday += 1

def loopRemoveFile(path, fileName):
    countday = 1
    for dayFolder in os.listdir(path):
        dayFolderPath = path + "/" + dayFolder
        coultClipped = 1
        for eachClippedFolder in os.listdir(dayFolderPath):
            eachClippedFolderPath = dayFolderPath + "/" + eachClippedFolder
            removeFileName(fileName, eachClippedFolderPath)
            print(eachClippedFolderPath)
            coultClipped += 1
        countday += 1

def crop_raster(input_raster, output_raster, offset_ratio):
    """
    Crop a raster image while preserving all layers and avoiding color changes.
    
    Parameters:
    - input_raster: Path to the input raster file.
    - output_raster: Path to save the cropped raster file.
    - offset_ratio: Ratio of the image to crop (e.g., 0.02 means 2% from each side).
    """
    with rasterio.open(input_raster) as src:
        # Get original dimensions
        height, width = src.height, src.width
        
        # data = src.read(window=window)
        # fig = plt.figure(figsize=[12,8])
        # # Plot the raster data using matplotlib
        # ax = fig.add_axes([0, 0, 1, 1])
        # raster_image=ax.imshow(data)
        # plt.show()

        # Compute cropping offsets
        x_offset = int(width * offset_ratio)
        y_offset = int(height * offset_ratio)

        # Define new width and height
        new_width = width - (2 * x_offset)
        new_height = height - (2 * y_offset)

        # Read only the cropped window (no color processing)
        window = Window(x_offset, y_offset, new_width, new_height)
        cropped_data = src.read(window=window)

        # Update metadata to match the cropped size
        out_meta = src.meta.copy()
        out_meta.update({
            "width": new_width,
            "height": new_height,
            "transform": src.window_transform(window)
        })

        # Save cropped raster
        with rasterio.open(output_raster, "w", **out_meta) as dest:
            dest.write(cropped_data)


def adjustImageContrastSaturation(input, Output, operations):
    # Define the operations
    operations = operations

    # Open the original multi-layer raster file
    with rasterio.open(input) as src:
        # Read the data
        data = src.read()

        # Normalize data to 0-1 range and cast to floatุภ
        data = data.astype(np.float64) / 255.0

        # Apply the color operations
        for func in parse_operations(operations):
            data = func(data)

        # Denormalize back to original range and cast to uint8
        data = np.clip(data * 255, 0, 255).astype(np.uint8)

        # Update metadata
        meta = src.meta
        meta.update(dtype='uint8')

        # Write the adjusted data to a new file
        with rasterio.open(Output, 'w', **meta) as dst:
            dst.write(data)

# crop_raster("tiltCorrected1.tif","tiltCorrected1cropped.tif", 0.01)
# adjustImageContrastSaturation("tiltCorrected1cropped.tif", "color_fixed.tif","sigmoidal rgb 8 0.35, saturation 0.75")

# mac path
# loopRemoveNotFileType("/Volumes/HD-PCFSU3-A/ice-wheat/data/dataForProcess/MUL")
# loopRemoveNotFileType("/Volumes/HD-PCFSU3-A/ice-wheat/data/dataForProcess/RGB")

# rotateAndDeleteEmptySpace("normal_87.tif", "normal_87_fixed.tif", 13)
# rotateAndDeleteEmptySpace("normal_67.tif", "normal_67_fixed.tif", 13)
# rotateAndDeleteEmptySpace("normal_11.tif", "normal_11_fixed.tif", 13)
# cropImage("normal_87_fixed.tif","normal_87_fixed_crop95.tif", 0.01265)
# cropImage("normal_67_fixed.tif","normal_67_fixed_crop95.tif", 0.01265)
# cropImage("normal_11_fixed.tif","normal_11_fixed_crop95.tif", 0.01265)

# adjust_luminance("./cropped.tif", "./croppedLuminance13.tif", 100)

# representRaster("cropped.tif")
# representRaster("croppedLuminance13.tif")