import os
import rasterio
import numpy as np
from PIL import Image
import geopandas as gpd
from rasterio.mask import mask
from rasterio.plot import show
import matplotlib.pyplot as plt
from shapely.geometry import box

def representRaster(filePath):
    with rasterio.open(filePath) as src:
        data = src.read(1)
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
        xOffset = (bounds.top - bounds.bottom)*offsetRatio
        yOffset = (bounds.right - bounds.left)*offsetRatio
        crop_bounds = (bounds.left+yOffset, bounds.bottom+xOffset, bounds.right-yOffset, bounds.top-xOffset)  # Replace with your bounding box coordinates
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
                    flip_raster(inputPath, outputPath)
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
            for imageFile in os.listdir(eachClippedFolderPath):
                oldPath = eachClippedFolderPath + "/" + imageFile
                newPath = oldPath[:(-len(eachClippedFolder)-4)] + "_" + oldPath[(-len(eachClippedFolder)-4):]
                # print(f"-------------------- mainLoop: - {dayFolder} - {countday} / {len(os.listdir(path))} -------------------- subLoop: - {eachClippedFolder} - {coultClipped} / {len(os.listdir(dayFolderPath))}----------------------------------------------------------------------------------------------- ")
                # print(eachClippedFolder)
                if(oldPath[(-len(eachClippedFolder)-5)] != "_"):
                    # os.rename(oldPath, newPath)
                    print("nope!")
                countImage+=1
            coultClipped += 1
        countday += 1


def adjust_luminance(input_raster, output_raster, scale_factor):
    """
    Adjust the luminance of all bands in a raster by scaling pixel values.

    Parameters:
    - input_raster: Path to the input raster file.
    - output_raster: Path to save the adjusted raster.
    - scale_factor: Factor to adjust brightness (>1 for brighter, <1 for darker).
    """
    with rasterio.open(input_raster) as src:
        # Read the raster data (shape: [bands, height, width])
        data = src.read()
        
        # Initialize an array for adjusted data with the same shape
        adjusted_data = np.zeros_like(data, dtype='float32')

        # Adjust luminance for each band
        for band in range(data.shape[0]):
            adjusted_data[band] = data[band] * scale_factor
        
        # Clip adjusted values to the valid range of the raster's data type
        dtype_info = (
            np.iinfo(src.dtypes[0]) 
            if np.issubdtype(src.dtypes[0], np.integer) 
            else np.finfo(src.dtypes[0])
        )
        adjusted_data = np.clip(adjusted_data, dtype_info.min, dtype_info.max)

        # Convert back to the original data type
        adjusted_data = adjusted_data.astype(src.dtypes[0])

        # Save the adjusted raster
        meta = src.meta.copy()
        with rasterio.open(output_raster, 'w', **meta) as dest:
            dest.write(adjusted_data)

# loopPathCrop("F:/ice-wheat/data/dataForProcess/RGB", "RGB", "crop95percent", "crop76percent", 0.1)
# loopPathCrop("F:/ice-wheat/data/dataForProcess/MUL", "MUL", "crop95percent", "crop76percent", 0.1)
# loopPathFlip("F:/ice-wheat/data/dataForProcess/RGB", "RGB1", "crop95percent", "crop95FlipHorizontal")
# loopPathFlip("F:/ice-wheat/data/dataForProcess/MUL", "MUL1", "crop95percent", "crop95FlipHorizontal")
# loopPathRotate("F:/ice-wheat/data/dataForProcess/RGB", "RGB2", "crop95percent", "crop95tilt90", 90)
# loopPathRotate("F:/ice-wheat/data/dataForProcess/MUL", "MUL2", "crop95percent", "crop95tilt90", 90)
# loopCheckFile("F:/ice-wheat/data/dataForProcess/RGB", 12)
# loopCheckFile("F:/ice-wheat/data/dataForProcess/MUL", 12)
# loopPathChangeName("F:/ice-wheat/data/dataForProcess/MUL")
# loopPathChangeName("F:/ice-wheat/data/dataForProcess/RGB")

# adjust_luminance("./cropped.tif", "./croppedLuminance13.tif", 100)

# representRaster("cropped.tif")
# representRaster("croppedLuminance13.tif")