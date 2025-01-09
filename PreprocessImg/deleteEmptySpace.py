import rasterio
import numpy as np
from PIL import Image
from rasterio.plot import show
import os

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

def loopPathRotate(path, type, inputFieName, outputFieName, degree):
    countday = 1
    for dayFolder in os.listdir(path):
        dayFolderPath = path + "/" + dayFolder
        coultClipped = 1
        for eachClippedFolder in os.listdir(dayFolderPath):
            eachClippedFolderPath = dayFolderPath + "/" + eachClippedFolder
            countImage = 1
            removeNotFileType(".tif", eachClippedFolderPath)
            removeFileName(outputFieName, eachClippedFolderPath)
            for imageFile in os.listdir(eachClippedFolderPath):
                if(checkFileName(imageFile, inputFieName)):
                    inputPath = eachClippedFolderPath + "/" + imageFile
                    outputPath = eachClippedFolderPath + "/" + outputFieName + eachClippedFolder +".tif"
                    rotateAndDeleteEmptySpace(inputPath, outputPath, degree)
                    print(f"{type}: -------------------- mainLoop: - {dayFolder} - {countday} / {len(os.listdir(path))} -------------------- subLoop: - {eachClippedFolder} - {coultClipped} / {len(os.listdir(dayFolderPath))}----------------------------------------------------------------------------------------------- ")
                    countImage+=1
                    break
            coultClipped += 1
        countday += 1

def loopCheckFile(path):
    countday = 1
    for dayFolder in os.listdir(path):
        dayFolderPath = path + "/" + dayFolder
        coultClipped = 1
        for eachClippedFolder in os.listdir(dayFolderPath):
            eachClippedFolderPath = dayFolderPath + "/" + eachClippedFolder
            countImage = 1
            if(len(os.listdir(eachClippedFolderPath))!=5):
                print(len(os.listdir(eachClippedFolderPath)))
                print(eachClippedFolderPath)

loopPathRotate("F:/ice-wheat/data/dataForProcess/MUL", "MUL", "tiltCorrected", "tilt270Degree", 270)
loopPathRotate("F:/ice-wheat/data/dataForProcess/RGB", "RGB", "tiltCorrected", "tilt270Degree", 270)
# checkFile()
loopCheckFile("F:/ice-wheat/data/dataForProcess/RGB")
loopCheckFile("F:/ice-wheat/data/dataForProcess/MUL")
# rotateAndDeleteEmptySpace('normal5.tif', 'cropped_rotated_back_image_multi_band.tif', 13)