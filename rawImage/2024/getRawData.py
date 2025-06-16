import os
import cv2
import pprint
import numpy as np
import easyidp as idp
import matplotlib.pyplot as plt

def getFilePath(filePath,filetype):
    fileList = []
    for fileName in os.listdir(filePath):
        fileNamePath = filePath + '/' + fileName
        for eachFile in os.listdir(fileNamePath):
            if(eachFile[-len(filetype):] == filetype):
                fileList.append(fileNamePath + '/' + eachFile)
    return(fileList)

def getAllfilePath(filePath):
    fileList = []
    for fileName in os.listdir(filePath):
        fileNamePath = filePath + '/' + fileName
        fileList.append(fileNamePath)
    return(fileList)

def getFileCoor(shapefilepath, pix4dpath, raw_img_folder):
    p4d = idp.Pix4D(pix4dpath,raw_img_folder = raw_img_folder)
    roi = idp.ROI(shapefilepath, name_field=["id"]) # seperate polygon id
    roi.change_crs(p4d.crs)
    roi.get_z_from_dsm(p4d.dsm)
    rev_results = roi.back2raw(p4d)

    img_dict_sort = p4d.sort_img_by_distance(
        rev_results, roi, 
        num=10 # only keep 10 closest images
    )
    return img_dict_sort

def show_cropped_rectangle(image_path, rectangle):
    """
    Extracts and displays the cropped region inside the rectangle.
    
    Parameters:
    - image_path: Path to the background image (JPG, PNG, etc.).
    - rectangle: NumPy array containing 4 corner points.
    """
    # Check if the image file exists
    if not os.path.exists(image_path):
        print(f"Error: The file {image_path} does not exist.")
        return
    
    # Load image
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Error: Failed to load image from {image_path}")
        return
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for Matplotlib

    # Get bounding box (min/max coordinates)
    x_min, y_min = np.min(rectangle, axis=0)
    x_max, y_max = np.max(rectangle, axis=0)

    # Convert to integer values
    x_min, y_min, x_max, y_max = map(int, [x_min, y_min, x_max, y_max])

    # Crop the image
    cropped_image = image[y_min:y_max, x_min:x_max]

    # Show the cropped region
    plt.figure(figsize=(6, 6))
    plt.imshow(cropped_image)
    plt.axis("off")  # Hide axis
    plt.show()

def writeFile(key, img_dict):
    rawImgDir = "/Users/ice/Desktop/MasterResearch/MasterProj/rawImage/rawImgPathData/"
    file = open(rawImgDir + str(key) + ".txt", "a")
    for index, fileShape in img_dict.items():
        file.write(index)
        file.write("\n")
        for fileName, shapeIndex in fileShape.items():
            content = fileName + ':' + str(shapeIndex[0]) + ',' + str(shapeIndex[1]) + ',' + str(shapeIndex[2]) + ',' + str(shapeIndex[3])
            file.write(content)
            file.write("\n")
    file.close()

if __name__ == '__main__':
    shapeFilePathList = getFilePath(filePath = "/Users/ice/Desktop/MasterResearch/MasterProj/rawImage/shapefileRGB", filetype = ".shp")
    pix4dPathList = getFilePath(filePath = "/Volumes/HD-PCFSU3-A/ice-wheat/data/Processed/MAVIC-RGB", filetype = ".p4d")
    rawImgFilePathList = getAllfilePath(filePath = "/Volumes/HD-PCFSU3-A/ice-wheat/data/RawData/MAVICK3-12M/ALL")

    shapeFilePathDict = {}
    for shapeFilePathIndex in range(len(shapeFilePathList)):
        shapeFilePath = shapeFilePathList[shapeFilePathIndex]
        shapeFilePathKey = shapeFilePath.split('/')[-2]
        shapeFilePathDict[shapeFilePathKey] = shapeFilePath

    pix4dPathDict = {}
    for pix4dPathIndex in range(len(pix4dPathList)):
        pix4dPath = pix4dPathList[pix4dPathIndex]
        pix4dPathKey = pix4dPath.split('/')[-2].split('_')[1]
        pix4dPathDict[pix4dPathKey] = pix4dPath

    rawImgFilePathDict = {}
    for rawImgFilePathListIndex in range(len(rawImgFilePathList)):
        rawImgFilePath = rawImgFilePathList[rawImgFilePathListIndex]
        rawImgFilePathKey = rawImgFilePath.split('/')[-1].split('_')[1]
        rawImgFilePathDict[rawImgFilePathKey] = rawImgFilePath

    allPath = {}
    for key in shapeFilePathDict.keys():
        if (key == '202406241205'):
            continue
        allPath[int(key)] = [shapeFilePathDict[key], pix4dPathDict[key], rawImgFilePathDict[key]]

    for key, pathList in sorted(allPath.items()):
        img_dict_sort = getFileCoor(pathList[0], pathList[1], pathList[2])
        writeFile(key, img_dict_sort)

    # img_dict_sort = getFileCoor(allPath[202406041351][0], allPath[202406041351][1], allPath[202406041351][2])
    # for index, set in img_dict_sort.items():
    #     for fileName, shape in set.items():
    #         show_cropped_rectangle(allPath[202406041351][2]+'/'+fileName+".JPG", shape[1:])

