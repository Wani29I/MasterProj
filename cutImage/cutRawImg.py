import os
import cv2
import numpy as np
from rasterio.mask import mask
from shapely.geometry import mapping

def openRawImgPath(rawImgPath):
    ''' open rawImgPath to get raw img path in dict
    # return{
            'file1name':
                '1':
                    [filename, (x,y),(x,y),(x,y),(x,y)], ... *10
                    ...
                '115':
                    [filename, (x,y),(x,y),(x,y),(x,y)], ... *10
            'file2name':
            ...
            }
    '''
    rawImgPathDict = {}

    #loop in each file
    for file in os.listdir(rawImgPath):

        #open file
        rawImgFiles = open(rawImgPath + '/' + file,'r').read()
        #split each line
        allLines = rawImgFiles.split('\n')
        key = []
        data =[]
        #loop each line in enumerate
        for countLine, line in enumerate(allLines):

            #check if that line is key(1-115)
            try:
                int(line)
                # add to key
                key.append(line)

            #if not key >> path data
            except:

                #'filename:[x,y],[x,y],[x,y],[x,y]' >> 'filename','[x,y],[x,y],[x,y],[x,y]'
                splittedLine = line.split(':')
                
                #loop each coor >> [x y]
                coorlist = [splittedLine[0]]
                for coor in splittedLine[1].split(','):
                    #'[x y]' >> 'x y'
                    coor = coor[1:-1]
                    #'x y' >> '(x,y)'
                    coors = tuple(coor.split())
                    # append each tuple of x,y to coor list
                    coorlist.append(coors)

                #add precessed path to data
                data.append(coorlist)

        # assign key and data to dict:   key(x): data(x:x+10)
        labelDict = {}
        for count in range(len(key)):
            labelDict[key[count]] = data[count*10:count*10+10]
            
        # assign labelDict (key = 1-115) to main dict(file)
        rawImgPathDict[file[:-4]] = labelDict

    return rawImgPathDict

def getAllfilePath(filePath):
    '''
    get all of the file path in the path 
    '''
    fileList = []
    # loop into each file
    for fileName in os.listdir(filePath):
        #add original path
        fileNamePath = filePath + '/' + fileName
        #add to list
        fileList.append(fileNamePath)
    return(fileList)

def ClipSaveRawdata(rawPath, coor, outPath):
    '''
    clip and save JPG data to output path using 4 coordinates
    '''
    if(not(os.path.exists(outPath))):
        # Load image
        image = cv2.imread(rawPath)

        # Convert coordinates to integers
        coordinates = np.array(coor, dtype=np.float32)

        # Get bounding box (min/max coordinates)
        x_min, y_min = np.min(coordinates, axis=0)
        x_max, y_max = np.max(coordinates, axis=0)

        # Convert to integer values
        x_min, y_min, x_max, y_max = map(int, [x_min, y_min, x_max, y_max])

        # Crop the image
        cropped_image = image[y_min:y_max, x_min:x_max]

        # Save the cropped image
        cv2.imwrite(outPath, cropped_image)

        # print(f"Cropped image saved to {outPath}")


def loopClipSaveRawdata(rawImgPathDict, rawImgFilePathList, outputFilePathList, clipNum):
    '''
    clip and save jpg data to outpust path
    '''
    # loop into mainFileName(key (file name)) and dataLabelDict(dict value [1-115])
    mainFlieCount = 0 
    for mainFileName, dataLabelDict in rawImgPathDict.items():

        # get raw imgfile path 
        rawImgFilePath = rawImgFilePathList[mainFlieCount]
        # get output file path
        outputFilePath = outputFilePathList[mainFlieCount]

        labelCount = 1
        # loop into dataLabel(key 1-115) and fileCoorList(list value*10 [filename, coor1 - 4])
        for dataLabel, fileCoorList in dataLabelDict.items():

            # assign output file path
            LabelOutputFilePath = outputFilePath + "/RGB_" + mainFileName + "_" + str(labelCount)

            #loop only first (clipNum) number of fileCoorList
            for clipDataCount in range(clipNum):

                eachLabelOutputFilePath = LabelOutputFilePath + "/RGB_" + mainFileName + "_" + str(labelCount) + "_raw" + str(clipDataCount+1) + '.jpg'

                # separate data >> filename, [coor]
                targetData = fileCoorList[clipDataCount]
                targetFileName = targetData[0]
                targetCoor = targetData[1:]

                # assign target file path
                targetFilePath = rawImgFilePath + '/' + targetFileName + '.jpg'

                # check if file already exist
                if(not(os.path.exists(eachLabelOutputFilePath))):
                    # call ClipSaveRawdata to clip and save
                    ClipSaveRawdata(targetFilePath, targetCoor, eachLabelOutputFilePath)
                else:
                    print("image already exist")
            
            print(f"{mainFlieCount+1} / 30 >>> {labelCount} / 115")

            labelCount += 1
                
        mainFlieCount += 1

if __name__ == '__main__':
    rawImgFilePathList = getAllfilePath(filePath = "D:/ice-wheat/data/RawData/MAVICK3-12M/ALL")
    rawImgFilePathList = rawImgFilePathList[1:-1]

    rawImgPathData = "C:/Users/pacha/Desktop/masterProj/MasterProj/rawImage/rawImgPathData"
    rawImgPathDict = openRawImgPath(rawImgPathData)

    outputPath = "D:/ice-wheat/data/dataForProcess/mainData/RGB"
    outputPathList = getAllfilePath(outputPath)
    outputPathList = outputPathList[:-1]

    loopClipSaveRawdata(rawImgPathDict, rawImgFilePathList, outputPathList, 10)