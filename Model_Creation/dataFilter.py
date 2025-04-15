import os
import cv2
import pprint
import rasterio
import numpy as np
from PIL import Image
from pathlib import Path
from datetime import datetime
from rasterio.mask import mask
from rasterio.plot import reshape_as_image
from shapely.geometry import Polygon, mapping

def openAndSplitData(dataFilePath):
    # open data path file
    returnData = []
    with open(dataFilePath, "r") as file:
        allData = file.read().splitlines()

    for data in allData:
        returnData.append(data.split(','))

    return returnData

def getDataByDataKeyDate(allData, dateList):
    ''' 
    get labelled data by date: ["202404251118", "202404301146", "202405071327", "202405131248", "202405171307", "202405221319", "202405271230", "202405311536", "202406041351", "202406071509", "202406111255", "202406141237", "202406171112", "202406241205"]
    '''
    returnData = []
        
    # loop into each line (date)
    for dataLine in allData:

        # get data date
        date = dataLine[1].split('_')[0]

        # check if date in datelist
        if (date in dateList):
            returnData.append(dataLine)

    return returnData

def selectDataByRawImgType(allData, selectedRawImgKey):
    '''
    filter data from original, raw1 - raw10 to only selected one
    '''
    returnData = []

    # loop into every data
    for eachData in allData:

        # get image type : original / raw1-raw10
        imageType = eachData[0].split("_")[-3]

        # check if data's img type in selected one
        if(imageType in selectedRawImgKey):
            returnData.append(eachData)

    return returnData

def selectDataByAugmentMethod(allData, selectedAugmentedMethod):
    '''
    filter data by augmentation method:
    '''
    returnData = []

    # loop into every data
    for eachData in allData:

        # get augmented method
        augmentedMethod = (eachData[0].split("_")[-1].split('.')[0])

        # check if data's img type in selected one
        if (augmentedMethod in selectedAugmentedMethod):
            returnData.append(eachData) 

    return returnData

def selectDataByDate(allData, selectedDate):
    '''
    filter data by date in path
    '''
    returnData = []

    # loop into every data
    for eachData in allData:

        # get augmented method
        eachDataDate = eachData[0].split("_")[-5]

        # check if data's img type is in selected one
        if (eachDataDate in selectedDate):
            returnData.append(eachData)

    return returnData

def filterDataColumn(allData, dataColumn, dataFilter):
    ''' 
    filter and get only selected data column
    '''

    returnData = []

    # get the index of each column to prepare creating new data list
    allIndex = []
    for eachFilter in dataFilter:
        allIndex.append(dataColumn.index(eachFilter))

    # loop into every data
    for data in allData:

        # get only the selected one
        filteredData = []
        # loop into each index to append to filtered data
        for index in allIndex:
            filteredData.append(data[index])
        # append each filteredData to returnData
        returnData.append(filteredData)

    return returnData

def filterData(allData, dateKeyList = [], dateList = [], ImgTypeList = [], AugmentMethodList = [], ColumnList = [], dataColumn = []):
    ''' 
    cal select data function to filter data
    input = alldata (data list)
    dateKeyList = [list of wanted datekey]
    dateList = [list of wanted date]
    ImgTypeList = [list of image type: original, raw1-raw10]
    AugmentMethodList = [list of wanted augment method]
    ColumnList = [list of wanted column]
    '''

    returnData = allData
    # check if dateKeyList is also selected to be filtered
    if(dateKeyList != []):
        returnData = getDataByDataKeyDate(returnData, dateKeyList)
    # check if dateList is also selected to be filtered
    if(dateList != []):
        returnData = selectDataByDate(returnData, dateList)
    # check if ImgTypeList is also selected to be filtered
    if(ImgTypeList != []):
        returnData = selectDataByRawImgType(returnData, ImgTypeList)
    # check if AugmentMethodList is also selected to be filtered
    if(AugmentMethodList != []):
        returnData = selectDataByAugmentMethod(returnData, AugmentMethodList)
    # check if ColumnList is also selected to be filtered
    if(ColumnList != []):
        returnData = filterDataColumn(returnData, dataColumn, ColumnList)

    return returnData

def deleteNull(filteredData):
    '''
    delete line if line have null value
    '''
    returnData = []
    null = False

    # loop into each line
    for eachLine in filteredData:
        
        # loop check each data if the data is '' (null)
        for data in eachLine:
            if(data == ''):
                # if null: set null = True
                null = True

        # if not null: append to return data
        if(not null):
           returnData.append(eachLine)

           # set null value back to False
        null = False

    print("original data:", len(filteredData), "----- deleted null data:", len(returnData))

    return returnData

def addTime(dataList, selectedDataColumn):
    '''
    add time to data
    get data list and selected column and return new of both
    '''
    returnData = []

    # loop into every line of data
    for dataLine in dataList:
        DataKey = dataLine[0]

        # calculate time from data key
        hhmm = int(DataKey[8:12])
        time_float = (hhmm // 100) + (hhmm % 100) / 60.0  

        # add time to each line
        dataLine.append(str(time_float))
        returnData.append(dataLine)

    # add column "time"
    selectedDataColumn.append('time')

    return returnData, selectedDataColumn

def getDayFromImagePath(allDataList, selectedDataColumn):
    returnData = []

    # add "days" to data column
    selectedDataColumn.append("days")

    # plant date
    plantDate = "20231101"
    plantDate = datetime.strptime(plantDate, "%Y%m%d")

    # loop into every line
    for dataLine in allDataList:

        # get only datekey from RGB path
        dateTime = dataLine[1].split('/')[2].split('_')[1]

        # get date from dateTime
        date = str(dateTime[:8])
        date = datetime.strptime(date, "%Y%m%d")

        # Calculate Days After Sowing (DAS)
        days = (date - plantDate).days

        dataLine.append(str(days))
        returnData.append(dataLine)


    return returnData, selectedDataColumn

def writeFileCSV(dataList, fileName):
    '''
    write file as csv 
    '''
    file = open(fileName, "a")

    for data in dataList:
        dataCSV = ",".join(data)
        file.write(dataCSV)
        file.write("\n")

    file.close()

# dataFilePath = "D:/ice-wheat/data/dataForProcess/mainData/completeLabelData.txt"
# dataKeyDateList = ["202404251118", "202404301146", "202405071327", "202405131248", 
#                    "202405171307", "202405221319", "202405271230", "202405311536", 
#                    "202406041351", "202406071509", "202406111255", "202406141237", 
#                    "202406171112", "202406241205", "999999999999"]
# rawImgKey = [ "original", "raw1", "raw2", "raw3", "raw4", "raw5", "raw6", "raw7", "raw8", "raw9", "raw10"]
# augmentMethod = ['original', 'flipped', 'rotated', 'zoomed', 'brightenOriginal', 'darkenOriginal', 'brightenFlipped', 'darkenFlipped', 'jittered', 'noisy']
# dataColumn = [ "imagePath", "DataKey","DATE", "Height", "SPAD", "LAI", "leafWidth", "leafLength", 
#               "centerEarWeight", "centerEarNum", "sideEarWeight", "sideEarNum", "totEarWeight",	
#               "totEarNum", "avgEarSize", "20StrawWeightBeforeDry", "20StrawWeightAfterDry", 
#               "strawWeightDecreasePercent", "totalSeedNum", "seedNumLessThan2MM", "totalSeedWeightBeforeDry", 
#               "seedLessThan2MMWeightBeforeDry", "totalSeedWeightAfterDry", "seedLessThan2MMWeightAfterDry", "DSMPath"]
# dateList = ['202401181250', '202401221100', '202401291321', '202402071317', '202402081107', 
#             '202402131116', '202402191131', '202402261154', '202403041133', '202403111217', 
#             '202403191047', '202403251407', '202404011045', '202404101010', '202404151134', 
#             '202404171400', '202404221142', '202404251118', '202404301146', '202405071327', 
#             '202405131248', '202405171307', '202405221319', '202405271230', '202405311536', 
#             '202406041351', '202406071509', '202406111255', '202406141237', '202406171112', 
#             '202406241205']
# testDataFilter = ["imagePath", "DataKey", "DATE", "Height", "SPAD", "LAI", "centerEarWeight", "centerEarNum", "sideEarWeight", "sideEarNum", "avgEarSize", "totalSeedNum", "totalSeedWeightBeforeDry"]

# allData = openAndSplitData(dataFilePath)
# databyDataDateKey = getDataByDataKeyDate(allData, dataKeyDateList[0:-1])
# selectedDataImgType = selectDataByRawImgType(allData, rawImgKey[0:4])
# selectedDataAugMethod = selectDataByAugmentMethod(allData, augmentMethod)
# dataByDate = selectDataByDate(allData, dateList[17:])
# filteredData = filterDataColumn(allData, dataColumn, testDataFilter)

if __name__ == '__main__':
    dataFilePath = "/Volumes/HD-PCFSU3-A/ice-wheat/data/dataForProcess/mainData/completeLabelDataLinkedDSM.txt"
    dataFilePath = "D:/ice-wheat/data/dataForProcess/mainData/completeLabelDataLinkedDSM.txt"
    allDateList = ['202401181250', '202401221100', '202401291321', '202402071317', '202402081107', 
            '202402131116', '202402191131', '202402261154', '202403041133', '202403111217', 
            '202403191047', '202403251407', '202404011045', '202404101010', '202404151134', 
            '202404171400', '202404221142', '202404251118', '202404301146', '202405071327', 
            '202405131248', '202405171307', '202405221319', '202405271230', '202405311536', 
            '202406041351', '202406071509', '202406111255', '202406141237', '202406171112', 
            '202406241205']
    selectedDataKeyDateList = ["202404251118", "202404301146", "202405071327", "202405131248", 
                    "202405171307", "202405221319", "202405271230", "202405311536", 
                    "202406041351", "202406071509", "202406111255", "202406141237", 
                    "202406171112"]
    selectedRawImgKey = [ "original", "raw1", "raw2", "raw3"]
    augmentMethod = ['original', 'flipped', 'rotated', 'zoomed', 'brightenOriginal', 'darkenOriginal', 'brightenFlipped', 'darkenFlipped', 'jittered', 'noisy']
    dataColumn = [ "rgb", "DataKey","DATE", "Height", "SPAD", "LAI", "leafWidth", "leafLength", 
                "centerEarWeight", "centerEarNum", "sideEarWeight", "sideEarNum", "totEarWeight",	
                "totEarNum", "avgEarSize", "20StrawWeightBeforeDry", "20StrawWeightAfterDry", 
                "strawWeightDecreasePercent", "totalSeedNum", "seedNumLessThan2MM", "totalSeedWeightBeforeDry", 
                "seedLessThan2MMWeightBeforeDry", "totalSeedWeightAfterDry", "seedLessThan2MMWeightAfterDry", "dsm"]
    
    # select data to be filtered
    selectedDataColumn = ["DataKey", "rgb", "dsm"]

    # get all data from file
    allData = openAndSplitData(dataFilePath)

    # filter data
    filteredData = filterData(allData, [], selectedDataKeyDateList, selectedRawImgKey, augmentMethod, selectedDataColumn, dataColumn)

    # delete data line with null
    finalData = deleteNull(filteredData)

    # add time to data
    # finalData, selectedDataColumn = addTime(finalData, selectedDataColumn)

    # add date to data
    finalData, selectedDataColumn = getDayFromImagePath(finalData, selectedDataColumn)

    # add data column
    finalData.insert(0, selectedDataColumn)

    # save data as csv
    writeFileCSV(finalData, "DataKey_RGB_DSM_days__NoERR_From3thM.csv")
