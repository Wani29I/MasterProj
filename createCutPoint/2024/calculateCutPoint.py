import numpy as np
import math
import pprint

def calAlphaBetaWaypoint(content):

    countPoint = 0
    calNum = 0
    summo = 0
    sumno = 0
    summn = 0
    for i in (content):
        if(countPoint>=7):
            M = content[countPoint]
            N = content[countPoint-7]
            my = float(M[1])
            mx = float(M[0])
            ny = float(N[1])
            nx = float(N[0])
            mo = my-ny
            no = nx-mx
            mn = math.sqrt((mo**2) + (no**2))
            summo += mo
            sumno += no
            summn += mn
            calNum+=1
        countPoint+=1

    avgmo = summo/calNum
    avgno = sumno/calNum
    avgmn = summn/calNum
    alpha = avgno/2
    beta = avgmo/2

    # print( avgmo, avgno, avgmn, alpha, beta )
    # testx = 140.828141869 - alpha
    # testy = 38.255167122 + beta
    # testxx = 140.828141869 + alpha
    # testyy = 38.255167122 - beta
    # print(testx, testy, testxx, testyy)
    return(alpha,beta)

def writeCutPointFile(cutPointList, fileName):
    # print(content)
    file = open(fileName, "a")
    file.write("<?xml version='1.0' encoding='utf-8' ?>")
    file.write("\n")
    file.write('<gpx xmlns="http://www.topografix.com/GPX/1/1" xmlns:b="http://www.bizstation.jp/waypoint_extension" creator="Drogger GPS for Android 2.12.229" version="1.1">"')

    for point in cutPointList:
        pointAX = point[0][0]
        pointAY = point[0][1]
        pointBX = point[1][0]
        pointBY = point[1][1]

        writeContent = '<wpt lat="' + str(pointAY) + '" lon="' + str(pointAX) + '">'
        file.write("\n")
        file.write(writeContent)
        file.write("\n")
        file.write("</wpt>")

        writeContent = '<wpt lat="' + str(pointBY) + '" lon="' + str(pointBX) + '">'
        file.write("\n")
        file.write(writeContent)
        file.write("\n")
        file.write("</wpt>")
        # <wpt lat="38.255167122" lon="140.828141869">
        # </wpt>

    file.write("\n")
    file.write("</gpx>")
    file.close()

def calculateSideWaypoint(content, xRange, yRange):
    newWaypointList = []
    for eachWaypoint in content:
        coordinate = eachWaypoint
        coorX = coordinate[0]
        coorY = coordinate[1]
        leftCoorX = coorX - xRange
        leftCoorY = coorY + yRange
        rightCoorX = coorX + xRange
        rightCoorY = coorY - yRange
        newWaypointList.append([leftCoorX, leftCoorY])
        newWaypointList.append([coorX, coorY])
        newWaypointList.append([rightCoorX, rightCoorY])
    return newWaypointList

def writeWayPointFileSimple(cutPointList, fileName):
    # print(content)
    file = open(fileName, "a")
    file.write("<?xml version='1.0' encoding='utf-8' ?>")
    file.write("\n")
    file.write('<gpx xmlns="http://www.topografix.com/GPX/1/1" xmlns:b="http://www.bizstation.jp/waypoint_extension" creator="Drogger GPS for Android 2.12.229" version="1.1">"')

    for point in cutPointList:
        writeContent = '<wpt lat="' + str(point[1]) + '" lon="' + str(point[0]) + '">'
        file.write("\n")
        file.write(writeContent)
        file.write("\n")
        file.write("</wpt>")
        # <wpt lat="38.255167122" lon="140.828141869">
        # </wpt>

    file.write("\n")
    file.write("</gpx>")
    file.close()

def meargePointList(mainList, subList):
    for eachContent in subList:
        mainList.append(eachContent)
    return mainList

def openWaypointFile(filename):
    content = open(filename,'r').read()
    content = content.split("\n")
    newList = []
    for waypoint in content:
        coor = waypoint.split()
        x = float(coor[1])
        y = float(coor[0])
        newList.append([x,y])
    return newList

def calLong(content):
    countPoint = 0
    calNum = 0
    summo = 0
    sumno = 0
    for i in (content):
        if((countPoint+1)%7 != 0 and countPoint != len(content)-1):
            M = content[countPoint]
            N = content[countPoint+1]
            my = M[1]
            mx = M[0]
            ny = N[1]
            nx = N[0]
            mo = my-ny
            no = mx-nx
            summo += mo
            sumno += no
            calNum+=1
            # print(M)
            # print(N)
            # print(calNum)
        countPoint+=1

    avgy = summo/calNum
    avgx = sumno/calNum
    # testx = 140.828141869 - avgx
    # testy = 38.255167122 - avgy
    # print(testx, testy)
    return avgx, avgy

def addRightEndpoint(content, xavg,yavg):
    newListWithEndPoint = []
    for waypoint in content:
        waypointX = waypoint[0]
        waypointY = waypoint[1]
        startPoint = [waypointX, waypointY]
        endPoint = [waypointX-xavg, waypointY-yavg]
        newListWithEndPoint.append([startPoint,endPoint])

    return newListWithEndPoint

def preprocessWaypointList(content):
    newList = []
    count = 0 
    for waypoint in content:
        coor = waypoint.split()
        x = float(coor[1])
        y = float(coor[0])
        newList.append([x,y])
        count += 1 
    return(newList)

def clearEndpoint(content):
    removeList = []
    for i in range(len(content)):
        if( i!=0 and content[i-1][1] < content[i][1] ):
            removeList.append(content[i-1])
    for i in removeList:
        content.remove(i)
    return content

def calSidePointForCut(content, alpha, beta):
    cutPointList = []
    for point in content:
        for coordinate in point:
            coorX = coordinate[0]
            coorY = coordinate[1]
            # print(coordinate)
            pointAX = coorX-alpha
            pointAY = coorY+beta
            pointBX = coorX+alpha
            pointBY = coorY-beta
            pointA = [pointAX, pointAY]
            pointB = [pointBX, pointBY]
            cutPointList.append([pointA, pointB])
            # testx = 140.828141869 - alpha
            # testy = 38.255167122 + beta
            # testxx = 140.828141869 + alpha
            # testyy = 38.255167122 - beta
    return cutPointList

def writeWayPointFile(content, fileName):
    # print(content)
    file = open(fileName, "a")
    file.write("<?xml version='1.0' encoding='utf-8' ?>")
    file.write("\n")
    file.write('<gpx xmlns="http://www.topografix.com/GPX/1/1" xmlns:b="http://www.bizstation.jp/waypoint_extension" creator="Drogger GPS for Android 2.12.229" version="1.1">"')

    for x in content:
        for point in x:
            writeContent = '<wpt lat="' + str(point[1]) + '" lon="' + str(point[0]) + '">'
            file.write("\n")
            file.write(writeContent)
            file.write("\n")
            file.write("</wpt>")
            # <wpt lat="38.255167122" lon="140.828141869">
            # </wpt>

    file.write("\n")
    file.write("</gpx>")
    file.close()

def writeCutList(content, fileName):
    # print(content)
    file = open(fileName, "a")

    for count in range(int(len(content)/2)):
        writeContent = ''
        # print(content[count*2][0], content[count*2][1], content[count*2+1][0], content[count*2+1][1])
        for point in content[count*2]:
            writeContent += str(point[0]) + ',' + str(point[1]) + ' '
        for point in content[count*2+1]:
            writeContent += str(point[0]) + ',' + str(point[1]) + ' '

        
        print (writeContent)
        


            # writeContent = '<wpt lat="' + str(point[1]) + '" lon="' + str(point[0]) + '">'
            # file.write("\n")
            # file.write(writeContent)
            # file.write("\n")
            # file.write("</wpt>")
            # <wpt lat="38.255167122" lon="140.828141869">
            # </wpt>

        file.write(writeContent)
        file.write("\n")
    file.close()

if __name__ == '__main__':
    content = openWaypointFile("waypoint.txt") 
    rightContent = openWaypointFile("rightFieldWaypoint.txt") 
    allRightContent = openWaypointFile("allRightFieldWaypoint.txt") 

    alpha, beta = calAlphaBetaWaypoint(content)
    longx, longy = calLong(content)
    content = clearEndpoint(content)


    allWayPoint = meargePointList(content, allRightContent)
    allWayPoint = addRightEndpoint(allWayPoint, longx, longy)
    cutPointList = calSidePointForCut(allWayPoint, alpha, beta)

    # pprint.pp(cutPointList)
    # print(np.shape(cutPointList))
    # writeCutPointFile(cutPointList, "testWayPoint.gpx")

    # newWaypointList = calculateSideWaypoint(rightContent, alpha*2, beta*2)
    # print(newWaypointList)
    # writeWayPointFileSimple(newWaypointList, "allRightFieldWaypoint.gpx")
    # writeCutList(cutPointList, "cutPointList.txt")