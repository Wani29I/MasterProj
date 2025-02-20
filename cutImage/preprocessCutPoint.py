import pprint
import pyproj

def openWaypointFile(filename):
    content = open(filename,'r').read()
    content = content.split("\n")
    newList = []
    for line in content:
        line = line.split(" ")
        newLine = line[:4]
        convertedLine = []
        for wayPoint in newLine:
            wayPoint = wayPoint.split(",")
            wayPoint[0] = float(wayPoint[0])
            wayPoint[1] = float(wayPoint[1])
            convertedLine.append(wayPoint)
        newList.append(convertedLine)
    return newList

def rearrange(cutPointList):
    for line in cutPointList:
        line[0], line[1] = line[1], line[0]

    return cutPointList

def lat_lon_to_epsg32654(latitude, longitude):
    """
    Converts latitude and longitude coordinates to EPSG:32654 (UTM Zone 54N).

    Args:
        latitude: Latitude in decimal degrees.
        longitude: Longitude in decimal degrees.

    Returns:
        Tuple containing the easting and northing coordinates in EPSG:32654.
    """
    try:
        # Define coordinate systems
        wgs84 = pyproj.CRS("EPSG:4326")  # WGS 84 (latitude/longitude)
        utm54n = pyproj.CRS("EPSG:32654") 

        # Create a transformation object
        transformer = pyproj.Transformer.from_crs(wgs84, utm54n)

        # Perform the coordinate transformation
        easting, northing = transformer.transform(longitude, latitude) 

        return easting, northing

    except pyproj.exceptions.CRSError:
        print("Error during coordinate transformation.")
        return None

def writeFile(cutPointList, fileName):
    # print(content)
    file = open(fileName, "a")

    for line in cutPointList:
        writeContent = str(line[0][0])+','+str(line[0][1])+' '+str(line[1][0])+','+str(line[1][1])+' '+str(line[2][0])+','+str(line[2][1])+' '+str(line[3][0])+','+str(line[3][1])
        # print(writeContent)
        file.write(writeContent)
        file.write("\n")
    file.close()

def callChangeLatLon(cutPointList):
    newData = []
    for line in cutPointList:
        newLine = []
        for wayPoint in line:
            x, y = lat_lon_to_epsg32654(wayPoint[0], wayPoint[1])
            newLine.append([x,y])
        newData.append(newLine)
    return newData

if __name__ == '__main__':
    cutPoint = openWaypointFile("cutPointList.txt")
    cutPoint = rearrange(cutPoint)
    cutPoint = callChangeLatLon(cutPoint)
    writeFile(cutPoint, "preprocessedCutPointFile.txt")
    # pprint.pprint(cutPoint)
