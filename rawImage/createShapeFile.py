import os
import pprint
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon

def getOffset(file):
    with open(file, "r") as file:
        lines = file.readlines()
    dictOffset = {}
    for offset in lines:
        offset = offset.split(' ')
        dictOffset[offset[0]] = [float(offset[1]), float(offset[2])]
    return(dictOffset)

def txt_to_shapefile(txt_file, offset):
    """
    Convert a TXT file containing sets of 4 latitude-longitude points into a rectangle shapefile.
    
    Parameters:
    - txt_file: Path to the input TXT file.
    - output_shapefile: Path to save the output shapefile (.shp).
    """
    
    rectangles = []
    ids = []
    
    with open(txt_file, "r") as file:
        lines = file.readlines()
        # pprint.pprint(lines)

    for idx,line in enumerate(lines):
        # Split the line by space and convert each pair to (lon, lat)
        coords = [tuple(map(float, point.split(','))) for point in line.strip().split()]
        # print(line)
        # print(coords)
        # rearrange cood for proper rectangle
        coords[0], coords[1] = coords[1], coords[0]

        # calculate offset for each shapefile
        for ite, eachCoords in enumerate(coords):
            coords[ite] = (coords[ite][0] + offset[1],coords[ite][1] + offset[0])

        # print(coords)
        
        # Ensure the first point is repeated to close the rectangle
        coords.append(coords[0])
        
        # Create a rectangle polygon
        rect = Polygon(coords)
        rectangles.append(rect)
        ids.append(idx+1)
        # break

    # Create a GeoDataFrame with WGS84 CRS (EPSG:4326)
    gdf = gpd.GeoDataFrame({'id': ids, 'geometry': rectangles}, crs="EPSG:4326")  
    
    # Save as a shapefile
    # gdf.to_file(output_shapefile)

    return gdf

def loopCreateShapeFile(cutPointFile, offset):
    shapefileDir = "C:/Users/pacha/Desktop/masterProj/MasterProj/rawImage/shapefileRGB"

    for key, value in offset.items():
        newShapefileDir = shapefileDir + '/' + key
        shapefile = txt_to_shapefile(cutPointFile, value)

        if not os.path.exists(newShapefileDir):
            os.makedirs(newShapefileDir)
        shapefile.to_file(newShapefileDir + '/' + key + '_shapefile.shp')

offset = getOffset("offset.txt")
# txt_to_shapefile("cutPointList.txt", "cutPointShape1.shp", offset['202406071509'])
loopCreateShapeFile("cutPointList.txt", offset)
