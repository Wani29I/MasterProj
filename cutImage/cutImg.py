import math
import pprint
import pyproj
import rasterio
import numpy as np
from rasterio.mask import mask
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, mapping

def clip_raster(raster_path, coordinates, output_path):
    """
    Clips a raster image using a polygon defined by a list of coordinates.

    Args:
        raster_path: Path to the input raster image (e.g., 'image.tif').
        coordinates: List of coordinate tuples, representing the polygon's vertices:
                     [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
        output_path: Path to save the clipped raster image.
    """

    with rasterio.open(raster_path) as src:
        # Create a Polygon object from the coordinates
        geom = Polygon(coordinates)

        # Convert the Polygon to GeoJSON format
        geojson = mapping(geom)

        # Clip the raster using the polygon
        out_img, out_transform = mask(src, [geojson], crop=True)

        # Write the clipped raster to a new file
        out_meta = src.meta.copy()
        out_meta.update({
            "height": out_img.shape[1],
            "width": out_img.shape[2],
            "transform": out_transform
        })

        with rasterio.open(output_path, "w", **out_meta) as dest:
            dest.write(out_img)
            print("done")

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

cutList = openWaypointFile("preprocessedCutPointFile.txt")

# # Example usage:
# raster_path = 'testImg.tif' 
# coordinates = [[484975.45867039415,4234127.070229156], [484974.7808189663,4234127.2488881], [484974.31193539954,4234125.209086271], [484974.9897869883,4234125.03042729]]
# output_path = 'clipped_image.tif'

# clip_raster(raster_path, coordinates, output_path)