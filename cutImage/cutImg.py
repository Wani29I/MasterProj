import math
import pprint
import pyproj
import rasterio
import numpy as np
from rasterio.mask import mask
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, mapping

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
            print("saved image to ", output_path)

def representRaster(filePath):
    with rasterio.open(filePath) as src:
        data = src.read(1)
        fig = plt.figure(figsize=[12,8])
        # Plot the raster data using matplotlib
        ax = fig.add_axes([0, 0, 1, 1])
        raster_image=ax.imshow(data)
        plt.show()

raster_path = "F:\ice-wheat\data\Processed\MAVIC-MUL\90\DJI_202405221319_001_processed\DJI_202405171319_001_processed_gdal_gps.tif"
coordinateList = openWaypointFile("preprocessedCutPointFile.txt")
# pprint.pprint(coordinateList)

count = 0
for coordinate in coordinateList:
    count += 1
    output_path = "../../testImg/test3/" + str(count) +".tif"
    clip_raster(raster_path, coordinate, output_path)        
    print(f"_________________________________________________________________________ {(count/115)*100:.2f} % done _________________________________________________________________________")

# representRaster("../../testImg/test3/2.tif")