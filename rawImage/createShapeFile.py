import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon
import pprint

def txt_to_shapefile(txt_file, output_shapefile):
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
        coords[0], coords[1] = coords[1], coords[0]
        
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
    gdf.to_file(output_shapefile)

# Example Usage:
txt_to_shapefile("cutPointList.txt", "cutPointShape.shp")
