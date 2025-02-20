import os
import math
import pprint
import pyproj
import rasterio
import numpy as np
from rasterio.mask import mask
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, mapping
from cutImg import openWaypointFile, clip_raster, representRaster
