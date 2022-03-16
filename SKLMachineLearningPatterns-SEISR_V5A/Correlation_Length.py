#!/opt/local/bin python
#import sys
#sys.path.reverse()

    #   Earthquake Methods library of methods and functions
    #   
    #   This code base collects the methods and functions used to make
    #   plots and maps of earthquake data and activity
    #
    ######################################################################

import sys
import matplotlib
import matplotlib.mlab as mlab
#from matplotlib.pyplot import figure, show
import numpy as np
from numpy import *
from mpl_toolkits.basemap import Basemap
from array import array
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.patches as patches

import datetime
import dateutil.parser

import urllib.request, urllib.parse, urllib.error
import urllib.request, urllib.error, urllib.parse
import os

import math
from math import exp

import MCUtilities
from MCUtilities import *
import MCCalcMethods

from matplotlib import cm

import http.client
from urllib.error import HTTPError

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

from numpy import median
from numpy import mean

    #####################################################################

def calculate_correlation_length(MagLo, completeness_mag):
    
    #   First estimate the correlation length in the region from the current circle catalog
    
  
    delta_mag = 0.25
    
    number_mags = int((MagLo - completeness_mag)/delta_mag)
    
    Mag_List         =  []
    Correlation_List =  []
    
    for i in range(number_mags):
    
        mag_limit = completeness_mag + float(i) * delta_mag
    
        EQ_succ_dist    =   []
        EQ_Lat          =   []
        EQ_Lng          =   []
        EQ_Mag          =   []
        
        input_file     = open("EQ_Correlation.catalog","r")
    
        for line in input_file:
            items = line.strip().split()
        
            if float(items[5]) >= mag_limit:
                EQ_Lng.append(float(items[3]))
                EQ_Lat.append(float(items[4]))
                EQ_Mag.append(float(items[5]))
    
        for i in range (len(EQ_Lat)-1):
            lat_1 = EQ_Lat[i]
            lat_2 = EQ_Lat[i+1]
            lng_1 = EQ_Lng[i]
            lng_2 = EQ_Lng[i+1]         
            great_circle_distance = compute_great_circle_distance(lat_1, lng_1, lat_2, lng_2)
            EQ_succ_dist.append(great_circle_distance)
        
        mean_distance = sum(EQ_succ_dist)/float(len(EQ_succ_dist))
        
        Mag_List.append(mag_limit)
        Correlation_List.append(mean_distance)
        
        input_file.close()
    
    

    return Mag_List, Correlation_List
    
def compute_great_circle_distance(lat_1, lng_1, lat_2, lng_2):

    # Build an array of x-y values in degrees defining a circle of the required radius

    pic = 3.1415926535/180.0
    Radius = 6371.0

    lat_1 = float(lat_1) * pic
    lng_1 = float(lng_1) * pic
    lat_2 = float(lat_2) * pic
    lng_2 = float(lng_2) * pic
    
    delta_lng = lng_1 - lng_2
    
    delta_radians = math.sin(lat_1)*math.sin(lat_2) + math.cos(lat_1)*math.cos(lat_2)*cos(delta_lng)
    delta_radians = math.acos(delta_radians)
    
    great_circle_distance = delta_radians * Radius

    return great_circle_distance
    
def write_correlation_catalog(City_Latitude, City_Longitude, Circle_Radius, earthquake_depth, MagLo, completeness_mag):

    input_file      = open("USGS_WorldWide.catalog","r")
    output_file     = open("EQ_Correlation.catalog","w")
    
    #   Compute vertices that define the circle around the city
    
    lng_circle_dg, lat_circle_dg = MCUtilities.createCircleAroundWithRadius(City_Latitude, City_Longitude, Circle_Radius)
    
    number_polygon_vertices = len(lng_circle_dg)
    
    point_list = []

    for i in range(number_polygon_vertices):
        point_list.append((float(lat_circle_dg[i]),float(lng_circle_dg[i])))
    
    polygon = Polygon(point_list)
    
#   print point_list
#       

    for line in input_file:
        items = line.strip().split()
        dep    = items[6]
        mag    = items[5]
        eq_lat = items[4]
        eq_lng = items[3]
        
        point = Point((float(eq_lat),float(eq_lng)))
        
        if (float(dep) <= float(earthquake_depth) and float(mag) >= float(completeness_mag) and polygon.contains(point) == True):
            print(items[0],items[1],items[2],items[3],items[4],items[5],items[6], file=output_file)
        
    output_file.close()
    input_file.close()

    return
    
if __name__ == '__main__':

    #   Generally choose completeness and min mag = 2.0
    
    
    CircleLat       = 35.6895
    CircleLng       = 139.6917
    CircleRadius    = 1000.0
    fit_range       = [5.75,7.75]
    Location        = 'Tokyo, Japan'
    
    CircleLat       = -6.0288
    CircleLng       = 106.8456
    CircleRadius    = 1000.0
    fit_range       = []
    Location        = 'Jakarta, Indonesia'
    
    CircleLat       = 25.0330
    CircleLng       = 121.5654
    CircleRadius    = 1000.0
#    fit_range       = [6,7]
    fit_range       = [5.75,7.25]
    Location        = 'Taipei, Taiwan'
    
    CircleLat       = -33.4489
    CircleLng       = -70.6693
    CircleRadius    = 1000.0
    fit_range       = [6,7.5]
    Location        = 'Santiago, Chile'
    
    CircleLat       = 37.9838
    CircleLng       = 23.7275
    CircleRadius    = 500.0
    fit_range       = []
    Location        = 'Athens, Greece'

    CircleLat       = -9.4438
    CircleLng       = 147.1803
    CircleRadius    = 1000.0
#   fit_range       = [6,7.5]
    fit_range       = [4.5,6.0]
    Location        = 'Port Moresby'

    CircleLat       = 14.5995
    CircleLng       = 120.9842
    CircleRadius    = 1000.0
    fit_range       = [6,7.5]
    Location        = 'Manila, Philippines'
    
    CircleLat       = 53.2194
    CircleLng       = 6.5665
    CircleRadius    = 200.0
    fit_range       = [2.75,5]
    Location        = 'Groningen, Netherlands'
    
    CircleLat       = 34.0522
    CircleLng       = -118.2437
    CircleRadius    = 500.0
    fit_range       = [4,6.5]
    Location        = 'Los Angeles, CA'

    CircleLat       = 32.13
    CircleLng       = -115.30
    CircleRadius    = 300.0
    fit_range       = [4,6.5]
    Location        = 'EMC Earthquake'

    #   For this area choose completeness and min mag = 0.0
    CircleLat       = 38.7749
    CircleLng       = -122.7553
    CircleRadius    = 100.0
    fit_range       = [2,5]
    Location        = 'The Geysers, CA'
    
    CircleLat       = 19.4069
    CircleLng       = -155.2834
    CircleRadius    = 200.0
    fit_range       = [2.75,5]
    Location        = 'Kilauea, HI'
    
    CircleLat       = 31.5833
    CircleLng       = 130.6500
    CircleRadius    = 100.0
    fit_range       = [4.75,6]
    Location        = 'Sakurajima, Japan'   #   Gives b = 0.71 for fit from 4.75 to 6.0
    
    CircleLat       = 15.1429
    CircleLng       = 120.3496
    CircleRadius    = 200.0
    fit_range       = [4.75,6]
    Location        = 'Pinatubo, Philippines'
    
    CircleLat       = 37.7510
    CircleLng       = 14.9934
    CircleRadius    = 200.0
    fit_range       = [4.75,6]    
    Location        = 'Mt. Etna, Sicily'
    
    CircleLat       = 35.4867
    CircleLng       = -96.6850
    CircleRadius    = 200.0
    fit_range       = [3,5]
    Location        = 'Prague, OK'
    
    CircleLat       = 47.0
    CircleLng       = -125.0
    CircleRadius    = 300.0
    fit_range       = [3,5]
    Location        = 'Cascadia'
    
    fit_range = []
    
    #   -----------------------------------------
    #
    #   Active location

    CircleLat       = 34.0522
    CircleLng       = -118.2437
    fit_range       = [4,6.5]
    Location        = 'Los Angeles, CA'

    #    
    #   -----------------------------------------

    #   -----------------------------------------
    #   
    
    MagLo = 6.0
    completeness_mag = 3.0
    
    CircleRadius = 800.0                 #   Defines radius in km of hazard circle around the city lat-lon (variable)
    depth = 100.0                       #   Defines maximum earthquake depth (variable)
    completeness_mag_uniform = 2.99     # 
    earthquake_depth = 100.0
    
    write_correlation_catalog(CircleLat, CircleLng, CircleRadius, earthquake_depth, MagLo, completeness_mag)
    
    Mag_List, Correlation_List = calculate_correlation_length(MagLo, completeness_mag)
    
    plt.plot(Mag_List, Correlation_List, 'or', ms=4)
    
    plt.show()
    