#!/opt/local/bin python

    #   Code MachineLearningPatterns.py    -   This version has been customized to produce a .csv file
    #       with the correlation data in it, and to produce images of the eigenpatterns.  Other independent
    #       codes will read the .csv file and use it for machine learning in various ways.  The code
    #       uses the Scikit_Learn python package in various ways.
    #
    #   This code steps through time, updating the eigenpatterns on each time step.  
    #
    #   This code downloads data from the USGS web site.
    #
    #   This code was written on a Mac using Macports python.  A list of the ports needed to run the code are available at:
    #       https://www.dropbox.com/s/8wr5su8d7l7a30z/myports-wailea.txt?dl=0
    
    #   ---------------------------------------------------------------------------------------
    
    # Copyright 2020 by John B Rundle, University of California, Davis, CA USA
    # 
    # Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated 
    # documentation files (the     "Software"), to deal in the Software without restriction, including without 
    # limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, 
    # and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
    # 
    # The above copyright notice and this permission notice shall be included in all copies or suSKLantial portions of the Software.
    # 
    # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
    # WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
    # COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, 
    # ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

    #   ---------------------------------------------------------------------------------------


import sys
import os
import numpy as np
from array import array

import OTFileMethods

    #################################################################
    #################################################################
    
    #   BEGIN INPUTS
    
    #################################################################
    #################################################################

#   Inputs and general definitions  


NELng = -108.0
SWLng = -128.0
NELat =  42.0
SWLat = 26.0
depth = 100.0          

completeness_mag = 2.99

max_depth = 30.0                    #   For the regional catalog
mag_large = 6.0

#   ------------------------------------------------------------
#   ------------------------------------------------------------

catalog = 'LosAngeles'

#   ------------------------------------------------------------
#   ------------------------------------------------------------

#   Region Definitions and data

if catalog == 'LosAngeles':

    Location = 'Los Angeles'

    center_lat = 34.0522
    center_lng = -118.2437
    
    delta_deg = 5.0
    delta_deg_lat = 5.0
    delta_deg_lng = 5.0
    
    NELng_local = center_lng + delta_deg_lng
    SWLng_local = center_lng - delta_deg_lng
    NELat_local = center_lat + delta_deg_lat
    SWLat_local = center_lat - delta_deg_lat


#   ------------------------------------------------------------
#   ------------------------------------------------------------

start_date = "1940/01/01"       #   Events downloaded occurred after this date

region_catalog_date_start = 1940.0      #   Must be the same as start_date
region_catalog_date_end   = 2021.0

    #################################################################
    #################################################################
    
    #   END INPUTS
    
    #################################################################
    #################################################################

print('')
print('Downloading the base catalog...')
print('')
OTFileMethods.get_base_catalog(NELat, NELng, SWLat, SWLng, completeness_mag, start_date)

    #   Build the regional catalog from the World Wide catalog
    
OTFileMethods.get_regional_catalog(NELat_local, NELng_local, SWLat_local, SWLng_local, minimum_mag, max_depth,\
        region_catalog_date_start, region_catalog_date_end)
        
