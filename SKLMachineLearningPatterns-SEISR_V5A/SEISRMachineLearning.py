#!/opt/local/bin python

    #   OTTimeSeries.py    -   This version will use the correlations to form time series,
    #                                       then do time series forecasting
    #
    #   This is an implementation of the code at:
    #
    #   https://machinelearningmastery.com/random-forest-for-time-series-forecasting/
    #
    #   Python code to use Scikit_Learn to identify earthquake alerts
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

import SEISRCalcMethods
import SEISRFileMethods
import SEISRPlotMethods
import SEISRUtilities

import datetime
import dateutil.parser

import time
from time import sleep  #   Added a pause of 30 seconds between downloads

import math

from tabulate import tabulate

#  Now we import the sklearn methods
import pandas as pd
import numpy as np
import scipy

from numpy import asarray
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from scipy.integrate import simps
from numpy import trapz

import itertools

#from sklearn.datasets import load_iris     #   Don't need this dataset

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor

import random
    

from matplotlib import pyplot

   
    
if __name__ == '__main__':


    #################################################################
    #################################################################
    
    #   BEGIN LOCATION INPUTS
    
    #################################################################
    #################################################################

    NELng = -108.0
    SWLng = -128.0
    NELat =  42.0
    SWLat = 26.0
    depth = 100.0          

    completeness_mag = 2.99

    max_depth = 30.0                    #   For the regional catalog

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
        
#         grid_size = 0.33333
#         grid_size = 0.2
        grid_size = 5.0
#         grid_size = 0.66666
#         grid_size = 0.5
#         grid_size = 1.0
#         grid_size = 2.0
    
        NELng_local = center_lng + delta_deg_lng
        SWLng_local = center_lng - delta_deg_lng
        NELat_local = center_lat + delta_deg_lat
        SWLat_local = center_lat - delta_deg_lat

        start_date = "1940/01/01"       #   Events downloaded occurred after this date

        region_catalog_date_start = 1940.0      #   Must be the same as start_date
        region_catalog_date_end   = 2022.0

#   ------------------------------------------------------------
#   ------------------------------------------------------------

    #################################################################
    #################################################################
    
    #   BEGIN PARAMETER INPUTS
    
    #################################################################
    #################################################################
    
    month_interval          = 0.07692
    
    plot_start_year = 1970.0 - 2.*month_interval
#     plot_start_year = 2020.0 - 2.*month_interval
    data_string_title = catalog
    Grid = str(delta_deg)
    

    
    delta_time     =  month_interval
        
    mag_large_plot = 6.0
    mag_large = 6.75
    min_mag = 3.75
    min_mag = 3.29
    lower_mag = 4.0
        
    NSteps = 36             #   Defines the EMA averaging time
    #   lower_cutoff can't be 0
#     lower_cutoff = 1.7      #  In units of mean number of small eqs.  Intervals with fewer are augmented
    lower_cutoff = 1.8
    
#     lower_cutoff = 0.
    
    NSTau = 36   #   decay constant in months for the intensity map
    
    forecast_intervals = [3.0]
    
    number_thresholds = 500
    
    #################################################################
    #################################################################
    
    #   END INPUTS
    
    #################################################################
    #################################################################
    
    #   .............................................................
    
    Download_Catalog                    = False
    
    Plot_SEISR_Timeseries               = False
    Plot_Event_Timeseries               = False
    Plot_Precision_Timeseries           = False
    
    Plot_Temporal_ROC                   = False
    Plot_Spatial_ROC                    = False
    
    Plot_Precision_v_Threshold          = False
    Compute_Random_Statistics           = False

    Optimize_Params                     = False
    Plot_Seismicity_Map                 = False
    Compute_Timeseries                  = False
    Plot_Total_RI                       = False
    Plot_Mean_EQs_Time                  = False
    
    #   Movies
    Plot_Timeseries_PPV_Movie           = False
    Plot_Timeseries_ACC_Movie           = False
    Plot_RTI_Time_Slice_Movie           = False
    Plot_Combine_Images                 = False

    
    #   .............................................................

    if Download_Catalog:
        print('')
        print('Downloading the base catalog...')
        print('')
        SEISRFileMethods.get_base_catalog(NELat, NELng, SWLat, SWLng, completeness_mag, start_date)

    #   Build the regional catalog from the World Wide catalog
    
        SEISRFileMethods.get_regional_catalog(NELat_local, NELng_local, SWLat_local, SWLng_local, min_mag, max_depth,\
            region_catalog_date_start, region_catalog_date_end)
            
    #   .............................................................
    
    year_large_eq, mag_large_eq, index_large_eq = SEISRCalcMethods.get_large_earthquakes(mag_large,min_mag)
    
    print('')
    print('year_large_eq: ', year_large_eq)
    print('mag_large_eq: ', mag_large_eq)
    print('')
    
    timeseries, time_bins, date_bins = SEISRCalcMethods.coarse_grain_seismic_timeseries(\
                NELat_local, NELng_local, SWLat_local, SWLng_local, \
                min_mag, max_depth, grid_size, delta_time)
                
    #   .............................................................
    #
    #   Define the fundamental list eqs_list from the timeseries
                
    time_list = time_bins
    
    eqs_list            =   []
    
    #   Compute the list of earthquakes for all spatial bins
    for i in range(len(timeseries[0])):         #   This sums over all times
        eqs_sum = 0.
        
        for j in range(len(timeseries)):        #   This j index sums over all spatial grid boxes for a given time index i
            eqs_sum += timeseries[j][i]
        
        if eqs_sum < 1:                         #   If no earthquakes that month
            eqs_sum = 1.0
            
        eqs_list.append(eqs_sum)
        
    #   .............................................................
    #
    #   Define the weights of the of the timeseries for each grid box 
        
    timeseries_weights  =   []
    
    for i in range(len(timeseries[0])): 
        weight_working_list = []
        
        for j in range(len(timeseries)):        #   This j index sums over all spatial grid boxes for a given time index i
            weight_working_list.append(timeseries[j][i]/eqs_list[j])
            
        timeseries_weights.append(weight_working_list)   #  First index is grid box number, second index will be time index

    #   .............................................................
    
    
    #   Adjust earthquake time series to have a minimum value            
    
    excluded_months = int(forecast_intervals[0] * 13)
    eqs_list_excluded = eqs_list[:-excluded_months]
    
    mean_eqs_excluded = round(np.mean(eqs_list_excluded),3)
    mean_eqs    = round(np.mean(eqs_list),3)
    
    
#     print('mean_eqs, mean_eqs_excluded: ',mean_eqs, mean_eqs_excluded )
#     std_eqs     = round(np.std(eqs_list),3)
#     median_eqs  = round(np.median(eqs_list),3)
    
#     print('mean, stdev, median:', mean_eqs, std_eqs, median_eqs)
    
    min_rate = lower_cutoff * mean_eqs_excluded
    
    for i in range(len(time_list)):
        if int(eqs_list[i]) <= min_rate:
            eqs_list[i] = min_rate

    #   Apply Exponential Moving Average to eqs_list
    eqs_list = SEISRCalcMethods.timeseries_to_EMA(eqs_list, NSteps)

    
    eqs_list_rounded = [round(eqs_list[i],0) for i in range(len(eqs_list))]

    #   Generate the SEISR (inverse seismicity rate) times and filter the timeseries data to occur only after the plot_start_year
    
    time_list_reduced, log_number_reduced, eqs_list_reduced = \
                SEISRCalcMethods.calc_seisr_timeseries(time_list, eqs_list, plot_start_year, mag_large,min_mag, delta_time)
                
    times_window = time_list_reduced
    values_window = log_number_reduced
    
    #   .............................................................
        
    number_points_to_plot = 0
    
    for k in range(len(time_list)):
        if time_list[k] >= plot_start_year:
            number_points_to_plot += 1
            
    NN = number_points_to_plot
    
    #   .............................................................
    #   Define the EMA timeseries
    
    timeseries_EMA = []
    for i in range(len(timeseries)):             #   Index i runs over all grid boxes
        EMA_working = []
        for j in range(len(timeseries[0])):      #   Index j runs over all time indices
            EMA_working.append(timeseries[i][j])

        EMA_out_list = SEISRCalcMethods.timeseries_to_EMA(EMA_working, NSteps) # EMA average over timeseries in the ith grid box
        
        timeseries_EMA.append(EMA_out_list) #   These are the timeseries in the grid boxes converted to EMA timeseries
        
    #   .............................................................
        
    if Plot_SEISR_Timeseries:
    
        for i in range(len(forecast_intervals)):
    
            forecast_interval = forecast_intervals[i]
            
            true_positive, false_positive, true_negative, false_negative, threshold_value\
                     = SEISRCalcMethods.compute_ROC(times_window, values_window, forecast_interval, \
                     mag_large, min_mag, number_thresholds, 0, 0)
                
            SEISRPlotMethods.plot_seisr_timeseries(time_list_reduced, log_number_reduced, plot_start_year, mag_large_plot,\
                NELng_local, SWLng_local, NELat_local, SWLat_local, Location, NSteps, delta_time, min_mag, lower_cutoff, min_rate,\
                forecast_interval, number_thresholds, \
                true_positive, false_positive, true_negative, false_negative, threshold_value)
                
    #   .............................................................
    
    if Plot_Event_Timeseries:
    
        time_list_unadjusted, eqs_list_unadjusted = SEISRCalcMethods.read_input_earthquake_data(delta_time, min_mag,0)
        
        NSteps = 36
        eqs_list_EMA = SEISRCalcMethods.timeseries_to_EMA(eqs_list_unadjusted, NSteps)
        
        
        eqs_list_EMA = [round(eqs_list_EMA[i],2) for i in range(len(eqs_list_EMA))]
        
        time_list_unfiltered_reduced, eqs_list_unfiltered_reduced = \
                SEISRCalcMethods.calc_eqs_unfiltered(time_list_unadjusted, eqs_list_unadjusted, plot_start_year)
                
        time_list_EMA_reduced, eqs_list_EMA_reduced = \
                SEISRCalcMethods.calc_eqs_unfiltered(time_list_unadjusted, eqs_list_EMA, plot_start_year)
    
        SEISRPlotMethods.plot_event_timeseries(time_list_unfiltered_reduced, eqs_list_unfiltered_reduced, eqs_list_EMA_reduced,\
                plot_start_year, mag_large_plot,\
                NELng_local, SWLng_local, NELat_local, SWLat_local, Location, NSteps, delta_time, min_mag, lower_cutoff)
                
    #   .............................................................
                
    if Plot_Temporal_ROC:

        for i in range(len(forecast_intervals)):
        
            forecast_interval = forecast_intervals[i]
                       
            true_positive, false_positive, true_negative, false_negative, threshold_value\
                     = SEISRCalcMethods.compute_ROC(times_window, values_window, forecast_interval, \
                    mag_large, min_mag, number_thresholds, 0, 0)
        
            SEISRPlotMethods.plot_temporal_ROC(values_window, times_window, true_positive, false_positive, true_negative, false_negative, \
                    threshold_value, forecast_interval, mag_large, min_mag,plot_start_year,\
                    data_string_title, number_thresholds, NELng_local, SWLng_local, NELat_local, SWLat_local, \
                    Grid, Location, NSteps, delta_time, lower_cutoff, min_rate)
                    
    #   .............................................................
    
    if Plot_Precision_v_Threshold:

        for i in range(len(forecast_intervals)):
    
            forecast_interval = forecast_intervals[i]
            
            true_positive, false_positive, true_negative, false_negative, threshold_value\
                     = SEISRCalcMethods.compute_ROC(times_window, values_window, forecast_interval, \
                    mag_large, min_mag, number_thresholds, 0, 0)
        
            SEISRPlotMethods.plot_precision_threshold(values_window, times_window, \
                    true_positive, false_positive, true_negative, false_negative, \
                    threshold_value, forecast_interval, mag_large, min_mag, plot_start_year,\
                    data_string_title, number_thresholds, NELng_local, SWLng_local, NELat_local, SWLat_local, \
                    Grid, Location, NSteps, delta_time, NSteps, lower_cutoff, min_rate)
                    
    #   .............................................................

    if Compute_Random_Statistics:

        for i in range(len(forecast_intervals)):
    
            forecast_interval = forecast_intervals[i]
        
            true_positive, false_positive, true_negative, false_negative, threshold_value\
                     = compute_ROC(times_window, values_window, forecast_interval, mag_large, number_thresholds, 0)
                    
            random_statistics(true_positive, false_positive, true_negative, false_negative, \
                    threshold_value, forecast_interval, mag_large, \
                    data_string_title, number_thresholds, NELng_local, SWLng_local, NELat_local, SWLat_local, \
                    Grid, Location)

    #   .............................................................
                    
    if Plot_Precision_Timeseries:
    
        for i in range(len(forecast_intervals)):
    
            forecast_interval = forecast_intervals[i]
            
            true_positive, false_positive, true_negative, false_negative, threshold_value\
                     = SEISRCalcMethods.compute_ROC(times_window, values_window, forecast_interval, \
                    mag_large, min_mag, number_thresholds, 0, 0)
        
            precision_list = \
                    SEISRCalcMethods.calc_precision_threshold(true_positive, false_positive, true_negative, false_negative)
                    
                    
            precision_timeseries = SEISRCalcMethods.compute_precision_timeseries\
                    (times_window, values_window, threshold_value, precision_list)
                    
                    
            SEISRPlotMethods.plot_precision_timeseries_prob\
                   (time_list_reduced, eqs_list_reduced, precision_timeseries, plot_start_year,\
                    NELng_local, SWLng_local, NELat_local, SWLat_local, Location, NSteps, delta_time, \
                    mag_large, mag_large_plot, min_mag, lower_cutoff, min_rate,\
                    forecast_interval, number_thresholds, \
                    true_positive, false_positive, true_negative, false_negative, threshold_value)
                    
    #   .............................................................
            
    if Optimize_Params:
    
#     
        forecast_interval = forecast_intervals[0]

        max_skill = -1000.0

        NSteps = 28
        
        for j in range(10):     #   number of NSteps values tested
    
            NSteps += 1
            
            lower_cutoff = 0.9
            
            for k in range(10):     #   Number of lower_cutoff values tested
            
                lower_cutoff += 0.1
    
                time_list_reduced, isr_times_reduced, eqs_list_reduced= SEISRCalcMethods.compute_isr_time_list(delta_time, lower_cutoff, \
                        NSteps, plot_start_year, mag_large, min_mag)
        
                skill_score_isr  = SEISRCalcMethods.calc_ROC_skill(time_list_reduced, isr_times_reduced, \
                        forecast_interval, mag_large, min_mag, number_thresholds)
                
                print(' sum eqlist_unfiltered', sum(eqs_list_unfiltered))
                print('sum time_list_unfiltered', sum(time_list_unfiltered))
                print('sum times_window', sum(times_window))
                print('NSteps, lower_cutoff, skill_score_isr', NSteps, round(lower_cutoff,2), round(skill_score_isr,3))
                
                if skill_score_isr > max_skill:
                    max_skill = skill_score_isr
                    NSteps_max = NSteps
                    Lower_cut_max = lower_cutoff
                print()
                print('So Far: max_skill, NSteps_max, Lower_cut_max: ', round(max_skill,3), NSteps_max, round(Lower_cut_max,2))
                print()
                    
        print('max_skill, NSteps_max, Lower_cut_max: ', round(max_skill,3), NSteps_max, round(Lower_cut_max,2))

    #   .............................................................
    
    if Plot_Mean_EQs_Time:
    
        forecast_interval = forecast_intervals[0]
    
        SEISRPlotMethods.plot_mean_eqs_timeseries(timeseries, time_bins, date_bins, plot_start_year, mag_large_plot, forecast_interval, \
                NELng_local, SWLng_local, NELat_local, SWLat_local, Location, NSteps, delta_time, min_mag, lower_cutoff, min_rate)
                
    #   .............................................................
    
    if Plot_Spatial_ROC:
    
    
        mag_array_large, date_array_large, time_array_large, year_array_large, depth_array_large, lat_array_large, lng_array_large = \
                SEISRFileMethods.read_regional_catalog(min_mag)
                
    #   ---------------------------------------------------
    #   Restricting the timeseries to only those values of interest 
    #
        ROC_range = len(time_list_reduced) -2
        
        date_bins_reduced       = date_bins[- ROC_range:]
        
#         print (date_bins)
#         
#         print()
#         
#         print(date_bins_reduced)
        
    #   ---------------------------------------------------
    #
        timeseries_EMA = []
        for i in range(len(timeseries)):
            EMA_working = []
            ROC_list = timeseries[i][-ROC_range:]
            mean_ROC_list    = np.mean(ROC_list)
            Rmin_local  = lower_cutoff * mean_ROC_list
            
            for j in range(len(timeseries[i])):
                if timeseries[i][j] < Rmin_local:
                    EMA_working.append(Rmin_local)
                else:
                    EMA_working.append(timeseries[i][j])
                    
            EMA_out_list = SEISRCalcMethods.timeseries_to_EMA(EMA_working, NSteps) # EMA average over timeseries in the ith grid box
        
            timeseries_EMA.append(EMA_out_list) #   These are the time
        
    #   ---------------------------------------------------
    #
        
        timeseries_EMA_reduced  = []
        for i in range(len(timeseries_EMA)):
            timeseries_EMA_reduced.append(timeseries_EMA[i][-ROC_range:])
            
    #   ---------------------------------------------------
    # 
        forecast_interval = forecast_intervals[0]
        
        plot_end_year = time_list_reduced[-1] - forecast_interval   #   The last years should not be used
        number_ROC_curves = int(plot_end_year - time_list_reduced[0]) + 1
        
        if number_ROC_curves > len(time_list_reduced)-1:
            number_ROC_curves = len(time_list_reduced)-1

        plot_end_index = (number_ROC_curves) * 13
            
            
    #   ---------------------------------------------------
    # 
        ROC_data        =   []
        ROC_data_print = [0. for i in range(len(timeseries))]
        
        tpr_list = []
        fpr_list = []

        for index_ROC in range(0, plot_end_index, 13):
#         for index_ROC in range(plot_start_index, plot_start_index + 1):
        
            print_text = 'Computing ROC File '+ str(index_ROC) + ' of ' + \
                        str(number_ROC_curves*13) + ' Frames on Date: ' + date_bins_reduced[index_ROC] 
            print(print_text)
            print()
        
            ROC_gridbox_events_list = \
                        SEISRCalcMethods.classify_large_earthquakes_grid_boxes\
                        (NELat_local, NELng_local, SWLat_local, SWLng_local, \
                        grid_size, index_ROC, time_list_reduced, forecast_interval,\
                        mag_array_large, year_array_large, depth_array_large, lat_array_large, lng_array_large)
#                        
            ROC_gridbox_threshold_list = SEISRCalcMethods.sort_list_EQ_RTI_order(\
                        ROC_gridbox_events_list,  NELat_local, NELng_local, SWLat_local, SWLng_local,\
                        min_mag, index_ROC, timeseries_EMA_reduced, NSTau, lower_cutoff)
                        
            true_positive, false_positive, true_negative, false_negative, threshold_value = \
                    SEISRCalcMethods.compute_spatial_ROC(ROC_gridbox_events_list, ROC_gridbox_threshold_list)
                
            true_positive_rate, false_positive_rate, false_negative_rate, true_negative_rate = \
                    SEISRCalcMethods.compute_spatial_ROC_rates(\
                    true_positive, false_positive, true_negative, false_negative)
# #         
            date_of_plot = date_bins[index_ROC]
            
            tpr_list.append(true_positive_rate)
            fpr_list.append(false_positive_rate)
            
            begin_date_of_plot = date_bins_reduced[0]
            end_date_of_plot   = date_bins_reduced[plot_end_index]
            
            SEISRPlotMethods.plot_spatial_ROC_diagram(tpr_list, fpr_list, NSTau, \
                    delta_time,lower_mag, min_rate, min_mag, grid_size, begin_date_of_plot, end_date_of_plot, forecast_interval,\
                    delta_deg_lat, delta_deg_lng, Location)
       
    #   .............................................................

    if Plot_Seismicity_Map:
    
        SEISRPlotMethods.map_seismicity(NELat, NELng, SWLat, SWLng, \
                NELat_local, NELng_local, SWLat_local, SWLng_local, plot_start_year, Location, catalog, mag_large_plot, mag_large, min_mag)
                
    #   .............................................................

    if Plot_Seismicity_Map:
    
        SEISRPlotMethods.map_seismicity(NELat, NELng, SWLat, SWLng, \
                NELat_local, NELng_local, SWLat_local, SWLng_local, plot_start_year, Location, catalog, mag_large_plot, mag_large, min_mag)
                
    #   .............................................................
    
    if Compute_Timeseries:
    
        timeseries, time_bins, date_bins = SEISRCalcMethods.coarse_grain_seismic_timeseries(\
                NELat_local, NELng_local, SWLat_local, SWLng_local, \
                min_mag, max_depth, grid_size, delta_time)
#     
    #   .............................................................
    
    if Plot_Total_RI:
    
        start_year = plot_start_year
        start_year = 1940.0
        end_year   = 2022
        
        time_stamp = 0.
        
#         timeseries, time_bins, date_bins = SEISRCalcMethods.coarse_grain_seismic_timeseries(\
#                 NELat_local, NELng_local, SWLat_local, SWLng_local, \
#                 min_mag, max_depth, grid_size, delta_time)

        SEISRPlotMethods.map_RI_contours(NELat_local, NELng_local, SWLat_local, SWLng_local, \
                    grid_size, start_year, end_year, time_stamp, catalog, min_mag, Location)
    
    #   .............................................................
    
    if Plot_Timeseries_PPV_Movie:
    
        forecast_interval = forecast_intervals[0]
        
        k = 0
        while float(time_bins[k]) < plot_start_year:
            plot_start_index = k
            k += 1
        

        time_list_reduced, log_number_reduced, eqs_list_reduced = \
                SEISRCalcMethods.compute_seisr_time_list(delta_time, lower_cutoff, \
                NSteps, plot_start_year, mag_large, min_mag)
        
        year_large_eq, mag_large_eq, index_large_eq = SEISRCalcMethods.get_large_earthquakes(mag_large_plot,min_mag)
            
        true_positive, false_positive, true_negative, false_negative, threshold_value = \
                    SEISRCalcMethods.compute_ROC(time_list_reduced, log_number_reduced, forecast_interval, \
                    mag_large, min_mag, number_thresholds, 0, 0)
                    
        true_positive_rate, false_positive_rate, false_negative_rate, true_negative_rate = \
                SEISRCalcMethods.compute_ROC_rates(true_positive, false_positive, true_negative, false_negative)   
                
        movie_range = len(time_list_reduced)
        
        date_bins_reduced = date_bins[- movie_range:]
        
        output_file_name = './DataMoviesNameFiles/Timeseries_PPV.txt'
#         output_file_name.truncate(0)
        
        output_file = open(output_file_name, 'w')

        for index_movie in range(movie_range):
        
            print_text = 'Computing Movie File '+ str(index_movie) + ' of ' + str(movie_range) + ' frames'
            print(print_text)
            print()
            
            figure_name = SEISRPlotMethods.plot_timeseries_precision_movie(time_list_reduced, log_number_reduced, plot_start_year, \
                    mag_large_plot, mag_large, min_mag,\
                    NELng_local, SWLng_local, NELat_local, SWLat_local, Location, NSteps, delta_time, lower_cutoff, min_rate,\
                    forecast_interval, number_thresholds, threshold_value, \
                    true_positive, false_positive, true_negative, false_negative, \
                    true_positive_rate, false_positive_rate, false_negative_rate, true_negative_rate,\
                    year_large_eq, mag_large_eq, index_large_eq,\
                    index_movie, date_bins_reduced)
                    
            print(figure_name, file=output_file)
            
        output_file.close()
                    
    #   .............................................................
    
    if Plot_Timeseries_ACC_Movie:
    
        forecast_interval = forecast_intervals[0]
        

        time_list_reduced, log_number_reduced, eqs_list_reduced = \
                SEISRCalcMethods.compute_seisr_time_list(delta_time, lower_cutoff, \
                NSteps, plot_start_year, mag_large, min_mag)
        
        year_large_eq, mag_large_eq, index_large_eq = SEISRCalcMethods.get_large_earthquakes(mag_large_plot,min_mag)
            
        true_positive, false_positive, true_negative, false_negative, threshold_value = \
                    SEISRCalcMethods.compute_ROC(time_list_reduced, log_number_reduced, forecast_interval, \
                    mag_large, min_mag, number_thresholds, 0, 0)
                    
        true_positive_rate, false_positive_rate, false_negative_rate, true_negative_rate = \
                SEISRCalcMethods.compute_ROC_rates(true_positive, false_positive, true_negative, false_negative)   
                
        movie_range = len(time_list_reduced)

        date_bins_reduced = date_bins[- movie_range:]
        
        output_file_name = './DataMoviesNameFiles/Timseries_ACC.txt'
#         output_file_name.truncate(0)
        
        output_file = open(output_file_name, 'w')
        
        for index_movie in range(movie_range):
        
            print_text = 'Computing Movie File '+ str(index_movie) + ' of ' + str(movie_range) + ' frames'
            print(print_text)
            print()
            
            figure_name = SEISRPlotMethods.plot_timeseries_accuracy_movie(time_list_reduced, log_number_reduced, plot_start_year, \
                    mag_large_plot, mag_large, min_mag,\
                    NELng_local, SWLng_local, NELat_local, SWLat_local, Location, NSteps, delta_time, lower_cutoff, min_rate,\
                    forecast_interval, number_thresholds, threshold_value, \
                    true_positive, false_positive, true_negative, false_negative, \
                    true_positive_rate, false_positive_rate, false_negative_rate, true_negative_rate,\
                    year_large_eq, mag_large_eq, index_large_eq,\
                    index_movie, date_bins_reduced)
                    
            print(figure_name, file=output_file)
            
        output_file.close()
                    
    #   .............................................................
    
    if Plot_RTI_Time_Slice_Movie:
    
#         SEISRFileMethods.get_regional_catalog(NELat_local, NELng_local, SWLat_local, SWLng_local, min_mag, max_depth,\
#             region_catalog_date_start, region_catalog_date_end)
    
#         timeseries, time_bins, date_bins = SEISRCalcMethods.coarse_grain_seismic_timeseries(\
#                 NELat_local, NELng_local, SWLat_local, SWLng_local, \
#                 min_mag, max_depth, grid_size, delta_time)

        time_list_reduced, log_number_reduced, eqs_list_reduced = \
                SEISRCalcMethods.compute_seisr_time_list(delta_time, lower_cutoff, \
                NSteps, plot_start_year, mag_large, min_mag)

                
        mag_array_large, date_array_large, time_array_large, year_array_large, depth_array_large, lat_array_large, lng_array_large = \
                SEISRFileMethods.read_regional_catalog(min_mag)
                
    #   ---------------------------------------------------
    #
    # 
        ROC_range = len(time_list_reduced) -2
        
        date_bins_reduced       = date_bins[- ROC_range:]
        
    #   ---------------------------------------------------
    #
        
        timeseries_EMA = []
        for i in range(len(timeseries)):
            EMA_working = []
            ROC_list = timeseries[i][-ROC_range:]
            mean_ROC_list    = np.mean(ROC_list)
            Rmin_local  = lower_cutoff * mean_ROC_list
            
            for j in range(len(timeseries[i])):
                if timeseries[i][j] < Rmin_local:
                    EMA_working.append(Rmin_local)
                else:
                    EMA_working.append(timeseries[i][j])
                    
            EMA_out_list = SEISRCalcMethods.timeseries_to_EMA(EMA_working, NSteps) # EMA average over timeseries in the ith grid box
        
            timeseries_EMA.append(EMA_out_list) #   These are the time
        
    #   ---------------------------------------------------

        forecast_interval = forecast_intervals[0]
        
        movie_range = len(time_list_reduced)
        
        date_bins_reduced       = date_bins[- movie_range:]
        
        timeseries_EMA_reduced  = []
        for i in range(len(timeseries_EMA)):
            timeseries_EMA_reduced.append(timeseries_EMA[i][-movie_range:])
        
#         print('len(timeseries_EMA, len(timeseries_EMA_reduced)',len(timeseries_EMA), len(timeseries_EMA_reduced)  )
#         print('len(timeseries_EMA[0], len(timeseries_EMA_reduced[0]', len(timeseries_EMA[0]), len(timeseries_EMA_reduced[0])  )
#         print()
#         print()
#         print()
# 
#         print()

        output_file_name = './DataMoviesNameFiles/RTI_Time_Slice.txt'
#         output_file_name.truncate(0)
        
        output_file = open(output_file_name, 'w')
 
        for index_movie in range(movie_range):

        
            print_text = 'Computing Movie File '+ str(index_movie) + ' of ' + str(movie_range) + ' frames'
            print(print_text)
            print()
            
            figure_name = SEISRPlotMethods.map_RTI_time_slice_contours(NELat_local, NELng_local, SWLat_local, SWLng_local, \
                        grid_size, min_mag, lower_mag, Location, index_movie, delta_time, catalog,\
                        timeseries_EMA_reduced, time_list_reduced, date_bins_reduced, NSTau, NSteps, \
                        forecast_interval, lower_cutoff, min_rate, \
                        mag_array_large, date_array_large, time_array_large, year_array_large,\
                        depth_array_large, lat_array_large, lng_array_large)
                        
            print(figure_name, file=output_file)
            
        output_file.close()
                        
        quit()  #   If it hasn't already terminated.
        
    #   .............................................................
    
    if Plot_Combine_Images:
    
    #   We use PIL to read the images and place two on the same page
    
    #   Use for plotting timeseris with spatial PDF
    
#         input_file1 = './DataMoviesNameFiles/DataPPV.txt'
#         input_file2 = './DataMoviesNameFiles/DataLogRTI.txt'
        
#         folder1 = './Data_for_Movies/'
#         folder2 = './DataMoviesLogRTI/'

    #   Use for plotting 2 spatial PDFs together
    
        input_file1 = './DataMoviesNameFiles/DataLogRTI.txt'
        input_file2 = './DataMoviesNameFiles/DataLogRTI.txt'
        
        folder1 = './Data_for_Movies/DataMoviesLogRTI_0.5Deg/'
        folder2 = './Data_for_Movies/DataMoviesLogRTI_1Deg/'
    
        SEISRCalcMethods.combine_images(input_file1, folder1, input_file2, folder2)
    

    #   .............................................................
    
