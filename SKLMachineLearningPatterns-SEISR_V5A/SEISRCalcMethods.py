    #   OTCalcMethods.py    -   This version will use the correlations to form time series,
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

from scipy.integrate import simps
from numpy import trapz

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

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
    
from PIL import Image

    ######################################################################
    
def read_input_data(input_file_name, min_mag):

    input_file = open(input_file_name,'r')
    
#     number_lines = 0
#     for line in input_file:
#         number_lines += 1
        
    values_window       =   []
    stddev_window       =   []
    times_window        =   []
    eqs_window          =   []
        
    number_lines = 0
    for line in input_file:
        items = line.strip().split(',')
#         print(items)
#         if '.' in items[0]:
#             print(items)
        number_lines += 1
        
        if number_lines > 2 and '.' in items[0]:
            items = line.strip().split(',')
#             print(items)
            values_window.append(float(items[3]))
            stddev_window.append(float(items[4]))
            times_window.append(float(items[1]))
            eqs_window.append(float(items[5]))
        
        if number_lines > 2 and 'Data:' in items[0]:
            items = line.strip().split(',')
            NELng_local = float(items[1])
            SWLng_local = float(items[2])
            NELat_local = float(items[3])
            SWLat_local = float(items[4])
            Grid        = float(items[5])
            Location = items[6]
            
    input_file.close()

    return values_window, stddev_window, times_window, eqs_window, NELng_local, SWLng_local, \
            NELat_local, SWLat_local, Grid, Location
            
    ######################################################################
    
def read_input_earthquake_data(delta_time, min_mag, lower_cutoff):

    #   Get catalog location and grid box data.  Then get list of small earthquakes.
    #
    #   Next, discretize the earthquake list into time intervals of length delta_time.  
    #
    #   This discretized earthquake list is then the number of small earthquakes in time intervals
    #       of length delta_time.  
    #
    #   This method returns the list of discrete times (time_list) and the number of small earthquakes in each
    #       discrete time interval (eqs_list)
    
    mag_array, date_array, time_array, year_array, depth_array, lat_array, lng_array = SEISRFileMethods.read_regional_catalog(min_mag)
    
    last_index  = int(len(year_array))-1
    first_index = 0
    
    year_diff = float(year_array[last_index]) - float(year_array[0])
    
    number_time_deltas = int (year_diff/delta_time)
    
    time_list   =   []
    eqs_list    =   []  
    
    jlast = number_time_deltas - 1
    last_time = jlast * delta_time
    number_eqs = 0
    initial_year = float(year_array[0])
    
    for i in range(len(time_array)):

        index = len(time_array) - i - 1
        
        j = int ( (float(year_array[index])- float(year_array[0]))  /delta_time)     #   j will be a monotone decreasing index
        
        if j == jlast:
            number_eqs += 1
        else:
            time_list.append((jlast-1) * delta_time + initial_year)
            
            if number_eqs == 0 and lower_cutoff > 0.0001:
#             if number_eqs == 0:
                number_eqs = 1
            eqs_list.append(number_eqs)
            number_eqs = 0
            jlast = j
        
    time_list.reverse()
    eqs_list.reverse()
    
    time_list = time_list[:-1]  #   Removing the extra last index that is introduced by the above code
    eqs_list  = eqs_list[:-1]

    return time_list,eqs_list
            
    ######################################################################
    
def get_large_earthquakes(mag_large, min_mag):

    mag_array, date_array, time_array, year_array, depth_array, lat_array, lng_array = \
            SEISRFileMethods.read_regional_catalog(min_mag)
            
    #   Find dates of large earthquakes
    
    mag_large_eq    =   []
    year_large_eq   =   []
    index_large_eq  =   []
        
    for i in range(len(year_array)):
        if float(mag_array[i]) >= mag_large:
            mag_large_eq.append(float(mag_array[i]))
            year_large_eq.append(float(year_array[i]))
            index_large_eq.append(i)

    return year_large_eq, mag_large_eq, index_large_eq
    
    ######################################################################
    
def adjust_year_times(year_large_eq, mag_large_eq, index_large_eq, time_list, plot_start_year):

    mag_large_eq_adj = mag_large_eq
    
    year_large_eq_adj   = []
    index_large_eq_adj  = []
    mag_large_eq_adj        = []
    
    for i in range(len(year_large_eq)):
    
        for j in range(len(time_list)-1):
        
            if year_large_eq[i] >= time_list[j] and year_large_eq[i] < time_list[j+1] and time_list[j] >= plot_start_year:
        
                year_large_eq_adj.append(time_list[j])
                index_large_eq_adj.append(j)
                mag_large_eq_adj.append(mag_large_eq[i])
                
    return year_large_eq_adj, mag_large_eq_adj, index_large_eq_adj 
    
def calc_eqs_unfiltered(time_list, eqs_list, plot_start_year):
        
#
#   ------------------------------------------------------------
#
    if plot_start_year <= time_list[0]:
        plot_start_year = time_list[0]
        
    number_points_to_plot = 0
    
    for k in range(len(time_list)):
        if time_list[k] >= plot_start_year:
            number_points_to_plot += 1
            
    
    time_list_unfiltered_reduced           = time_list[- number_points_to_plot:]
    eqs_list_unfiltered_reduced            = eqs_list[- number_points_to_plot:]
    
    return time_list_unfiltered_reduced, eqs_list_unfiltered_reduced
    
    ######################################################################

    
def calc_seisr_timeseries(time_list, eqs_list, plot_start_year, mag_large,min_mag, delta_time):
        
#
#   ------------------------------------------------------------
#
    year_large_eq, mag_large_eq, index_large_eq = get_large_earthquakes(mag_large,min_mag)
    
    if plot_start_year <= time_list[0]:
        plot_start_year = time_list[0]
        
    number_points_to_plot = 0
    
    for k in range(len(time_list)):
        if time_list[k] >= plot_start_year:
            number_points_to_plot += 1
            
    for i in range(len(time_list)):
        time_list[i] += 2.*delta_time        #    Adjust times to properly align the EMA times with the large EQ times
        
    last_index = len(time_list)-1
    time_list[last_index] += 2.*delta_time   #   adjustment to ensure correct last event time sequence
    
    log_number = [math.log(1.0+eqs_list[i],10) for i in range(len(eqs_list))]
    
    
    log_number_reduced          = log_number[- number_points_to_plot:]
    time_list_reduced           = time_list[- number_points_to_plot:]
    eqs_list_reduced            = eqs_list[- number_points_to_plot:]
   
#
#   ------------------------------------------------------------
#
    return time_list_reduced, log_number_reduced, eqs_list_reduced
    
    ######################################################################

def random_statistics(true_positive, false_positive, true_negative, false_negative, \
                    threshold_value, forecast_interval, mag_large, \
                    data_string_title, number_thresholds, NELng_local, SWLng_local, NELat_local, SWLat_local, \
                    Grid, Location):
# 
#   ------------------------------------------------------------
#
#   Plot ROC and random ROCs

    true_positive_rate, false_positive_rate, false_negative_rate, true_negative_rate = \
                compute_ROC_rates(true_positive, false_positive, true_negative, false_negative)   
                
    accuracy    =   []
    precision   =   []
    hit_rate    =   []
    
    for k in range(len(true_positive)):
        numer = true_positive[k] + true_negative[k]
        denom = true_positive[k] + false_positive[k] + true_negative [k] + false_negative[k]
        
        accuracy.append(numer/denom)
        hit_rate.append(true_positive[k]/(true_positive[k] + false_negative[k]))
        precision.append(true_positive[k]/(true_positive[k] + false_positive[k]))
        

    number_random_timeseries = 500
#     number_random_timeseries = 50

    random_true_positive_list       = []
    random_false_positive_list      = []
    random_false_negative_list      = []
    random_true_negative_list       = []
    
    random_accuracy_list    = []
    random_hit_rate_list    = []
    random_precision_list   = []
    random_specificity_list = []

    for i in range(number_random_timeseries):
    
        random_values = random_timeseries(values_window, times_window)
        
        true_positive_random, false_positive_random, true_negative_random, false_negative_random, threshold_value_random = \
                    compute_ROC(times_window, random_values, forecast_interval, mag_large, number_thresholds, i+1)
        
        denom = []
        for k in range(len(true_positive_random)):
            denom.append(true_positive_random[k] + false_positive_random[k] + true_negative_random[k] + false_negative_random[k])
            
        true_positive_random    = [ true_positive_random[m]/denom[m] for m in range(len(true_positive_random))]
        false_positive_random   = [false_positive_random[m]/denom[m] for m in range(len(false_positive_random))]
        false_negative_random   = [false_negative_random[m]/denom[m] for m in range(len(false_negative_random))]
        true_negative_random    = [ true_negative_random[m]/denom[m] for m in range(len(true_negative_random))]

        random_true_positive_list.append(true_positive_random)
        random_false_positive_list.append(false_positive_random)
        random_false_negative_list.append(false_negative_random)
        random_true_negative_list.append(true_negative_random)
        
        accuracy_random     =   []
        hit_rate_random     =   []
        precision_random    =   []
        specificity_random  =   []
        
        for k in range(len(true_positive_random)):  #======>  thi sshould be number of timeseries, not number of thresholds
            numer = true_positive_random[k] + true_negative_random[k]
            denom = true_positive_random[k] + false_positive_random[k] + true_negative_random[k] + false_negative_random[k]
            
            try:
                accuracy_random.append(numer/denom)
            except:
                pass
            
            try:
                hit_rate_random.append(true_positive_random[k]/(true_positive_random[k] + false_negative_random[k]))
            except:
                pass
            
            try:
                precision_random.append(true_positive_random[k]/(true_positive_random[k] + false_positive_random[k]))
            except:
                pass
                
            try:
                specificity_random.append(true_negative_random[k]/(true_negative_random[k] + false_positive_random[k]))
            except:
                pass
            
        random_accuracy_list.append(accuracy_random)
            
        random_hit_rate_list.append(hit_rate_random)
            
        random_precision_list.append(precision_random)
        
        random_specificity_list.append(specificity_random)
            
#     
#   ------------------------------------------------------------
#
    value_list   =   []
    for i in range(len(random_true_positive_list)):
        value_list.append(random_true_positive_list[i][optimal_index])
    mean_tp = np.mean(value_list)
    stddev_tp = np.std(value_list)
    
    value_list   =   []
    for i in range(len(random_false_positive_list)):
        value_list.append(random_false_positive_list[i][optimal_index])
    mean_fp = np.mean(value_list)
    stddev_fp = np.std(value_list)
        
    value_list   =   []
    for i in range(len(random_false_negative_list)):
        value_list.append(random_false_negative_list[i][optimal_index])
    mean_fn = np.mean(value_list)
    stddev_fn = np.std(value_list)

    value_list   =   []
    for i in range(len(random_true_negative_list)):
        value_list.append(random_true_negative_list[i][optimal_index])
    mean_tn = np.mean(value_list)
    stddev_tn = np.std(value_list)

    value_list   =   []
    for i in range(len(random_hit_rate_list)):
        value_list.append(random_hit_rate_list[i][optimal_index])
    mean_hr = np.mean(value_list)
    stddev_hr = np.std(value_list)
    
    value_list   =   []
    for i in range(len(random_specificity_list)):
        value_list.append(random_specificity_list[i][optimal_index])
    mean_spec = np.mean(value_list)
    stddev_spec = np.std(value_list)

    value_list   =   []
    for i in range(len(random_precision_list)):
        value_list.append(random_precision_list[i][optimal_index])
    mean_pre = np.mean(value_list)
    stddev_pre = np.std(value_list)

    value_list   =   []
    for i in range(len(random_accuracy_list)):
        value_list.append(random_accuracy_list[i][optimal_index])
    mean_acc = np.mean(value_list)
    stddev_acc = np.std(value_list)

# 
#   ------------------------------------------------------------
#

    print()
    print('--------------------------------------')
    print('')
    print('Forecast Interval: ', str(forecast_interval) + ' Years')
    print('')
    print('Threshold Value: ', str(round(float(threshold_value[optimal_index]),3)) )
    print('')
    print('Mean TP: ', str(round(mean_tp,3)) + ' +/- ' + str(round(stddev_tp,3)))
    print('')
    print('Mean FP: ', str(round(mean_fp,3)) + ' +/- ' + str(round(stddev_fp,3)))
    print('')
    print('Mean FN: ', str(round(mean_fn,3)) + ' +/- ' + str(round(stddev_fn,3)))
    print('')
    print('Mean TN: ', str(round(mean_tn,3)) + ' +/- ' + str(round(stddev_tn,3)))
    print('')
    print('Mean Random Hit Rate: ', str(round(mean_hr,3)) + ' +/- ' + str(round(stddev_hr,3)))
    print('')
    print('Mean Random Specificity: ', str(round(mean_spec,3)) + ' +/- ' + str(round(stddev_spec,3)))
    print('')
    print('Mean Random Precision: ', str(round(mean_pre,3)) + ' +/- ' + str(round(stddev_pre,3)))
    print('')
    print('Mean Random Accuracy: ', str(round(mean_acc,3)) + ' +/- ' + str(round(stddev_acc,3)))
    print('')
    print('--------------------------------------')
    print('')
        
    return

    ##############################################r########################
    
def timeseries_to_EMA(timeseries_orig, N_Steps):

    #   timeseries_orig is a list input.  Output is a list that is an Exponential Moving Average

    timeseries_EMA = []
        
    for i in range(1,len(timeseries_orig)+1):
        timeseries_raw = []
         
        for j in range(i):
            timeseries_raw.append(timeseries_orig[j])
             
        datapoint_EMA = EMA_weighted_time_series(timeseries_raw, N_Steps)
         
        timeseries_EMA.append(datapoint_EMA)
                  
    return timeseries_EMA
    
    ######################################################################
    
def EMA_weighted_time_series(time_series, NSteps):

    #   This method computes the Exponential Weighted Average of a list.  Last
    #       in the list elements are exponentially weighted the most

    N_events = len(time_series)
    
    weights = EMA_weights(N_events, NSteps)
    
    weights_reversed = list(reversed(weights))

    EMA_weighted_ts = []
    partial_weight_sum = 0.
    
    for i in range(N_events):
        partial_weight_sum += weights[i]
        weighted_ts = round(float(time_series[i])*weights_reversed[i],4)
        
        EMA_weighted_ts.append(weighted_ts)
        
    partial_weight_sum = round(partial_weight_sum,4)
    sum_value = sum(EMA_weighted_ts)
    
    if (float(partial_weight_sum)) <= 0.0:
        sum_value = 0.0001
        partial_weight_sum = 1.
    
    try:
        weighted_sum = float(sum_value)/float(partial_weight_sum)
    except:
        weighted_sum = 0.0
    
    return weighted_sum
    
    ######################################################################
    
def EMA_weights(N_events, N_Steps):

    #   This method computes the weights for the Exponential Weighted Average (EMA)

    alpha = 2./float((N_Steps+1))

    #   time_series_list is the time series of floating point values
    #       arranged in order of first element in list being earliest

    assert 0 < alpha <= 1
    
    weights = []
    
    #   Define the weights
    
    for i in range(0,N_events):
        weight_i = (1.0-alpha)**i
        weights.append(weight_i)
        
    sum_weights = sum(weights)
    weights =  [i/sum_weights for i in weights]
     
    return weights
    
    ######################################################################
    
def compute_ROC(times_window, values_window, forecast_interval, mag_large, min_mag, \
        number_thresholds, number_random_timeseries, time_number):

    #   First we find the min value, then progressively lower (actually raise) the threshold and determine the
    #       hit rate and false alarm rate
    
    true_positive_rate              =   []
    false_positive_rate             =   []
    true_negative_rate              =   []
    false_negative_rate             =   []
    
    true_positive               =   []
    false_positive              =   []
    true_negative               =   []
    false_negative              =   []
    
    acc_rate                        =   []
    threshold_value                 =   []
    
    year_large_eq, mag_large_eq, index_large_eq = get_large_earthquakes(mag_large, min_mag)
    
    values_window = [float(values_window[i]) for i in range(len(values_window))]
    
    min_value = min(values_window)
    max_value = max(values_window)
    delta_threshold = (max_value - min_value) / number_thresholds
    
#     if time_number == 0:
#         print('Calculating Data Time Series')
# #         print('')
#     else:    
#         print('Calculating Random Time Series: '+ str(time_number) + ' out of ' + str(number_random_timeseries), end="\r", flush=True)
        
    if time_number > 0:
        print('Calculating Random Time Series: '+ str(time_number) + ' out of ' + str(number_random_timeseries), end="\r", flush=True)
    
    threshold = min_value - delta_threshold
    
    print('')
    
    excluded_time = int(forecast_interval * 13)
    
    for i in range(number_thresholds):
        threshold = threshold + delta_threshold
        fp = 0.
        tp = 0.
        tn = 0.
        fn = 0.
        
        for j in range(len(times_window) - excluded_time):  #   We exclude the last time that has incomplete data
        
            test_flag = True
        
            for k in range(len(year_large_eq)):
            
                delta_time = year_large_eq[k] - times_window[j]

                #   if value greater than threshold and at least 1 eq occurs within forecast interval, tp
                if delta_time <= forecast_interval and delta_time >= 0 and float(values_window[j]) <= threshold and test_flag:
                    tp += 1.0
                    test_flag = False
                        
                #   if value greater than threshold, so predicted to occur within forecast interval,
                #       and eq does not occur within forecast interval, fp        
                if delta_time > forecast_interval and delta_time >= 0 and float(values_window[j]) <= threshold and test_flag:
                    fp += 1.0
                    test_flag = False

                #   if value less than threshold, so predicted not to occur within forecast interval, 
                #       and eq does occur within forecast interval, fn      
                if delta_time <= forecast_interval and delta_time >= 0 and float(values_window[j]) > threshold and test_flag:
                    fn += 1.0
                    test_flag = False

                #   if value less than threshold and eq does not occur within forecast interval, tn      
                if delta_time > forecast_interval and delta_time >= 0 and float(values_window[j]) > threshold and test_flag:
                    tn += 1.0
                    test_flag = False
                        
#         if (tp+fn)>0. and (fp+tn)>0.:
#             tpr = tp/(tp + fn)
#             fpr = fp/(fp + tn)
#         else:
#             tpr = 0.
#             fpr = 0.

        true_positive.append(tp)
        false_positive.append(fp)
        true_negative.append(tn)
        false_negative.append(fn)
        
        threshold_value.append(threshold)
        
#         print('Threshold Value: ', threshold, 'Hit Rate: ', tpr, 'False Alarm Rate: ', fpr, 'Ratio tpr/fpr: ', round(ratio,4))
        
        
#     true_positive_rate.append(0.)
#     false_positive_rate.append(0.)

#     print()
#     print(times_window)

    return  true_positive, false_positive, true_negative, false_negative, threshold_value
    
    ######################################################################
    
def compute_ROC_rates(true_positive, false_positive, true_negative, false_negative):

    true_positive_rate      =   []
    false_positive_rate     =   []
    true_negative_rate      =   []
    false_negative_rate     =   []

    for i in range(len(true_positive)):
    
        tp = true_positive[i]
        fp = false_positive[i]
        tn = true_negative[i]
        fn = false_negative[i]
    
        tpr = tp/(tp + fn)
        fpr = fp/(fp + tn)
        fnr = fn/(fn + tp)
        tnr = tn/(tn + fp)
        
        tpr = round(tpr,4)
        fpr = round(fpr,4)
        fnr = round(fnr,4)
        tnr = round(tnr,4)
        
        #   For each value of i
        true_positive_rate.append(tpr)
        false_positive_rate.append(fpr)
        true_negative_rate.append(tnr)
        false_negative_rate.append(fnr)

    return true_positive_rate, false_positive_rate, false_negative_rate, true_negative_rate
    
    ######################################################################
    
def random_timeseries(values_window, times_window):

    random_values = []
    
    for i in range(len(values_window)):
        random_values.append(random.choice(values_window))

    return random_values
    
    ######################################################################
    
def calc_precision_threshold(true_positive, false_positive, true_negative, false_negative):
# 
#   ------------------------------------------------------------
#
#   This method computes the precision list from the thresholds

    true_positive_rate, false_positive_rate, false_negative_rate, true_negative_rate = \
                compute_ROC_rates(true_positive, false_positive, true_negative, false_negative)   
                
    precision_list = []
    for i in range(len(true_positive)):
        numer = true_positive[i]
        denom = false_positive[i] + true_positive[i]
        
        try:
            precision_value = (numer/denom)
        except:
            precision_value = 0.
            
        precision_list.append(precision_value)
        
    return precision_list

    ######################################################################
    
def compute_precision_timeseries(times_window, values_window, threshold_value, precision_list):

    #   This method converts the isr timeseries into a precision timeseries
    
    precision_timeseries = []
    
    for i in range(len(times_window)):
    
        precision = 0.
    
        for j in range(len(threshold_value)):
        
            if values_window[i] >= threshold_value[j]:
                 precision = precision_list[j]
                 
        precision_timeseries.append(round(100.0*precision,3))
        
#     print('len of precision_timeseries', len(precision_timeseries))

    return precision_timeseries
    
    ######################################################################
    
def calc_ROC_skill(times_window, values_window, forecast_interval, mag_large, min_mag, number_thresholds):
# 
#   ------------------------------------------------------------
#
#   Calculate ROC and random ROCs

    true_positive, false_positive, true_negative, false_negative, threshold_value\
                     = compute_ROC(times_window, values_window, forecast_interval, mag_large, min_mag, number_thresholds, 0, 0)

    true_positive_rate, false_positive_rate, false_negative_rate, true_negative_rate = \
                compute_ROC_rates(true_positive, false_positive, true_negative, false_negative)   
# 
#   ------------------------------------------------------------
#        
 
    
    #   Redefine the hit rate and false alarm rate arrays
    number_intervals = 100
    
    fal_bins = list(range(0,number_intervals))
    fal_bins = [float(fal_bins[i])/float(number_intervals) for i in range(number_intervals)]
    delta_bins = 1./float(number_intervals)
    
    hit_bins        =   []
    
    for i in range(number_intervals):
        hit_value = 0.0
        counter   = 0
        for j in range(len(false_positive_rate)):
            if false_positive_rate[j] >= i*delta_bins and false_positive_rate[j] < (i+1)*delta_bins:  
                hit_value += true_positive_rate[j]
                counter += 1
        try:
            hit_bins.append(hit_value/float(counter))
        except:
            hit_bins.append(hit_value)
        
        fal_bins_array = np.array(fal_bins)
        hit_bins_array = np.array(hit_bins)
        
    # Compute the area using the composite trapezoidal rule.
    area_trapz = trapz(hit_bins_array, dx=delta_bins)
#     print("Trap Area =", area_trapz)

    # Compute the area using the composite Simpson's rule.
    area_simp = simps(hit_bins_array, dx=delta_bins)
    
#     print("Simp Area =", area_simp)
    
    skill_score_simp = round(area_simp,3)
    skill_score_trapz = round(area_trapz,3)
    
# 
#   ------------------------------------------------------------
#
#   Set the skill score to the average of simpson and trapz

    skill_score = 0.5*(skill_score_simp + skill_score_trapz)
        
    return skill_score

    ######################################################################
    
def compute_seisr_time_list(delta_time, lower_cutoff, NSteps, plot_start_year, mag_large, min_mag):


#                 eqs_list = eqs_list_unfiltered
#                 time_list = time_list_unfiltered

    time_list, eqs_list = read_input_earthquake_data(delta_time, min_mag, lower_cutoff)
    mean_eqs    = round(np.mean(eqs_list),3)
                
    eqs_list_unfiltered  = eqs_list
            
    for i in range(len(time_list)):
        if int(eqs_list[i]) <= lower_cutoff*mean_eqs:
            eqs_list[i] = lower_cutoff*mean_eqs
    
    #   Apply Exponential Moving Average to eqs_list
    eqs_list_EMA = timeseries_to_EMA(eqs_list, NSteps)
    
    #   Generate the SEISR times and filter the timeseries data to occur only after the plot_start_year
    
    time_list_reduced, log_number_reduced, eqs_list_reduced = \
            calc_seisr_timeseries(time_list, eqs_list_EMA, plot_start_year, mag_large,min_mag, delta_time)
                
    return time_list_reduced, log_number_reduced, eqs_list_reduced
    
    ######################################################################
    
def calc_forecast_hits_threshold(times_window, values_window, forecast_interval, year_large_eq, threshold_value, index_threshold):

    windows_number    =   0      #   Same number of elements as times window.  1 if qualifying window, 0 otherwise
    hits_number       =   0      #   Likewise.  For each element, 1 if eq occurs in the window, 0 otherwise
    
    freq_hits = 0
    
    for i in range(len(times_window)):      #   Hits for a given threshold

        if values_window[i] >= threshold_value[index_threshold]: #   Will be counted among qualifying time intervals
        
            windows_number += 1   #   Add to the number of qualifying time windows
            end_time = times_window[i] + forecast_interval
            hits_value = 0
            
            for j in range(len(year_large_eq)):
                if year_large_eq[j] >= times_window[i] and year_large_eq[j] < end_time:
                    hits_value = 1          #   Can only be counted once
                    
            hits_number += hits_value
    
    if windows_number > 0:
        freq_hits = float(hits_number)/float(windows_number)

    return freq_hits
    
    ######################################################################
    
def calc_observed_frequency(times_window, values_window, forecast_interval, year_large_eq, threshold_value):

    observed_stats_list = []

    for index_threshold in range(len(threshold_value)):
        
        freq_hits = calc_forecast_hits_threshold(times_window, values_window, \
                forecast_interval, year_large_eq, threshold_value, index_threshold)
        
        observed_stats_list.append(freq_hits)

    return observed_stats_list
    
    ######################################################################
    
def compute_raw_timeseries():

    time_bins, timeseries = SKLFileMethods.get_timeseries_data()

    return
    
    ######################################################################
    
    
def coarse_grain_seismic_timeseries(NELat_local, NELng_local, SWLat_local, SWLng_local, \
            min_mag, max_depth, grid_size, delta_time_interval):
    
    #   This method builds the local timeseries in small grid boxes. 

    #   We assume that the time interval for the seismicity time series will
    #       be weekly = 7 days = 0.01923 fraction of a year
    
    #   Read the regional catalog
    
#     mag_array_all, date_array_all, time_array_all, year_array_all, depth_array_all, lat_array_all, lng_array_all = \
#             SEISRFileMethods.read_regional_catalog(min_mag)

    mag_array, date_array, time_array, year_array, depth_array, lat_array, lng_array = \
            SEISRFileMethods.read_regional_catalog(min_mag)
            
    #   Use only events after plot_start_year
    
    
    num_lat_boxes = int( (NELat_local - SWLat_local)/grid_size )
    num_lng_boxes = int( (NELng_local - SWLng_local)/grid_size)
    
    num_total_boxes = num_lat_boxes * num_lng_boxes
    
    number_timeseries_found = 0
    total_counter = 0
    
    grid_box_locations  =   []
    grid_box_indices    =   []
    
    timeseries  =   []
    
#   ------------------------------------------------------------
#
#   The cutoff factor determines the minimum number of small earthquakes that are needed for each grid box
#
    total_time_interval = float(year_array[len(year_array)-1]) - float(year_array[0]) 
    
    last_event_year = float(year_array[len(year_array) - 1])
        
    number_year_bins = int((last_event_year - float(year_array[0]))/delta_time_interval) +1
    
    print('total_time_interval, last_event_year, number_year_bins', total_time_interval, last_event_year, number_year_bins)

#   ------------------------------------------------------------
    
    #   Define times of bins
    
    time_bins   =   []
    date_bins   =   []
    
    for i in range(number_year_bins):
        time_bins.append(float(year_array[0]) + float(i)*delta_time_interval)
    
    print('')
    print('Length of time_bins: ', len(time_bins))
    print('')
            
#   ------------------------------------------------------------
#   
    #   Define the grid boxes: Filter the regional data into (num_total_boxes) time series
    
    number_polygon_vertices = 4

    #   Construct the string of polygon vertices.  Note that the order is lat, long pairs
    
    for i in range(num_lat_boxes):
        for j in range(num_lng_boxes):
        
            ll = i+j
        
            vertex_lat = []
            vertex_lng = []
            
            mag_file   = []
            year_file  = []
            
    #   Order of vertices of large rectangular region:  SW, SE, NE, NW
    
            W_box_lng = SWLng_local + j*grid_size           #   West side of small box
            E_box_lng = SWLng_local + (j+1)*grid_size       #   East side of small box
            
#             N_box_lat = NELat_local - i*grid_size           #   North side of small box
#             S_box_lat = NELat_local - (i+1)*grid_size       #   South side of small box

            S_box_lat = SWLat_local + i*grid_size           #   North side of small box
            N_box_lat = SWLat_local + (i+1)*grid_size       #   South side of small box
            
            vertex_lat.append(S_box_lat)
            vertex_lat.append(S_box_lat)
            vertex_lat.append(N_box_lat)
            vertex_lat.append(N_box_lat)
    
            vertex_lng.append(W_box_lng)
            vertex_lng.append(E_box_lng)
            vertex_lng.append(E_box_lng)
            vertex_lng.append(W_box_lng)

            point_list = []
            
            for k in range(number_polygon_vertices):
                point_list.append((float(vertex_lat[k]),float(vertex_lng[k])))
    
            polygon = Polygon(point_list)
            
            index_timeseries = int(i + j*(num_lat_boxes))
        
#   ------------------------------------------------------------

    #   Compute the timeseries here and then timeseries[index_timeseries] = the timeseries you computed
    
            for kk in range(len(year_array)):

                point = Point((float(lat_array[kk]),float(lng_array[kk])))
        
                if (float(depth_array[kk]) <= float(max_depth) and float(mag_array[kk]) >= float(min_mag) \
                        and polygon.contains(point) == True):
                        
                    mag_file.append(float(mag_array[kk]))
                    year_file.append(float(year_array[kk]))
                
#   ------------------------------------------------------------

    #   Fill the working_file with the events over the time period.  Each week in working_file will
    #       record the number of events that occurred that week.
    
            last_event_year = float(year_array[len(year_array) - 1])

            working_file = [0.0 for i in range(int(number_year_bins))]
            
            for k in range(len(year_file)):
                index_working = int((float(year_file[k]) - float(year_array[0]))/delta_time_interval )
                working_file[index_working] += 1.0       #  This is a number timeseries
#
#           For the activity time series, number_years is the minimum number of active time bins required
#
            total_counter += 1
            
            lat_center = 0.5*(N_box_lat + S_box_lat)
            lng_center = 0.5*(W_box_lng + E_box_lng)
            grid_box_locations.append((lng_center,lat_center))
            grid_box_indices.append((j,i))

            timeseries.append(working_file)
            
            number_timeseries_found += 1
            
            print('')
            print('***************************************************')
            print('Found Timeseries Number ', number_timeseries_found, ' of ', num_total_boxes)
            print('Total number of events: ', sum(working_file))
            print('Grid Box Center @ Lat, Long: ', round(lat_center,3), round(lng_center,3))
            print('With indices @ Lat Index, Long Index: ', i,j)
            print('For minimum magnitude events >= ', min_mag)
            print('***************************************************')
            print('')

#     date_bins.append(date_array[0])

    for i in range(len(time_bins)):
        k = 0
        while float(year_array[k]) <= float(time_bins[i]) and k < len(year_array):
            date_value = date_array[k+1]
            k += 1
            
        date_bins.append(date_value)
                
    print('')
    print('Total Grid Boxes: ', num_total_boxes)
    print('')
    
    lat_print = []
    lng_print = []
    
    output_file = open('gridboxes.txt','w')
    for i in range(len(grid_box_locations)):
        lat_print = grid_box_locations[i][1]
        lng_print = grid_box_locations[i][0]
        lat_index = grid_box_indices[i][1]
        lng_index = grid_box_indices[i][0]
        
        print(round(float(lat_print),4), round(float(lng_print),4), lat_index, lng_index, file=output_file) 
    output_file.close()                                                     #      with space between
    
    output_file = open('timeseries.txt','w')
    print(' '.join(map(str,time_bins)), file=output_file)
    for i in range(len(timeseries)):
        timeseries_print = timeseries[i]                
        print(' '.join(map(str,timeseries_print)), file=output_file) #   Map converts list to string, joins elements
    output_file.close()                                                     #      with space between

#   Note: Refer to the elements of each timeseries as, e.g., timeseries[0][0] for the first list
#     return timeseries, grid_box_locations   

    return timeseries, time_bins, date_bins
    
    ######################################################################
    
def  define_EMA_timeseries(NSteps, min_mag):

    time_bins, timeseries = SEISRFileMethods.get_timeseries_data(min_mag)

    timeseries_N = []
    
    for i in range(len(timeseries)):
            working_list = timeseries[i]
            working_list_EMA = timeseries_to_EMA(working_list, NSteps)
            working_list_EMA = [round(working_list_EMA[j], 3) for j in range(len(working_list_EMA))]
            timeseries_N.append(working_list_EMA)
            
    output_file = open('timeseries_EMA' + '.txt','w')
    print(' '.join(map(str,time_bins)), file=output_file)
    for i in range(len(timeseries_N)):
        timeseries_print = timeseries_N[i]                
        print(' '.join(map(str,timeseries_print)), file=output_file) #   Map converts list to string, joins elements
    output_file.close()        

    return
    
    ######################################################################
    
def  define_EMA_timeseries_LS(LS, NSteps, min_mag):

    time_bins, timeseries = SEISRFileMethods.get_timeseries_data(min_mag)

    timeseries_N = []
    
    for i in range(len(timeseries)):
            working_list = timeseries[i]
            working_list_EMA = timeseries_to_EMA(working_list, NSteps)
            working_list_EMA = [round(working_list_EMA[j], 3) for j in range(len(working_list_EMA))]
            timeseries_N.append(working_list_EMA)
            
    output_file = open('timeseries_EMA_' + LS + '.txt','w')
    print(' '.join(map(str,time_bins)), file=output_file)
    for i in range(len(timeseries_N)):
        timeseries_print = timeseries_N[i]                
        print(' '.join(map(str,timeseries_print)), file=output_file) #   Map converts list to string, joins elements
    output_file.close()        

    return
    
    ######################################################################
    
def classify_large_earthquakes_grid_boxes(NELat_local, NELng_local, SWLat_local, SWLng_local, \
                        grid_size, index_ROC, time_list_reduced, forecast_interval,\
                        mag_array_large, year_array_large, depth_array_large, lat_array_large, lng_array_large):
                        
                        
    #   This code classifies the large earthquakes (M>4.95) between time_bins[i] and time_bins[i] + forecast_interval
    #       into the appropriate grid boxes.  Result is a list whose elements are the number of large earthquakes in 
    #       those grid boxes

    num_lat_boxes = int( (NELat_local - SWLat_local)/grid_size )
    num_lng_boxes = int( (NELng_local - SWLng_local)/grid_size)
    
    num_total_boxes = num_lat_boxes * num_lng_boxes
    
    ROC_event_list    =   [0 for i in range(num_total_boxes)]
#    
#   ------------------------------------------------------------
#   
    #   >>>>>>>>>  Define the grid boxes: Classify the large earthquakes into the appropriate grid boxes  <<<<<<<<<<
    
    number_polygon_vertices = 4

    #   Construct the string of polygon vertices.  Note that the order is lat, long pairs
    
    total_events = 0
    
    for i in range(num_lat_boxes):
        for j in range(num_lng_boxes):
        
#             ll = i+j    #   The grid box number
        
            vertex_lat = []
            vertex_lng = []
            
            
    #   Order of vertices of large rectangular region:  SW, SE, NE, NW
    
            W_box_lng = SWLng_local + j*grid_size           #   West side of small box
            E_box_lng = SWLng_local + (j+1)*grid_size       #   East side of small box
            
#             N_box_lat = NELat_local - i*grid_size           #   North side of small box
#             S_box_lat = NELat_local - (i+1)*grid_size       #   South side of small box

            S_box_lat = SWLat_local + i*grid_size           #   North side of small box
            N_box_lat = SWLat_local + (i+1)*grid_size       #   South side of small box
            
            vertex_lat.append(S_box_lat)
            vertex_lat.append(S_box_lat)
            vertex_lat.append(N_box_lat)
            vertex_lat.append(N_box_lat)
    
            vertex_lng.append(W_box_lng)
            vertex_lng.append(E_box_lng)
            vertex_lng.append(E_box_lng)
            vertex_lng.append(W_box_lng)

            point_list = []
            
            for k in range(number_polygon_vertices):
                point_list.append((float(vertex_lat[k]),float(vertex_lng[k])))
    
            polygon = Polygon(point_list)
            
            index_grid_box = int(i + j*(num_lat_boxes))
        
#   ------------------------------------------------------------

    #   Compute the timeseries here and then timeseries[index_timeseries] = the timeseries you computed
    
            for kk in range(len(year_array_large)):

                point = Point((float(lat_array_large[kk]),float(lng_array_large[kk])))
                
                current_time = float(time_list_reduced[index_ROC])
                later_time   = current_time + forecast_interval
        
                if (float(year_array_large[kk]) >= current_time and float(year_array_large[kk]) < later_time  \
                        and polygon.contains(point) == True):
                        
#                     print(mag_array_large[kk], year_array_large[kk], lat_array_large[kk], lng_array_large[kk])
                    
                    ROC_event_list[index_grid_box] += 1
                    
                    total_events += 1
                
 #    print('sum(ROC_list), total_events: ', sum(ROC_list), total_events)
#     print()
#     print(ROC_list)
#     print()
                
    return ROC_event_list   # Number of events by grid box
    
    ######################################################################
    
def sort_list_EQ_RTI_order(ROC_event_list,  NELat_local, NELng_local, SWLat_local, SWLng_local, min_mag, index_time, \
        timeseries_EMA, NSTau, lower_cutoff):
                        
    #   This method ingests the ROC_event_list and places that and the linked list into a list of descending order
    #       so that it can be used to plot the ROC curve
                        
#   ------------------------------------------------------------
#
#   Compute Relative Total Intensity list
    
    ROC_gridbox_threshold_list = []
    
    for i in range(len(timeseries_EMA)):    #   Number of gridboxes
    
#       Partial_ROC_list = timeseries_EMA[i][:index_time]

        ROC_list = timeseries_EMA[i][:]
        mean_ROC_list    = np.mean(ROC_list)
        
        ROC_test = timeseries_EMA[i][index_time]

        ROC_gridbox_threshold_list.append(ROC_test)
        
    sum_norm = 100.0/sum(ROC_gridbox_threshold_list)  #   Normalize all the spatial probability to 100%
#     
    ROC_gridbox_threshold_list = [ROC_gridbox_threshold_list[i]*sum_norm for i in range(len(ROC_gridbox_threshold_list))]
    
    ROC_gridbox_threshold_list = [math.log(1.0 + ROC_gridbox_threshold_list[i], 10) for i in range(len(ROC_gridbox_threshold_list))]
    
    ROC_gridbox_threshold_list = [round(ROC_gridbox_threshold_list[i],3) for i in range(len(ROC_gridbox_threshold_list))]
#  
#   ------------------------------------------------------------
#
    return ROC_gridbox_threshold_list
#  
    ######################################################################

def compute_spatial_ROC(ROC_gridbox_events_list, ROC_gridbox_threshold_list):
                    

    #   First we find the min value, then progressively lower the threshold and determine the
    #       hit rate and false alarm rate
    #
    #   Point here is to determine how many boxes above the threshold have at least 1 event in them = tp
    #       No events = fp.  Etc.
    
    true_positive               =   []
    false_positive              =   []
    true_negative               =   []
    false_negative              =   []
    threshold_value             =   []
    
    number_thresholds = 500
    number_grid_boxes = len(ROC_gridbox_threshold_list)
    
#     number_thresholds = 100
#     index_steps = int(len(ROC_data[0])/number_thresholds)
    
    min_value = 0
    max_value = max(ROC_gridbox_threshold_list)      # In %
    
    delta_threshold = (max_value) / number_thresholds
    
    #   Classify the grid boxes to compute the ROC
    
    
    
    
    for i in range(number_thresholds):
        current_threshold = max_value - float(1+i)*delta_threshold
        
#         print('current_threshold', delta_threshold, float(i)*delta_threshold, current_threshold)
#         print()
#         print(ROC_gridbox_events_list)
#         print(ROC_gridbox_threshold_list)
        
        tp = 0
        fp = 0
        fn = 0
        tn = 0
        
        for j in range(number_grid_boxes):
        
            
        
            # Current threshold less than grid box value, and SOME events occurred: tp
            if ROC_gridbox_threshold_list[j] >= current_threshold and ROC_gridbox_events_list[j] > 0:  
                tp += 1
            
            # Current threshold less than grid box value, and NO events occurred: fp
            if ROC_gridbox_threshold_list[j] >= current_threshold and ROC_gridbox_events_list[j] == 0:  
                fp += 1
                
            # Current threshold greater than grid box value, and SOME events occurred: fn
            if ROC_gridbox_threshold_list[j] < current_threshold and ROC_gridbox_events_list[j] > 0:  
                fn += 1
                
            # Current threshold greater than grid box value, and NO events occurred: tn
            if ROC_gridbox_threshold_list[j] < current_threshold and ROC_gridbox_events_list[j] == 0:  
                tn += 1
                
#         print()
#         print('Threshold: ', i, round(current_threshold,4))
#         print('tp, fp, fn, tn: ', tp, fp, fn, tn)
#         print()
        
        true_positive.append(tp)
        false_positive.append(fp)
        true_negative.append(tn)
        false_negative.append(fn)
        
        threshold_value.append(current_threshold)
        
        
        
    return true_positive, false_positive, true_negative, false_negative, threshold_value
    
    ######################################################################


def compute_spatial_ROC_rates(true_positive, false_positive, true_negative, false_negative):

    true_positive_rate      =   [0.]
    false_positive_rate     =   [0.]
    true_negative_rate      =   [1.]
    false_negative_rate     =   [1.]

    for i in range(len(true_positive)):
    
        tp = true_positive[i]
        fp = false_positive[i]
        tn = true_negative[i]
        fn = false_negative[i]
    
        tpr = 0.
        try:
            tpr = tp/(tp + fn)
        except:
            pass
            
        fpr = 0.
        try:
            fpr = fp/(fp + tn)
        except:
            pass
        
        fnr = 0.
        try:
            fnr = fn/(fn + tp)
        except:
            pass
            
        tnr = 0.
        try:
            tnr = tn/(tn + fp)
        except:
            pass
    
        tpr = round(tpr,4)
        fpr = round(fpr,4)
        fnr = round(fnr,4)
        tnr = round(tnr,4)
        
        #   For each value of i
        true_positive_rate.append(tpr)
        false_positive_rate.append(fpr)
        true_negative_rate.append(tnr)
        false_negative_rate.append(fnr)

    return true_positive_rate, false_positive_rate, false_negative_rate, true_negative_rate
    
    ####################################################################
    
def combine_images(input_file1, folder1, input_file2, folder2):


    # get images    
    
    input_list1 = []
    input_list2 = []
    
    input_file_images1 = open(input_file1, 'r')
    
    for line in input_file_images1:
        items1 = line.strip().split('/')
        input_list1.append(folder1 + items1[-1])
        
    input_file_images1.close()
    
    input_file_images2 = open(input_file2, 'r')
    
    for line in input_file_images2:
        items2 = line.strip().split('/')
        input_list2.append(folder2 + items2[-1])
        
    input_file_images2.close()
    
    for i in range(len(input_list1)):

    
        image1 = input_list1[i]
        image2 = input_list2[i]
    
        img1 = Image.open(image1)
        img2 = Image.open(image2)

    # get width and height
        w1, h1 = img1.size
        w2, h2 = img2.size
        
        # to calculate size of new image 
        w = max(w1, w2)
        h = max(h1, h2)
        
#         img1 = img1.resize((int(w*1.105),h2)) #   Use if timeseries is plotted with spatial pdf
        
        # create big empty image with place for images
        combined_image = Image.new('RGB', (w*2, h*1))

        # put images on new_image
        combined_image.paste(img1, (0, 0))
        combined_image.paste(img2, (w, 0))
#         combined_image.paste(img2, (int(w*0.9), 0)) #   To move images a bit closer together
        
        # save it
        combined_image.save('./DataMoviesCombined/PPV_LogRTI_combined_image_000' + str(i) + '.png')
        
    return
    
    ####################################################################

    