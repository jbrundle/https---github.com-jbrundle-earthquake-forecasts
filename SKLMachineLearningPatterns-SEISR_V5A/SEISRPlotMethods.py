#!/opt/local/bin python

    #   OTPlotMethods.py    -   This version will use the correlations to form time series,
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
from numpy import arange
from scipy.interpolate import UnivariateSpline

from numpy import asarray
from pandas import read_csv
from pandas import DataFrame
from pandas import concat

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import matplotlib.patches as mpatches
from matplotlib import gridspec

from matplotlib.offsetbox import AnchoredText
from matplotlib.image import imread
import matplotlib.ticker as mticker

from scipy.integrate import simps
from numpy import trapz

import itertools

import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature

import cartopy.io.img_tiles as cimgt
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from cartopy.feature import NaturalEarthFeature, LAND, COASTLINE



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

    ######################################################################
    ######################################################################
    
def plot_timeseries_prediction(values_window, stddev_window, times_window, eqs_window, \
        NELng_local, SWLng_local, NELat_local, SWLat_local, Grid, Location, n_feature, NN):
        
    plot_start_year = times_window[0]
    mag_large = 6.0
    
    year_large_eq, mag_large_eq, index_large_eq = get_large_earthquakes(mag_large)
    
    data = series_to_supervised(values_window, n_feature)
    
#     # evaluate, and predict the last NN data points

    time_data   = times_window[-NN:]
    stddev_data = stddev_window[-NN:]

    mae, y, yhat = walk_forward_validation(data, NN, time_data)  
    
#
#   ------------------------------------------------------------
#

    plt.plot(time_data, y, linestyle='-', lw=0.75, color='b', zorder=3, label='Expected (Mean Corr.)')
#     plt.plot(time_data, stddev_data, linestyle='-', lw=0.75, color='k', zorder=3, label='Expected (Std. Dev.)')
    plt.plot(time_data, y, 'b.', ms=4, zorder=3)
        
    plt.plot(time_data, yhat, linestyle='-', lw=0.75, color='r', zorder=3, label='Predicted')
    
    plt.gca().invert_yaxis()
    plt.legend()
    
    plt.yscale('linear')
    
    xmin,xmax = plt.xlim()
    ymin, ymax = plt.ylim()
    
    min_plot_line = [ymin for i in range(len(time_data))]
    plt.fill_between(time_data , min_plot_line, y, color='c', alpha=0.1, zorder=0)
    
    for i in range(len(year_large_eq)):
        x_eq = [year_large_eq[i], year_large_eq[i]]
        y_eq = [ymin,ymax]
#                         
        if float(mag_large_eq[i]) >= 6.0 and float(mag_large_eq[i]) < 6.8999  and float(year_large_eq[i]) >= time_data[0]:
            plt.plot(x_eq, y_eq, linestyle='dotted', color='k', lw=0.7, zorder=1)
            
        if float(mag_large_eq[i]) >= 6.8999 and float(year_large_eq[i]) >= time_data[0]:
            plt.plot(x_eq, y_eq, 'r', linestyle='--', lw=0.7, zorder=2)
    
    SupTitle_text = 'Predicting $\chi$(t) Timeseries 1 Step Ahead, ' + str(n_feature) + ' Features'
    plt.suptitle(SupTitle_text, fontsize=11, y = 0.96)
    
    delta_deg_lat = (NELat_local  - SWLat_local) * 0.5
    delta_deg_lng = (NELng_local  - SWLng_local) * 0.5
    
    Title_text = 'Within ' + str(delta_deg_lat) + '$^o$ Latitude and ' + str(delta_deg_lng) + '$^o$ Longitude of ' + Location
    plt.title(Title_text, fontsize=9)
    
    plt.ylabel('Weighted Correlation Value', fontsize = 11)
    plt.xlabel('Time (Year)', fontsize = 11)
    
    figure_name = 'Predictions/Expected-Predicted.png'
    plt.savefig(figure_name,dpi=600)
    plt.show()
        
    return

    ######################################################################
    
def plot_timeseries(values_window, stddev_window, times_window, eqs_window, plot_start_year, mag_large, data_string_title,\
        NELng_local, SWLng_local, NELat_local, SWLat_local, Grid, Location, NSteps):
#
#   ------------------------------------------------------------
#

    year_large_eq, mag_large_eq, index_large_eq = get_large_earthquakes(mag_large)
    
    if plot_start_year <= times_window[0]:
        plot_start_year <= times_window[0]
        
    last_element = len(times_window)-1
        
    delta_time_interval = times_window[last_element] - times_window[last_element - 1]
    
    number_points_to_plot = (times_window[last_element] - plot_start_year)/delta_time_interval
    number_points_to_plot = int(number_points_to_plot)
    
    time_list_reduced           = times_window[- number_points_to_plot:]
    correlation_list_reduced    = values_window[- number_points_to_plot:]
    stddev_list_reduced         = stddev_window[- number_points_to_plot:]
    
    plt.plot(time_list_reduced,correlation_list_reduced, linestyle='-', lw=1.0, color='b', zorder=3)

    xmin, xmax = plt.xlim()

#     x = [xmin, xmax]
#     y = [1.18,1.18]
#     plt.plot(x,y, linestyle='dashdot', lw=0.75, color='darkgreen', zorder=4)
    
    x = [xmin, xmax]
    y = [1.22,1.22]
    plt.plot(x,y, linestyle='dashdot', lw=0.75, color='darkgreen', zorder=4)
    
    plt.gca().invert_yaxis()
    
    ymin, ymax = plt.ylim()

    plt.grid(True, lw = 0.5, linestyle='dotted', zorder=0, axis = 'y')
    
    min_plot_line = [ymin for i in range(len(time_list_reduced))]
    plt.fill_between(time_list_reduced , min_plot_line, correlation_list_reduced, color='c', alpha=0.1, zorder=0)
    
#     plt.legend()
    
    for i in range(len(year_large_eq)):
        x_eq = [year_large_eq[i], year_large_eq[i]]
        y_eq = [ymin,ymax]
#                         
        if float(mag_large_eq[i]) >= 6.0 and float(mag_large_eq[i]) < 6.89999  and float(year_large_eq[i]) >= plot_start_year:
            plt.plot(x_eq, y_eq, linestyle='dotted', color='k', lw=0.7, zorder=1)
            
        if float(mag_large_eq[i]) >= 6.89999 and float(year_large_eq[i]) >= plot_start_year:
            plt.plot(x_eq, y_eq, 'r', linestyle='--', lw=0.7, zorder=2)
            
#     
#     plt.minorticks_on()
    
    delta_deg_lat = (NELat_local  - SWLat_local) * 0.5
    delta_deg_lng = (NELng_local  - SWLng_local) * 0.5
    
    SupTitle_text = 'Regional Seismicity Correlation $\chi$(t) Timeseries'

    plt.suptitle(SupTitle_text, fontsize=12, y = 0.96)
    
    Title_text = 'Within ' + str(delta_deg_lat) + '$^o$ Latitude and ' + str(delta_deg_lng) + '$^o$ Longitude of ' + Location
            
    plt.title(Title_text, fontsize=9)
    
    plt.ylabel('Weighted Correlation Value, $\chi$(t)', fontsize = 12)
    plt.xlabel('Time (Year)', fontsize = 12)
    
    data_string_title_reduced = data_string_title[15:]

    figure_name = './Data/Correlation_Time_' + data_string_title_reduced + '.png'
    plt.savefig(figure_name,dpi=600)
    
#     plt.show()
#     matplotlib.pyplot.close('all')
    plt.close('all')

    return 
    
    ######################################################################
    
def plot_seisr_timeseries(time_list_reduced, log_number_reduced, plot_start_year, mag_large,\
        NELng_local, SWLng_local, NELat_local, SWLat_local, Location, NSteps, delta_time_interval, min_mag, lower_cutoff, min_rate,\
        forecast_interval, number_thresholds, \
        true_positive, false_positive, true_negative, false_negative, threshold_value):
        
#     
#
#   ------------------------------------------------------------
#
    year_large_eq, mag_large_eq, index_large_eq = SEISRCalcMethods.get_large_earthquakes(mag_large,min_mag)
    
#     optimal_threshold = \
#             SEISRCalcMethods.compute_optimal_threshold(true_positive, false_positive, true_negative, false_negative,\
#             threshold_value)
#
#   ------------------------------------------------------------
#
        
    fig, ax = plt.subplots()
    
    ax.plot(time_list_reduced, log_number_reduced, linestyle='-', lw=1.0, color='b', zorder=3)

    xmin, xmax = plt.xlim()
    ymin, ymax = plt.ylim()
    
    year_large_eq, mag_large_eq, index_large_eq = \
            SEISRCalcMethods.adjust_year_times(year_large_eq, mag_large_eq, index_large_eq, time_list_reduced, plot_start_year)
            
   #   -------------------------------------------------------------

    for i in range(len(year_large_eq)):

        if float(mag_large_eq[i]) >= 6.0 and float(mag_large_eq[i]) < 6.89999  and float(year_large_eq[i]) >= plot_start_year:
            x_eq = [year_large_eq[i], year_large_eq[i]]
            y_eq = [ymax,log_number_reduced[index_large_eq[i]]]
            
            ax.plot(x_eq, y_eq, linestyle='dotted', color='k', lw=0.7, zorder=2)
            
    ax.plot(x_eq,y_eq, linestyle='dotted', color='k', lw=0.7, zorder=2, label = '6.9 $>$ M $\geq$ 6.0')
            
    for i in range(len(year_large_eq)):

        if float(mag_large_eq[i]) >= 6.89999 and float(year_large_eq[i]) >= plot_start_year:
            x_eq = [year_large_eq[i], year_large_eq[i]]
            y_eq = [ymax,log_number_reduced[index_large_eq[i]]]
            
            ax.plot(x_eq, y_eq, linestyle='--', color='r', lw=0.7, zorder=2)
            
    ax.plot(x_eq,y_eq, linestyle='--', color='r', lw=0.7, zorder=2, label='M $\geq$ 6.9')

    #   -------------------------------------------------------------
    
    max_plot_line = [ymax for i in range(len(time_list_reduced))]
    ax.fill_between(time_list_reduced , max_plot_line, log_number_reduced, color='c', alpha=0.1, zorder=0)
    
    plt.gca().invert_yaxis()
            
    ax.grid(True, lw = 0.5, which='major', linestyle='dotted', axis = 'both')
    
    ax.legend(loc = 'upper left', fontsize=6)
    
    #     
    #   ------------------------------------------------------------
    #
            
    test_time_interval = delta_time_interval/0.07692
    if abs(test_time_interval-1.0) <0.01:
        str_time_interval = '1 Month'
    elif abs(test_time_interval-0.25) < 0.01:
        str_time_interval = '1 Week'
    elif abs(test_time_interval-2.0) < 0.01:
        str_time_interval = '2 Months'
    elif abs(test_time_interval-3.0) < 0.01:
        str_time_interval = '3 Months'
        
    textstr =   'EMA Samples (N): ' + str(NSteps) +\
                '\nTime Step: ' + str_time_interval +\
                '\n$R_{min}$: ' + str(round(min_rate,0)) +\
                '\n$M_{min}$: ' + str(round(min_mag,2))

# 
#     # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='white', edgecolor = 'gray', alpha=0.75)

#     # place a text box in upper left in axes coords
    ax.text(0.015, 0.02, textstr, transform=ax.transAxes, fontsize=6,
        verticalalignment='bottom', horizontalalignment = 'left', bbox=props, linespacing = 1.8)

#   ------------------------------------------------------------
#

    ax.minorticks_on()
    
    delta_deg_lat = (NELat_local  - SWLat_local) * 0.5
    delta_deg_lng = (NELng_local  - SWLng_local) * 0.5
    
    SupTitle_text = 'Seismicity: ' + str(NSteps) + ' Month Exponential Moving Average'

    plt.suptitle(SupTitle_text, fontsize=12, y = 0.96)
    
    Title_text = 'Within ' + str(delta_deg_lat) + '$^o$ Latitude and ' + str(delta_deg_lng) + '$^o$ Longitude of ' + Location\
        + ' (Note Inverted Y-Axis)'
            
    plt.title(Title_text, fontsize=8)
    
    plt.ylabel('$Log_{10}$ (1 + Monthly Number)', fontsize = 12)
    plt.xlabel('Time (Year)', fontsize = 12)
    
    data_string_title = 'EMA' + '_FI' + str(forecast_interval) + '_TTI' + str(test_time_interval) + \
            '_NSTP' + str(NSteps) + '_MM' + str(min_mag) + '_CF' + str(lower_cutoff) 

    figure_name = './Data/SEISR_' + data_string_title + '_' + str(plot_start_year) + '.png'
    plt.savefig(figure_name,dpi=600)
    
#     plt.show()
#     matplotlib.pyplot.close('all')
    plt.close('all')

    return 
    
    ######################################################################
    
def plot_event_timeseries(time_list_reduced, eqs_list_reduced, eqs_list_EMA_reduced, plot_start_year, mag_large_plot,\
        NELng_local, SWLng_local, NELat_local, SWLat_local, Location, NSteps, delta_time_interval, min_mag, lower_cutoff):
#     
#
#   ------------------------------------------------------------
#
    year_large_eq, mag_large_eq, index_large_eq = SEISRCalcMethods.get_large_earthquakes(mag_large_plot,min_mag)
    
    
    fig, ax = plt.subplots()
    
    for j in range(len(eqs_list_reduced)):
    
        event_time = time_list_reduced[j] + 2.*delta_time_interval  #   adjustment to ensure correct event time registration
        x_eq = [event_time, event_time]
        y_eq = [0.,math.log(1.+eqs_list_reduced[j],10)]
#         y_eq = [0.,eqs_list_reduced[j]]
    
        ax.plot(x_eq,y_eq, linestyle='-', lw=0.5, color='c', zorder=5, alpha=0.30)
#         ax.plot(x_eq,y_eq, 'o', ms=2, color='b', zorder=5, alpha=0.70)
        
    ax.plot(x_eq,y_eq, linestyle='-', lw=0.5, color='c', zorder=5, alpha=0.30,label='Number M $\geq$ '+str(min_mag))

    time_list_adjusted  = [time_list_reduced[i] + 2.*delta_time_interval for i in range(len(time_list_reduced))]
    
    year_large_eq, mag_large_eq, index_large_eq = \
            SEISRCalcMethods.adjust_year_times(year_large_eq, mag_large_eq, index_large_eq, time_list_adjusted, plot_start_year)
            
    log_eqs_list        = [math.log(1.+eqs_list_reduced[i],10) for i in range(len(eqs_list_reduced))]
    log_eqs_list_EMA    = [math.log(1.+eqs_list_EMA_reduced[i],10) for i in range(len(eqs_list_EMA_reduced))]

    ax.plot(time_list_adjusted,log_eqs_list_EMA, '--', color='b', lw=1.0, zorder=5, label='EMA')

    xmin, xmax = plt.xlim()
    ymin, ymax = plt.ylim()
    
    #   -------------------------------------------------------------

    for i in range(len(year_large_eq)):

        if float(mag_large_eq[i]) >= 6.0 and float(mag_large_eq[i]) < 6.89999  and float(year_large_eq[i]) >= plot_start_year:
            x_eq = [year_large_eq[i], year_large_eq[i]]
            y_eq = [0,log_eqs_list_EMA[index_large_eq[i]]]
            
            ax.plot(x_eq, y_eq, linestyle='dotted', color='k', lw=0.7, zorder=2)
            
    ax.plot(x_eq,y_eq, linestyle='dotted', color='k', lw=0.7, zorder=2, label = '6.9 $>$ M $\geq$ 6.0')
            
    for i in range(len(year_large_eq)):

        if float(mag_large_eq[i]) >= 6.89999 and float(year_large_eq[i]) >= plot_start_year:
            x_eq = [year_large_eq[i], year_large_eq[i]]
            y_eq = [0,log_eqs_list_EMA[index_large_eq[i]]]
            
            ax.plot(x_eq, y_eq, linestyle='--', color='r', lw=0.7, zorder=2)
            
    ax.plot(x_eq,y_eq, linestyle='--', color='r', lw=0.7, zorder=2, label='M $\geq$ 6.9')

    #   -------------------------------------------------------------
    #
    #   If you want/don't want/ the bottom triangular markers, uncomment/comment the below
#     
    year_M6 = []
    time_M6 = []
    
    for i in range(len(year_large_eq)):
        x_eq = [year_large_eq[i], year_large_eq[i]]
        y_eq = [0,ymax]

        x_eq_mark = [year_large_eq[i]]
        y_eq_mark = [ymin]
                        
        if float(mag_large_eq[i]) >= 6.0 and float(mag_large_eq[i]) < 6.89999  and float(year_large_eq[i]) >= plot_start_year:
            year_M6.append(year_large_eq[i])
            
            ax.plot(x_eq, y_eq, linestyle='dotted', color='k', lw=0.7, zorder=2)

    year_M7 = []
            
    for i in range(len(year_large_eq)):
    
        last_index = len(year_large_eq)-1
        x_eq = [year_large_eq[i], year_large_eq[i]]
        y_eq = [0,ymax]
        
        x_eq_mark = [year_large_eq[i]]
        y_eq_mark = [ymin]
            
        if float(mag_large_eq[i]) >= 6.89999 and float(year_large_eq[i]) >= plot_start_year:
            year_M7.append(year_large_eq[i])
   
            ax.plot(x_eq, y_eq, 'r', linestyle='--', lw=0.7, zorder=2)

#             ax.plot(x_eq_mark, y_eq_mark, 'v', color='r', ms=5, zorder=2)    
            
    y_M6 = [ymin for i in range(len(year_M6))]
    y_M7 = [ymin for i in range(len(year_M7))]

    ax.plot(year_M6, y_M6, 'v', color='k', ms=2, zorder=3, label = '6.9 $>$ M $\geq$ 6.0')
    ax.plot(year_M7, y_M7, 'v', color='r', ms=5, zorder=2, label='M $\geq$ 6.9')
    
    #   The below masks off the line extensions above - sloppy way of erasing them!
    
    max_plot_line = [ymax for i in range(len(time_list_adjusted))]
    
    ax.fill_between(time_list_adjusted, log_eqs_list_EMA , max_plot_line, color='w', alpha=1, zorder=4)
    
    #   -------------------------------------------------------------
    #
    ax.grid(True, lw = 0.5, which='major', linestyle='dotted', axis = 'y')
    
    ax.legend(loc = 'upper right', fontsize=8)
    
    #     
    #   ------------------------------------------------------------
    #
    test_time_interval = delta_time_interval/0.07692
    if abs(test_time_interval-1.0) <0.01:
        str_time_interval = '1 Month'
    elif abs(test_time_interval-0.25) < 0.01:
        str_time_interval = '1 Week'
    elif abs(test_time_interval-2.0) < 0.01:
        str_time_interval = '2 Months'
    elif abs(test_time_interval-3.0) < 0.01:
        str_time_interval = '3 Months'

#     textstr =   'EMA Samples (N): ' + str(NSteps) +\
#                 '\nTime Step: ' + str_time_interval +\
#                 '\n$M_{min}$: ' + str(round(min_mag,2))

    textstr =   'Time Increment: ' + str_time_interval +\
                '\nN for EMA: ' + str(NSteps) +\
                '\n$M_{min}$: ' + str(round(min_mag,2))

# 
#     # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='white', edgecolor = 'gray', alpha=0.75)

#     # place a text box in upper left in axes coords
    ax.text(0.015, 0.980, textstr, transform=ax.transAxes, fontsize=8,
        verticalalignment='top', horizontalalignment = 'left', bbox=props, linespacing = 1.8, zorder=5)

#   ------------------------------------------------------------
#

    ax.minorticks_on()
    
    delta_deg_lat = (NELat_local  - SWLat_local) * 0.5
    delta_deg_lng = (NELng_local  - SWLng_local) * 0.5
    
    SupTitle_text = 'Seismicity Rate($t$) vs. Time'

    plt.suptitle(SupTitle_text, fontsize=12, y = 0.96)
    
    Title_text = 'Within ' + str(delta_deg_lat) + '$^o$ Latitude and ' + str(delta_deg_lng) + '$^o$ Longitude of ' + Location
            
    plt.title(Title_text, fontsize=9)
    
    plt.ylabel('$Log_{10}$(1 + $Number$)', fontsize = 10)
    plt.xlabel('Time (Year)', fontsize = 10)
    
    data_string_title = 'EMA' + '_TTI' + str(test_time_interval) + \
            '_NSTP' + str(NSteps) + '_MM' + str(min_mag) + '_CF' + str(lower_cutoff) 

    figure_name = './Data/Seismicity_' + data_string_title + '_' + str(plot_start_year) + '_' + str(plot_start_year) + '.png'
    plt.savefig(figure_name,dpi=600)
    
#     plt.show()
#     matplotlib.pyplot.close('all')
    plt.close('all')

    return 
    
    ######################################################################
    
def plot_precision_timeseries_prob\
        (time_list_reduced, eqs_list_reduced, isr_times_reduced, plot_start_year,\
        NELng_local, SWLng_local, NELat_local, SWLat_local, Location, NSteps, delta_time_interval, \
        mag_large, mag_large_plot, min_mag, lower_cutoff, min_rate,\
        forecast_interval, number_thresholds,\
        true_positive, false_positive, true_negative, false_negative, threshold_value):
        
#
#   ------------------------------------------------------------
#
    year_large_eq, mag_large_eq, index_large_eq = SEISRCalcMethods.get_large_earthquakes(mag_large_plot,min_mag)
    
#
#   ------------------------------------------------------------
#
        
    fig, ax = plt.subplots()
    
    ax.plot(time_list_reduced,isr_times_reduced, linestyle='-', lw=1.0, color='b', zorder=3)

#     xmin, xmax = ax.get_xlim()
#     ymin, ymax = ax.get_ylim()
#     
#     print(ymin, ymax)
#     
#     ax.set_ylim = (bottom=0, top=102)

    xmin, xmax = plt.xlim()
    ymin, ymax = plt.ylim()
    
#     ax.set_ylim(bottom=-2.0, top=102)
    
    
#     x_thresh = [xmin, xmax]
#     y_thresh = [optimal_threshold, optimal_threshold]
#     
#     precision_percent = round(critical_value*100.0,1)
#     label_text = '$D(T_W)$ for ' + str(precision_percent) + '% ' 
#     ax.plot(x_thresh,y_thresh, linestyle='-.', lw=0.75, color='g', zorder=4, label=label_text)
    
#     ax.legend(loc = 'upper right', fontsize=8)
    
    ax.grid(True, lw = 0.5, linestyle='dotted', zorder=0, axis = 'y')
    
    min_plot_line = [ymin for i in range(len(time_list_reduced))]
    ax.fill_between(time_list_reduced , min_plot_line, isr_times_reduced, color='c', alpha=0.1, zorder=0)
    
#     plt.legend()
    
    for i in range(len(year_large_eq)):
        x_eq = [year_large_eq[i], year_large_eq[i]]
        y_eq = [ymin,ymax]
#                         
        if float(mag_large_eq[i]) >= 6.0 and float(mag_large_eq[i]) < 6.89999  and float(year_large_eq[i]) >= plot_start_year:
            ax.plot(x_eq, y_eq, linestyle='dotted', color='k', lw=0.7, zorder=1)
            
        if float(mag_large_eq[i]) >= 6.89999 and float(year_large_eq[i]) >= plot_start_year:
            ax.plot(x_eq, y_eq, 'r', linestyle='--', lw=0.7, zorder=2)
            
#     
#   ------------------------------------------------------------
#
    test_time_interval = delta_time_interval/0.07692
    if abs(test_time_interval-1.0) <0.01:
        str_time_interval = '1 Month'
    elif abs(test_time_interval-0.25) < 0.01:
        str_time_interval = '1 Week'
    elif abs(test_time_interval-2.0) < 0.01:
        str_time_interval = '2 Months'
    elif abs(test_time_interval-3.0) < 0.01:
        str_time_interval = '3 Months'

    textstr =   'EMA Samples (N): ' + str(NSteps) +\
                '\n$M_{Large} \geq$' + str(mag_large) + \
                '\n$T_W$: ' + str(forecast_interval) + ' Years' +\
                '\nTime Step: ' + str_time_interval +\
                '\n$R_{min}$ = ' + str(lower_cutoff*min_mag) +\
                '\n$M_{Min}$: ' + str(round(min_mag,2))

# 
#     # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='white', edgecolor = 'gray', alpha=0.55)

#     # place a text box in upper left in axes coords
#     ax.text(0,1,textstr, transform=ax.transAxes, fontsize=8,
#         verticalalignment='top', horizontalalignment = 'left', bbox=props, linespacing = 1.8)

    ax.text(0.015, 0.715, textstr, transform=ax.transAxes, fontsize=8, bbox=props, linespacing = 1.8)

#   ------------------------------------------------------------
#

    ax.minorticks_on()
    
    delta_deg_lat = (NELat_local  - SWLat_local) * 0.5
    delta_deg_lng = (NELng_local  - SWLng_local) * 0.5
    
    SupTitle_text = 'Nowcast Precision(%) vs. Time for '

    plt.suptitle(SupTitle_text, fontsize=12, y = 0.96)
    
    Title_text = 'Within ' + str(delta_deg_lat) + '$^o$ Latitude and ' + str(delta_deg_lng) + '$^o$ Longitude of ' + Location
            
    plt.title(Title_text, fontsize=9)
    
    plt.ylabel('Precision (Of Current State, %)', fontsize = 12)
    plt.xlabel('Time (Year)', fontsize = 12)
    
    data_string_title = 'EMA' + '_FI' + str(forecast_interval) + '_TTI' + str(test_time_interval) + \
            '_NSTP' + str(NSteps) + '_MM' + str(min_mag) + '_CF' + str(lower_cutoff) 

    figure_name = './Data/Precision_Timeseries_' + data_string_title + '.png'
    plt.savefig(figure_name,dpi=600)
    
#     plt.show()
#     matplotlib.pyplot.close('all')
    plt.close('all')

    return 
    
    ######################################################################
    
def plot_temporal_ROC(values_window, times_window, true_positive, false_positive, true_negative, false_negative, \
                    threshold_value, forecast_interval, mag_large, min_mag, plot_start_year,\
                    data_string_title, number_thresholds, NELng_local, SWLng_local, NELat_local, SWLat_local, \
                    Grid, Location, NSteps, delta_time_interval, lower_cutoff, min_rate):
                    
# 
#   ------------------------------------------------------------
#
#   Plot ROC and random ROCs

    true_positive_rate, false_positive_rate, false_negative_rate, true_negative_rate = \
                SEISRCalcMethods.compute_ROC_rates(true_positive, false_positive, true_negative, false_negative)   

    number_random_timeseries = 500
    number_random_timeseries = 200
#     number_random_timeseries = 100
#     number_random_timeseries = 3
    
    opt_precision = 0.8
    
    fig, ax = plt.subplots()

    label_text = 'ROC for $M\geq$'+ str(mag_large) 

    ax.plot(false_positive_rate, true_positive_rate, linestyle='-', lw=1.0, color='r', zorder=3, label = label_text)
    
    ax.minorticks_on()
        
    x_line = [0.,1.]
    y_line = [0.,1.]
    
    ax.plot(x_line, y_line, linestyle='-', lw=1.0, color='k', zorder=2, label = 'Random Mean')
    
    random_true_positive_rate_list = [[] for i in range(number_thresholds)]
    
    for i in range(number_random_timeseries):
    
        random_values = SEISRCalcMethods.random_timeseries(values_window, times_window)
        
        true_positive_random, false_positive_random, true_negative_random, false_negative_random, threshold_value_random = \
                SEISRCalcMethods.compute_ROC(times_window, random_values, forecast_interval, mag_large, min_mag, \
                number_thresholds, number_random_timeseries, i+1)
                
        true_positive_rate_random, false_positive_rate_random, false_negative_rate_random, true_negative_rate_random = \
                SEISRCalcMethods.compute_ROC_rates(true_positive_random, false_positive_random, true_negative_random, false_negative_random)   
            
        for j in range(len(true_positive_rate_random)):
            random_true_positive_rate_list[j].append(true_positive_rate_random[j])
        
        ax.plot(false_positive_rate_random, true_positive_rate_random, linestyle='-', lw=2.0, color='cyan', zorder=1, alpha = 0.15)
# 
#   ------------------------------------------------------------
#        
    stddev_curve    =   []
    for i in range(len(random_true_positive_rate_list)):
        stddev_curve.append(np.std(random_true_positive_rate_list[i]))
        
    random_upper = []
    random_lower = []
    
    for i in range(number_thresholds):
        random_upper.append(false_positive_rate_random[i] + stddev_curve[i])
        random_lower.append(false_positive_rate_random[i] - stddev_curve[i])
        
    ax.plot(false_positive_rate_random, random_upper, linestyle='dotted', lw=0.75, color='k', zorder=2, label = '1 $\sigma$ Confidence')
    ax.plot(false_positive_rate_random, random_lower, linestyle='dotted', lw=0.75, color='k', zorder=2)
         
# 
#   ------------------------------------------------------------
#
    skill_score =   trapz(true_positive_rate, false_positive_rate)  #   Use the trapezoidal integration rule
    
    skill_score_upper    =   trapz(random_upper,false_positive_rate_random)
    skill_score_lower    =   trapz(random_lower,false_positive_rate_random)

    stddev_skill_score = 0.5*(abs(skill_score_upper- 0.5) + abs(skill_score_lower - 0.5))
    
    print()
    print('--------------------------------------')
    print()
    print('Skill Score: ', skill_score)
    print()
    print('Skill Score Random: ', '0.5 +/- ' + str(round(stddev_skill_score,3) ))
    print()
    print('--------------------------------------')
    print()
# 
    ax.legend(bbox_to_anchor=(0, 1), loc ='upper left', fontsize=8)
# 
#   ------------------------------------------------------------
#
#     skill_score = sum(hit_bins)/float(len(hit_bins))
    test_time_interval = delta_time_interval/0.07692
    if abs(test_time_interval-1.0) <0.01:
        str_time_interval = '1 Month'
    elif abs(test_time_interval-0.25) < 0.01:
        str_time_interval = '1 Week'
    elif abs(test_time_interval-2.0) < 0.01:
        str_time_interval = '2 Months'
    elif abs(test_time_interval-3.0) < 0.01:
        str_time_interval = '3 Months'

    textstr =       'Skill Score = ' + str(round(skill_score,3)) + \
                    '\n$T_W$ = ' + str(forecast_interval) + ' Years'+\
                    '\nEMA Samples (N): ' + str(NSteps) +\
                    '\nTime Step: ' + str_time_interval +\
                    '\n$R_{min}$: ' + str(round(min_rate,0)) +\
                    '\n$M_{min}$: ' + str(round(min_mag,2))


    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='white', edgecolor = 'gray', alpha=0.5)
# 
    # place a text box in bottom right in axes coords
    ax.text(0.975, 0.025, textstr, transform=ax.transAxes, fontsize=8,
        verticalalignment='bottom', horizontalalignment = 'right', bbox=props, linespacing = 1.8)
    
    SupTitle_text = 'Receiver Operating Characteristic'
    plt.suptitle(SupTitle_text, fontsize=12, y = 0.96)
    
    delta_deg_lat = (NELat_local  - SWLat_local) * 0.5
    delta_deg_lng = (NELng_local  - SWLng_local) * 0.5
    
    Title_text = 'Within ' + str(delta_deg_lat) + '$^o$ Latitude and ' + str(delta_deg_lng) + '$^o$ Longitude of ' + Location 
    plt.title(Title_text, fontsize=8)
    
    plt.ylabel('Hit Rate (TPR)', fontsize = 12)
    plt.xlabel('False Alarm Rate (FPR)', fontsize = 12)
    
    figure_name = 'Predictions/ROC_M' + str(mag_large) + '_FI' + str(forecast_interval) + '_' + str(plot_start_year) + '.png'
    
    plt.savefig(figure_name,dpi=300)
#     plt.show()
        
    return

    ##############################################r########################
 
def plot_precision_threshold(values_window, times_window, true_positive, false_positive, true_negative, false_negative, \
                    threshold_value, forecast_interval, mag_large, min_mag, plot_start_year,\
                    data_string_title, number_thresholds, NELng_local, SWLng_local, NELat_local, SWLat_local, \
                    Grid, Location, NStep, delta_time_interval, NSteps, lower_cutoff, min_rate):
# 
#   ------------------------------------------------------------
#
#   Plot ROC and random ROCs

    true_positive_rate, false_positive_rate, false_negative_rate, true_negative_rate = \
                SEISRCalcMethods.compute_ROC_rates(true_positive, false_positive, true_negative, false_negative)   
                
    threshold_reduced   =   []
    precision           =   []
    for i in range(1,len(true_positive)):
        numer = true_positive[i]
        denom = false_positive[i] + true_positive[i]
        threshold_reduced.append(threshold_value[i])
#         print(i, numer, denom)
        precision.append(numer/denom)
        
    number_random_timeseries = 200
#     number_random_timeseries = 5
    
    fig, ax = plt.subplots()

    label_text = 'Precision for $M\geq$'+ str(mag_large)

    precision = [precision[i]*100.0 for i in range(len(precision))]

    ax.plot(threshold_reduced, precision, linestyle='-', lw=1.0, color='r', zorder=3, label = label_text)
    
    xmin, xmax = plt.xlim()
    ymin, ymax = plt.ylim()
    
    plt.gca().invert_xaxis()
    
    ax.set_ylim(bottom=ymin-2, top=ymax+2)
    
    plt.grid(linestyle = 'dotted', linewidth=0.5)
    
    ax.minorticks_on()
        
    random_precision_list = [[] for i in range(number_thresholds)]
    
#   -------------------------------------------------

#   Plot all the random precision curves
    
    for i in range(number_random_timeseries):
    
        random_values = SEISRCalcMethods.random_timeseries(values_window, times_window)
        
        true_positive_random, false_positive_random, true_negative_random, false_negative_random, threshold_value_random = \
                SEISRCalcMethods.compute_ROC(times_window, random_values, forecast_interval, mag_large, min_mag,\
                number_thresholds, number_random_timeseries, i+1)
                
#         true_positive_rate_random, false_positive_rate_random, false_negative_rate_random, true_negative_rate_random = \
#                 compute_ROC_rates(true_positive_random, false_positive_random, true_negative_random, false_negative_random)   

        precision_random = []
        for k in range(1,len(true_positive_random)):
            numer = true_positive_random[k]
            denom = false_positive_random[k] + true_positive_random[k]
            trial_arg = 0.0
            try:
                trial_arg = numer/denom
            except:
                pass
            precision_random.append(trial_arg)
            
        precision_plot_random = precision_random
        for k in range(len(precision_random)):
            if precision_plot_random[k] > max(precision):
                precision_plot_random[k] = max(precision)
            
        for j in range(len(precision_random)):
            random_precision_list[j].append(precision_random[j])
            
        precision_plot_random= [precision_plot_random[i]*100.0 for i in range(len(precision_plot_random))]
        ax.plot(threshold_reduced, precision_plot_random, linestyle='-', lw=2.0, color='cyan', zorder=1, alpha = 0.15)
        
#        
#   ------------------------------------------------------------
#
    
    stddev_curve    =   []
    mean_curve      =   []
    
    for i in range(len(random_precision_list)):
        stddev_curve.append(np.std(random_precision_list[i]))
        mean_curve.append(np.mean(random_precision_list[i]))
        
#     mean_curve = [mean_curve[i] for i in range(len(mean_curve))]

    random_upper = []
    random_lower = []
    
    for i in range(number_thresholds):
        trial_plus = mean_curve[i] + stddev_curve[i]
        if trial_plus > max(precision):
            trial_plus = max(precision)
        random_upper.append(trial_plus)
        
        trial_minus = mean_curve[i] - stddev_curve[i]
        if trial_minus < 0.0:
            trial_minus = 0.0
        random_lower.append(trial_minus)
        
    mean_curve = [mean_curve[i]*100.0 for i in range(len(mean_curve))]
    ax.plot(threshold_value, mean_curve, linestyle='-', lw=0.75, color='k', zorder=2, label = 'Random Mean')
    
    random_upper = [random_upper[i]*100.0 for i in range(len(random_upper))]
    random_lower = [random_lower[i]*100.0 for i in range(len(random_lower))]

    ax.plot(threshold_value, random_upper, linestyle='dotted', lw=0.75, color='k', zorder=2, label = '1 $\sigma$ Confidence')
    ax.plot(threshold_value, random_lower, linestyle='dotted', lw=0.75, color='k', zorder=2)
    
    xmin, xmax = plt.xlim()
    ymin, ymax = plt.ylim()
    
    ax.set_ylim(bottom=ymin-5, top=ymax)
#          
# 
#   ------------------------------------------------------------
#
# 
    xmin,xmax = plt.xlim()
    ymin, ymax = plt.ylim()
    
#     ax.plot(x_max_thresh, y_max_thresh, linestyle='--', lw=0.75, color='b', zorder=4, label = 'Optimal $D_{\chi}(T_W)$')
 #    ax.plot(x_max_thresh, y_max_thresh, linestyle='-.', lw=1.0, color='g', zorder=4)
# 
#   ------------------------------------------------------------
#
    test_time_interval = delta_time_interval/0.07692
    if abs(test_time_interval-1.0) <0.01:
        str_time_interval = '1 Month'
    elif abs(test_time_interval-0.25) < 0.01:
        str_time_interval = '1 Week'
    elif abs(test_time_interval-2.0) < 0.01:
        str_time_interval = '2 Months'
    elif abs(test_time_interval-3.0) < 0.01:
        str_time_interval = '3 Months'

    textstr =  '$T_W$ = ' + str(forecast_interval) + ' Years'+\
                    '\nEMA Samples (N): ' + str(NSteps) +\
                    '\nTime Step: ' + str_time_interval +\
                    '\n$R_{min}$: ' + str(round(min_rate,0)) +\
                    '\n$M_{min}$: ' + str(round(min_mag,2))

    props = dict(boxstyle='round', facecolor='white', edgecolor = 'gray', alpha=0.5)
# 
    # place a text box in bottom right in axes coords
    ax.text(0.02, 0.5, textstr, transform=ax.transAxes, fontsize=8,
        verticalalignment='center', horizontalalignment = 'left', bbox=props, linespacing = 1.8)

    leg = ax.legend(loc = 'upper left', fontsize=8)
#     leg.set_title(title= legend_title_text,  prop={'size': 8})
    
# 
#   ------------------------------------------------------------
#

    SupTitle_text = 'Precision: Chance of a Correct Prediction'
    plt.suptitle(SupTitle_text, fontsize=12, y = 0.96)
    
    delta_deg_lat = (NELat_local  - SWLat_local) * 0.5
    delta_deg_lng = (NELng_local  - SWLng_local) * 0.5
    
    Title_text = 'Within ' + str(delta_deg_lat) + '$^o$ Latitude and ' + str(delta_deg_lng) + '$^o$ Longitude of ' + Location
    plt.title(Title_text, fontsize=9)
    
    plt.ylabel('Precision - PPV (%)', fontsize = 12)
    plt.xlabel('Decision Threshold $D(T_W)$ (Hours)', fontsize = 12)
    
    figure_name = 'Predictions/Precision_v_Threshold_M' + str(mag_large) + '_FI' + str(forecast_interval) + '_' + str(plot_start_year)+ '.png'
    plt.savefig(figure_name,dpi=300)
#     plt.show()
        
    return

    ######################################################################
    
def plot_timeseries_precision_movie(time_list_reduced, log_number_reduced, \
        plot_start_year, mag_large_plot, mag_large, min_mag,\
        NELng_local, SWLng_local, NELat_local, SWLat_local, Location, NSteps, delta_time_interval, lower_cutoff, min_rate,\
        forecast_interval, number_thresholds, threshold_value, \
        true_positive, false_positive, true_negative, false_negative, \
        true_positive_rate, false_positive_rate, false_negative_rate, true_negative_rate,\
        year_large_eq, mag_large_eq, index_large_eq,\
        index_movie, date_bins_reduced):
        
    #
    #   ------------------------------------------------------------
    #
    #  Set up the plots

    fig = plt.figure(figsize=(10, 6))        #   Define large figure and thermometer - 4 axes needed

    gs = gridspec.GridSpec(1,2,width_ratios=[15, 5], wspace = 0.2) 
    ax0 = plt.subplot(gs[0])
    
    #
    #   =======================================================================
    #
    #   First get the large earthquakes
    #
#     year_large_eq, mag_large_eq, index_large_eq = SEISRCalcMethods.get_large_earthquakes(mag_large_plot,min_mag)
#     print(mag_large_eq)
    #
    #   -------------------------------------------------------------
#     fig, ax = plt.subplots()
    
    ax0.plot(time_list_reduced,log_number_reduced, linestyle='-', lw=1.0, color='b', zorder=3)

    xmin, xmax = plt.xlim()
    ymin, ymax = plt.ylim()
    
    year_large_eq, mag_large_eq, index_large_eq = \
            SEISRCalcMethods.adjust_year_times(year_large_eq, mag_large_eq, index_large_eq, time_list_reduced, plot_start_year)
            
   #   -------------------------------------------------------------

    for i in range(len(year_large_eq)):

        if float(mag_large_eq[i]) >= 6.0 and float(mag_large_eq[i]) < 6.89999  and float(year_large_eq[i]) >= plot_start_year:
            x_eq = [year_large_eq[i], year_large_eq[i]]
            y_eq = [ymax,log_number_reduced[index_large_eq[i]]]
            
            ax0.plot(x_eq, y_eq, linestyle='dotted', color='k', lw=0.7, zorder=2)
            
    ax0.plot(x_eq,y_eq, linestyle='dotted', color='k', lw=0.7, zorder=2, label = '6.9 $>$ M $\geq$ 6.0')
            
    for i in range(len(year_large_eq)):

        if float(mag_large_eq[i]) >= 6.89999 and float(year_large_eq[i]) >= plot_start_year:
            x_eq = [year_large_eq[i], year_large_eq[i]]
            y_eq = [ymax,log_number_reduced[index_large_eq[i]]]
            
            ax0.plot(x_eq, y_eq, linestyle='--', color='r', lw=0.7, zorder=2)
            
    ax0.plot(x_eq,y_eq, linestyle='--', color='r', lw=0.7, zorder=2, label='M $\geq$ 6.9')

    #   -------------------------------------------------------------
    
    min_plot_line = [ymax for i in range(len(time_list_reduced))]
    ax0.fill_between(time_list_reduced , min_plot_line, log_number_reduced, color='c', alpha=0.1, zorder=0)
    
    plt.gca().invert_yaxis()
            
    ax0.grid(True, lw = 0.5, which='major', linestyle='dotted', axis = 'both')
    
    ax0.legend(loc = 'upper left', fontsize=8)
    
    #  
    #   ------------------------------------------------------------

    current_time     = time_list_reduced[index_movie]
    current_log_number      = log_number_reduced[index_movie]
    
#     print('current_log_number', current_log_number)
#     
#     print('current_time, current_log_number', round(current_time,4), round(current_log_number,2))

    ax0.plot(current_time,current_log_number, 'ro', ms=6, zorder=4)
    #     
    #   -------------------------------------------------------------
    #
    test_time_interval = delta_time_interval/0.07692
    if abs(test_time_interval-1.0) <0.01:
        str_time_interval = '1 Month'
    elif abs(test_time_interval-0.25) < 0.01:
        str_time_interval = '1 Week'
    elif abs(test_time_interval-2.0) < 0.01:
        str_time_interval = '2 Months'
    elif abs(test_time_interval-3.0) < 0.01:
        str_time_interval = '3 Months'

    textstr =   'EMA Samples (N): ' + str(NSteps) +\
                '\nTime Step: ' + str_time_interval +\
                '\n$R_{min}$: ' + str(round(min_rate,0))+\
                '\n$M_{min}$: ' + str(round(min_mag,2))
 
    #     # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='white', edgecolor = 'gray', alpha=0.75)

    #     # place a text box in upper left in axes coords
    ax0.text(0.015, 0.02, textstr, transform=ax0.transAxes, fontsize=8,
        verticalalignment='bottom', horizontalalignment = 'left', bbox=props, linespacing = 1.8)
        
    #     
    #   -------------------------------------------------------------
    #

#     textstr =   'Date: ' + str(round(time_list_reduced[index_seisr],3))
    textstr =   'Date: ' + str(date_bins_reduced[index_movie])
 
    #     # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='white', edgecolor = 'gray', alpha=0.75)

    #     # place a text box in upper left in axes coords
    ax0.text(0.5, 0.98, textstr, transform=ax0.transAxes, fontsize=10,
        verticalalignment='top', horizontalalignment = 'center', bbox=props, linespacing = 1.8)
        
    #     
    #   -------------------------------------------------------------
    #
    ax0.minorticks_on()
    
    delta_deg_lat = (NELat_local  - SWLat_local) * 0.5
    delta_deg_lng = (NELng_local  - SWLng_local) * 0.5
    
    SupTitle_text = 'Seismicity: ' + str(NSteps) + ' Month Exponential Moving Average'

    plt.suptitle(SupTitle_text, fontsize=12, y = 0.97)
    
    Title_text = 'Within ' + str(delta_deg_lat) + '$^o$ Latitude and ' + str(delta_deg_lng) + '$^o$ Longitude of ' + Location\
            + ' (Note Y-Axis is Inverted)'
            
    plt.title(Title_text, fontsize=9)
    
    plt.ylabel('$Log_{10}$ (1 + Monthly Number)', fontsize = 12)
    plt.xlabel('Time (Year)', fontsize = 8)
    
    #
    #   =======================================================================
    #
    #   Second plot: Precision    
    
    ax1 = plt.subplot(gs[1])
    frame1 = plt.gca()
    # 
    #   -------------------------------------------------------------
    #
    #   Plot ROC and random ROCs

#     true_positive_rate, false_positive_rate, false_negative_rate, true_negative_rate = \
#                 SEISRCalcMethods.compute_ROC_rates(true_positive, false_positive, true_negative, false_negative)   
                
    #   Might have to create a list with the threshold values in it
    
    threshold_reduced   =   []
    precision           =   []
    for i in range(1,len(true_positive)):
        numer = true_positive[i]
        denom = false_positive[i] + true_positive[i]
        threshold_reduced.append(threshold_value[i])
#         print(i, numer, denom)
        precision.append(numer/denom)
    
    index_PPV = (current_log_number - min(log_number_reduced)) /(max(log_number_reduced) - min(log_number_reduced))
    index_PPV = int(index_PPV * (len(true_positive)-1))
    
    if index_PPV > len(precision)-1:
        index_PPV = len(precision)-1
        
    current_PPV = 100*precision[index_PPV]
    print('current_PPV', round(current_PPV,2))
    
#     print('index_PPV, number_thresholds, len(true_positive)', index_PPV, number_thresholds, len(true_positive))
#     
#     current_PPV = precision[index_PPV]
        
    label_text = 'Precision for M'+ str(mag_large)

    precision = [precision[i]*100.0 for i in range(len(precision))]

    ax1.plot(precision, threshold_reduced, linestyle='-', lw=1.25, color='m', zorder=3, label = label_text)
    
    xmin,xmax = plt.xlim()
    ymin, ymax = plt.ylim()
    
    x_line = [current_PPV, current_PPV]
    y_line = [current_log_number, min(log_number_reduced)]

    
#     ax1.plot(x_line, y_line, linestyle='--', lw=1.0, color='r', zorder=4)
    
    x_line = [current_PPV, min(precision)]
    y_line = [current_log_number, current_log_number]
    
#     ax1.plot(x_line, y_line, linestyle='--', lw=1.0, color='r', zorder=4)
    
    ax1.plot(current_PPV, current_log_number, 'ro', ms=6, zorder=4)
    
    xmin, xmax = plt.xlim()
    ymin, ymax = plt.ylim()
    
#     ax1.set_ylim(bottom=ymin-2, top=ymax+2)

    plt.gca().invert_yaxis()
    
    plt.grid(linestyle = 'dotted', linewidth=0.5)
    
    ax1.minorticks_on()
    
    xmin,xmax = plt.xlim()
    ymin, ymax = plt.ylim()
 
    # 
    #   -------------------------------------------------------------
    #
    
    alphabox=0.5
    
    alert_color = 'w'
        
    if current_PPV >40:
        alert_color = 'deepskyblue'
        
    if current_PPV >50:
        alert_color = 'lime'
        
    if current_PPV >60:
        alert_color = 'yellow'
        
    if current_PPV >70:
        alert_color = 'orange'
        
    if current_PPV >80:
        alert_color = 'orangered'
        
    if current_PPV >90:
        alert_color = 'red'
        alphabox = 0.75

    textstr =   str(round(current_PPV,1)) + '%'
 
    #     # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor=alert_color, edgecolor = 'k', alpha=alphabox)

    #     # place a text box in upper left in axes coords
    ax1.text(0.05, 0.98, textstr, transform=ax1.transAxes, fontsize=12,
        verticalalignment='top', horizontalalignment = 'left', bbox=props, linespacing = 1.4)
            
    ax1.plot(current_PPV, ymin, 'rv', ms=10, zorder=4)
        
# 
#   -----------------------------------------------------------------
#


#     
    Title_text = 'Chance of an M$\geq$' + str(mag_large) + ' Earthquake ' + '\nWithin ' + str(round(forecast_interval,1)) + ' Years (PPV)'
    plt.title(Title_text, fontsize=9)
    
    plt.xlabel('Precision - PPV (%)', fontsize = 12)
    plt.ylabel('$Log_{10}$ (1 + Monthly Number)', fontsize = 12)
    
#     figure_name = 'Predictions/Timeseries_Precision_M' + str(mag_large) + '_FI' + str(forecast_interval) + '.png'
#     plt.savefig(figure_name,dpi=600)
    
    figure_name = './DataMoviesPPV/Timeseries_Precision_000' + str(index_movie) + '.png'
    plt.savefig(figure_name,dpi=150)
    
    plt.close()
    
    return 


    ######################################################################
    
def plot_timeseries_accuracy_movie(time_list_reduced, log_number_reduced, \
        plot_start_year, mag_large_plot, mag_large, min_mag,\
        NELng_local, SWLng_local, NELat_local, SWLat_local, Location, NSteps, delta_time_interval, lower_cutoff, min_rate,\
        forecast_interval, number_thresholds, threshold_value, \
        true_positive, false_positive, true_negative, false_negative, \
        true_positive_rate, false_positive_rate, false_negative_rate, true_negative_rate,\
        year_large_eq, mag_large_eq, index_large_eq,\
        index_movie, date_bins_reduced):
        
    #
    #   ------------------------------------------------------------
    #
    #  Set up the plots

    fig = plt.figure(figsize=(10, 6))        #   Define large figure and thermometer - 4 axes needed

    gs = gridspec.GridSpec(1,2,width_ratios=[15, 5], wspace = 0.2) 
    ax0 = plt.subplot(gs[0])
    
    #
    #   =======================================================================
    #
    #   First get the large earthquakes
    #
#     year_large_eq, mag_large_eq, index_large_eq = SEISRCalcMethods.get_large_earthquakes(mag_large_plot,min_mag)
#     print(mag_large_eq)
    #
    #   -------------------------------------------------------------
#     fig, ax = plt.subplots()
    
    ax0.plot(time_list_reduced,log_number_reduced, linestyle='-', lw=1.0, color='b', zorder=3)

    xmin, xmax = plt.xlim()
    ymin, ymax = plt.ylim()
    
    year_large_eq, mag_large_eq, index_large_eq = \
            SEISRCalcMethods.adjust_year_times(year_large_eq, mag_large_eq, index_large_eq, time_list_reduced, plot_start_year)
            
   #   -------------------------------------------------------------

    for i in range(len(year_large_eq)):

        if float(mag_large_eq[i]) >= 6.0 and float(mag_large_eq[i]) < 6.89999  and float(year_large_eq[i]) >= plot_start_year:
            x_eq = [year_large_eq[i], year_large_eq[i]]
            y_eq = [ymax,log_number_reduced[index_large_eq[i]]]
            
            ax0.plot(x_eq, y_eq, linestyle='dotted', color='k', lw=0.7, zorder=2)
            
    ax0.plot(x_eq,y_eq, linestyle='dotted', color='k', lw=0.7, zorder=2, label = '6.9 $>$ M $\geq$ 6.0')
            
    for i in range(len(year_large_eq)):

        if float(mag_large_eq[i]) >= 6.89999 and float(year_large_eq[i]) >= plot_start_year:
            x_eq = [year_large_eq[i], year_large_eq[i]]
            y_eq = [ymax,log_number_reduced[index_large_eq[i]]]
            
            ax0.plot(x_eq, y_eq, linestyle='--', color='r', lw=0.7, zorder=2)
            
    ax0.plot(x_eq,y_eq, linestyle='--', color='r', lw=0.7, zorder=2, label='M $\geq$ 6.9')

    #   -------------------------------------------------------------
    
    min_plot_line = [ymax for i in range(len(time_list_reduced))]
    ax0.fill_between(time_list_reduced , min_plot_line, log_number_reduced, color='c', alpha=0.1, zorder=0)
    
    plt.gca().invert_yaxis()
            
    ax0.grid(True, lw = 0.5, which='major', linestyle='dotted', axis = 'both')
    
    ax0.legend(loc = 'upper left', fontsize=8)
    
    #  
    #   ------------------------------------------------------------

    current_time     = time_list_reduced[index_movie]
    current_log_number      = log_number_reduced[index_movie]
    
#     print('current_log_number', current_log_number)
#     
#     print('current_time, current_log_number', round(current_time,4), round(current_log_number,2))

    ax0.plot(current_time,current_log_number, 'ro', ms=6, zorder=4)
    #     
    #   -------------------------------------------------------------
    #
    test_time_interval = delta_time_interval/0.07692
    if abs(test_time_interval-1.0) <0.01:
        str_time_interval = '1 Month'
    elif abs(test_time_interval-0.25) < 0.01:
        str_time_interval = '1 Week'
    elif abs(test_time_interval-2.0) < 0.01:
        str_time_interval = '2 Months'
    elif abs(test_time_interval-3.0) < 0.01:
        str_time_interval = '3 Months'

    textstr =   'EMA Samples (N): ' + str(NSteps) +\
                '\nTime Step: ' + str_time_interval +\
                '\n$R_{min}$: ' + str(round(min_rate,0))+\
                '\n$M_{min}$: ' + str(round(min_mag,2))
 
    #     # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='white', edgecolor = 'gray', alpha=0.75)

    #     # place a text box in upper left in axes coords
    ax0.text(0.015, 0.02, textstr, transform=ax0.transAxes, fontsize=8,
        verticalalignment='bottom', horizontalalignment = 'left', bbox=props, linespacing = 1.8)
        
    #     
    #   -------------------------------------------------------------
    #

#     textstr =   'Date: ' + str(round(time_list_reduced[index_seisr],3))
    textstr =   'Date: ' + str(date_bins_reduced[index_movie])
 
 
    #     # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='white', edgecolor = 'gray', alpha=0.75)

    #     # place a text box in upper left in axes coords
    ax0.text(0.5, 0.98, textstr, transform=ax0.transAxes, fontsize=10,
        verticalalignment='top', horizontalalignment = 'center', bbox=props, linespacing = 1.8)
        
    #     
    #   -------------------------------------------------------------
    #
    ax0.minorticks_on()
    
    delta_deg_lat = (NELat_local  - SWLat_local) * 0.5
    delta_deg_lng = (NELng_local  - SWLng_local) * 0.5
    
    SupTitle_text = 'Seismicity: ' + str(NSteps) + ' Month Exponential Moving Average'

    plt.suptitle(SupTitle_text, fontsize=12, y = 0.97)
    
    Title_text = 'Within ' + str(delta_deg_lat) + '$^o$ Latitude and ' + str(delta_deg_lng) + '$^o$ Longitude of ' + Location\
            + ' (Note Y-Axis is Inverted)'
            
    plt.title(Title_text, fontsize=9)
    
    plt.ylabel('$Log_{10}$ (1 + Monthly Number)', fontsize = 12)
    plt.xlabel('Time (Year)', fontsize = 8)
    
    #
    #   =======================================================================
    #
    #   Second plot: Precision    
    
    ax1 = plt.subplot(gs[1])
    frame1 = plt.gca()
    # 
    #   -------------------------------------------------------------
    #
    #   Plot ROC and random ROCs

    #   Might have to create a list with the threshold values in it
    
    accuracy = []
    for i in range(len(true_positive)):
        numer = true_positive[i] + true_negative[i]
        denom = true_positive[i] + true_negative[i] + false_positive[i] + false_negative[i]
        accuracy.append(numer/denom)
        
    index_ACC = (current_log_number - min(log_number_reduced)) /(max(log_number_reduced) - min(log_number_reduced))
    index_ACC = int(index_ACC * (len(true_positive)-1))
    
    current_ACC = 100*accuracy[index_ACC]
    print('current_ACC', round(current_ACC,2))
    
#     print('index_PPV, number_thresholds, len(true_positive)', index_PPV, number_thresholds, len(true_positive))
#     
#     current_PPV = precision[index_PPV]
        
    label_text = 'Accuracy for M'+ str(mag_large)

    accuracy = [accuracy[i]*100.0 for i in range(len(accuracy))]

    ax1.plot(accuracy, threshold_value, linestyle='-', lw=1.25, color='m', zorder=3, label = label_text)
    
    xmin,xmax = plt.xlim()
    ymin, ymax = plt.ylim()
    
    x_line = [current_ACC, current_ACC]
    y_line = [current_log_number, min(log_number_reduced)]

    
#     ax1.plot(x_line, y_line, linestyle='--', lw=1.0, color='r', zorder=4)
    
    x_line = [current_ACC, min(accuracy)]
    y_line = [current_log_number, current_log_number]
    
#     ax1.plot(x_line, y_line, linestyle='--', lw=1.0, color='r', zorder=4)
    
    ax1.plot(current_ACC, current_log_number, 'ro', ms=6, zorder=4)
    
    xmin, xmax = plt.xlim()
    ymin, ymax = plt.ylim()
    
#     ax1.set_ylim(bottom=ymin-2, top=ymax+2)

    plt.gca().invert_yaxis()
    
    plt.grid(linestyle = 'dotted', linewidth=0.5)
    
    ax1.minorticks_on()
    
    xmin,xmax = plt.xlim()
    ymin, ymax = plt.ylim()
 
    # 
    #   -------------------------------------------------------------
    #
    
    alphabox=0.5
    
    alert_color = 'w'
        
    if current_ACC >40:
        alert_color = 'deepskyblue'
        
    if current_ACC >50:
        alert_color = 'lime'
        
    if current_ACC >60:
        alert_color = 'yellow'
        
    if current_ACC >70:
        alert_color = 'orange'
        
    if current_ACC >80:
        alert_color = 'orangered'
        
    if current_ACC >90:
        alert_color = 'red'
        alphabox = 0.75

    textstr =   str(round(current_ACC,1)) + '%'
 
    #     # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor=alert_color, edgecolor = 'k', alpha=alphabox)

    #     # place a text box in upper left in axes coords
    ax1.text(0.05, 0.98, textstr, transform=ax1.transAxes, fontsize=12,
        verticalalignment='top', horizontalalignment = 'left', bbox=props, linespacing = 1.4)
            
    ax1.plot(current_ACC, ymin, 'rv', ms=10, zorder=4)
        
# 
#   -----------------------------------------------------------------
#

#     
    Title_text = 'Accuracy for M$\geq$' + str(mag_large) + ' Earthquake ' + '\nWithin ' + str(round(forecast_interval,1)) + ' Years (ACC)'
    plt.title(Title_text, fontsize=9)
    
    plt.xlabel('Accuracy - ACC (%)', fontsize = 12)
    plt.ylabel('$Log_{10}$ (1 + Monthly Number)', fontsize = 12)
    
#     figure_name = 'Predictions/Timeseries_Precision_M' + str(mag_large) + '_FI' + str(forecast_interval) + '.png'
#     plt.savefig(figure_name,dpi=600)
    
    figure_name = './DataMoviesACC/Timeseries_Accuracy_000' + str(index_movie) + '.png'
    plt.savefig(figure_name,dpi=150)
    
    plt.close()
    
    return 


    ######################################################################
    
def map_seismicity(NELat, NELng, SWLat, SWLng, \
        NELat_local, NELng_local, SWLat_local, SWLng_local, plot_start_year, Location, catalog, mag_large_plot, mag_large, min_mag):

    #   Note:  eigen_number is the number of the eigenvector as defined in descending
    #       order by eigenvalue
    
    #   Note:  This uses the new Cartopy interface
    #
    #   -----------------------------------------
    #
    #   Define plot map
    
    dateline_crossing = False
    
    #   California
    left_long   = SWLng_local
    right_long  = NELng_local
    top_lat     = SWLat_local
    bottom_lat  = NELat_local
    
    delta_deg_lat = (NELat_local  - SWLat_local) * 0.5
    delta_deg_lng = (NELng_local  - SWLng_local) * 0.5
    
    longitude_labels = [left_long, right_long]
    longitude_labels_dateline = [left_long, 180, right_long, 360]   #   If the map crosses the dateline
    
    central_long_value = 0
    if dateline_crossing:
        central_long_value = 180
        
    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=central_long_value))
    ax.set_extent([left_long, right_long, bottom_lat, top_lat])

    # Create a feature for States/Admin 1 regions at 1:50m from Natural Earth
    states_provinces = cfeature.NaturalEarthFeature(
        category='cultural',
        name='admin_1_states_provinces_lines',
        scale='50m',
        facecolor='none')

    land_10m = cfeature.NaturalEarthFeature('physical', 'land', '10m',
                                        edgecolor='face',
                                        facecolor='coral')
                                        

                                        
    ocean_10m_3000 = cfeature.NaturalEarthFeature('physical', 'bathymetry_H_3000', '10m',
#                                         edgecolor='black',
                                        facecolor='#0000FF',
                                        alpha = 0.3)
                                        

                                        
    lakes_10m = cfeature.NaturalEarthFeature('physical', 'lakes', '10m',
#                                        edgecolor='black',
                                        facecolor='blue',
                                        alpha = 0.75)
                                        
    rivers_and_lakes = cfeature.NaturalEarthFeature('physical', 'rivers_lakes_centerlines', '10m',
#                                        edgecolor='aqua',
                                        facecolor='blue',
                                        alpha = 0.75)

    ax.add_feature(ocean_10m_3000)

    ax.add_feature(cfeature.BORDERS, linestyle='-', alpha=.5, linewidth=0.5)
    ax.add_feature(cfeature.LAKES, alpha=0.95)
    ax.add_feature(cfeature.RIVERS, linewidth= 0.5)
    ax.add_feature(cfeature.STATES, edgecolor='gray',linewidth= 0.5)
#     ax.add_feature(states_provinces, edgecolor='gray')
    ax.coastlines(resolution='10m', color='black', linewidth=0.5)
    
#     stamen_terrain = cimgt.StamenTerrain()
    stamen_terrain = cimgt.Stamen('terrain-background')
    #   Zoom level should not be set to higher than about 6
    ax.add_image(stamen_terrain, 6)

    if dateline_crossing == False:
        gl = ax.gridlines(crs = ccrs.PlateCarree(), draw_labels=True,
                   linewidth=0.25, color='black', alpha=0.5, linestyle='dotted')
                   
    if dateline_crossing == True:
        gl = ax.gridlines(xlocs=longitude_labels_dateline, draw_labels=True,
                   linewidth=1.0, color='white', alpha=0.5, linestyle='--')

    gl.top_labels = False
    gl.right_labels = False
    gl.xlines = True
    gl.ylines = True

    if catalog == 'LosAngeles':
        gl.xlocator = mticker.FixedLocator([-112,-114,-116, -118, -120, -122])
    
    if catalog == 'Tokyo':
        gl.xlocator = mticker.FixedLocator([132,134,136, 138, 140, 142, 144, 146])
#    gl.xlocator = mticker.FixedLocator([left_side, right_side])

#    gl.xlocator = mticker.FixedLocator([left_side, right_side])

    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    
    #   -----------------------------------------
    #   Put california faults on the map
    
    input_file_name = './California_Faults.txt'
    input_file  =   open(input_file_name, 'r')
    
    for line in input_file:
        items = line.strip().split()
        number_points = int(len(items)/2)
        
        for i in range(number_points-1):
            x = [float(items[2*i]),float(items[2*i+2])]
            y = [float(items[2*i+1]), float(items[2*i+3])]
            ax.plot(x,y,'-', color='darkgreen',lw=0.55, zorder=2)
    
    input_file.close()
    #    
    #   -----------------------------------------
    #
    #   Plot the data
    
    mag_array, date_array, time_array, year_array, depth_array, lat_array, lng_array = \
            SEISRFileMethods.read_regional_catalog(min_mag)
            
    for i in range(len(mag_array)): #   First plot them all as black dots
        if float(mag_array[i]) >= min_mag and float(year_array[i]) >= plot_start_year:
            ax.plot(float(lng_array[i]), float(lat_array[i]), '.k', ms=1, zorder=1)
        
        if float(mag_array[i]) >= 6.0 and float(mag_array[i]) < 6.89999  and float(year_array[i]) >= plot_start_year:
 #            ax.plot(float(lng_array[i]), float(lat_array[i]), 'g*', ms=11, zorder=2)
            ax.plot(float(lng_array[i]), float(lat_array[i]), 'o', mec='b', mfc='None', mew=1.25, \
                ms=6, zorder=2)
            
        if float(mag_array[i]) >= 6.89999 and float(year_array[i]) >= plot_start_year:
#             ax.plot(float(lng_array[i]), float(lat_array[i]), 'y*', ms=15, zorder=2)
            ax.plot(float(lng_array[i]), float(lat_array[i]), 'o', mec='r', mfc='None', mew=1.25,\
                ms=12, zorder=2)
# 
    SupTitle_text = 'Seismicity for $M\geq$' + str(min_mag)  + ' after ' + str(plot_start_year)
    plt.suptitle(SupTitle_text, fontsize=14, y=0.98)
#     
    Title_text = 'Within ' + str(delta_deg_lat) + '$^o$ Latitude and ' + str(delta_deg_lng) + '$^o$ Longitude of ' + Location 
    plt.title(Title_text, fontsize=10)
    
    #   -------------------------------------------------------------

    figure_name = './Data/Seismicity_Map_' + Location + '_' + str(plot_start_year) + '.png'
    plt.savefig(figure_name,dpi=600)

    plt.show()
    
    plt.close()

    #   -------------------------------------------------------------
    
    return None

    ######################################################################
    
def plot_Reliability(observed_stats_list, precision_list, \
                    forecast_interval, mag_large, mag_large_plot, min_mag,\
                    data_string_title, number_thresholds, NELng_local, SWLng_local, NELat_local, SWLat_local, \
                    Location, NSteps, delta_time, lower_cutoff):
                    
    fig, ax = plt.subplots()

    label_text = 'Reliability for $M\geq$'+ str(mag_large) 
    
    precision_list = [precision_list[i]*100.0 for i in range(len(precision_list))]
    observed_stats_list = [observed_stats_list[i]*100.0 for i in range(len(observed_stats_list))]

    ax.plot(precision_list, observed_stats_list, linestyle='-', lw=1.0, color='b', zorder=3, label=label_text)
    
    ax.minorticks_on()
        
    x_line = [0.,100.]
    y_line = [0.,100.]
    
    ax.plot(x_line, y_line, linestyle='-', lw=1.0, color='k', zorder=2, label = 'Random Mean')
    
    #   -----------------------------------------------------------------------
    
    SupTitle_text = 'Reliability'
    plt.suptitle(SupTitle_text, fontsize=12, y = 0.96)
    
    delta_deg_lat = (NELat_local  - SWLat_local) * 0.5
    delta_deg_lng = (NELng_local  - SWLng_local) * 0.5
#     
    Title_text = 'Within ' + str(delta_deg_lat) + '$^o$ Latitude and ' + str(delta_deg_lng) + '$^o$ Longitude of ' + Location 
    plt.title(Title_text, fontsize=10)
    
    plt.ylabel('Observed Frequency (%)', fontsize = 12)
    plt.xlabel('Probability (PPV, %)', fontsize = 12)
    
    figure_name = './Data/Reliability_' + Location + '.png'
    plt.savefig(figure_name,dpi=200)

    plt.show()
                    
    return
    
    ######################################################################
    
def map_RI_contours(NELat_local, NELng_local, SWLat_local, SWLng_local, \
        grid_size, start_year, end_year, time_stamp, catalog, min_mag, Location):

    #   Note:  eigen_number is the number of the eigenvector as defined in descending
    #       order by eigenvalue
    
    #   Note:  This uses the new Cartopy interface
    #
    #   -----------------------------------------
    #   Set up data to plot
    
    mag_array, date_array, time_array, year_array, depth_array, lat_array, lng_array = \
            ISRFileMethods.read_regional_catalog(min_mag)
            
    lat, lng, lat_index, lng_index = ISRFileMethods.read_grid_file()
    
    #
    #   -----------------------------------------
    #   Define plot map
    
    dateline_crossing = False
    
    #   California
    left_long   = SWLng_local
    right_long  = NELng_local - 0.1
    top_lat     = SWLat_local - 0.1
    bottom_lat  = NELat_local
    
#     left_long   =   -124
#     right_long  =   -112
    
    longitude_labels = [left_long, right_long]
    longitude_labels_dateline = [left_long, 180, right_long, 360]   #   If the map crosses the dateline
    
    central_long_value = 0
    if dateline_crossing:
        central_long_value = 180
        
    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=central_long_value))
    ax.set_extent([left_long, right_long, bottom_lat, top_lat])

    # Create a feature for States/Admin 1 regions at 1:50m from Natural Earth
    states_provinces = cfeature.NaturalEarthFeature(
        category='cultural',
        name='admin_1_states_provinces_lines',
        scale='50m',
        facecolor='none')

    land_10m = cfeature.NaturalEarthFeature('physical', 'land', '10m',
                                        edgecolor='face',
                                        facecolor='coral')
                                        

                                        
    ocean_10m_3000 = cfeature.NaturalEarthFeature('physical', 'bathymetry_H_3000', '10m',
#                                         edgecolor='black',
                                        facecolor='#0000FF',
                                        alpha = 0.3)
                                        

                                        
    lakes_10m = cfeature.NaturalEarthFeature('physical', 'lakes', '10m',
#                                        edgecolor='black',
                                        facecolor='blue',
                                        alpha = 0.75)
                                        
    rivers_and_lakes = cfeature.NaturalEarthFeature('physical', 'rivers_lakes_centerlines', '10m',
#                                        edgecolor='aqua',
                                        facecolor='blue',
                                        alpha = 0.75)

    ax.add_feature(ocean_10m_3000)

    ax.add_feature(cfeature.BORDERS, linestyle='-', alpha=.5, linewidth=0.5)
    ax.add_feature(cfeature.LAKES, alpha=0.95)
    ax.add_feature(cfeature.RIVERS, linewidth= 0.5)
    ax.add_feature(cfeature.STATES, edgecolor='gray',linewidth= 0.5)
#     ax.add_feature(states_provinces, edgecolor='gray')
    ax.coastlines(resolution='10m', color='black', linewidth=0.5)
    
#     stamen_terrain = cimgt.StamenTerrain()
    stamen_terrain = cimgt.Stamen('terrain-background')
    #   Zoom level should not be set to higher than about 6
    ax.add_image(stamen_terrain, 6)

    if dateline_crossing == False:
        gl = ax.gridlines(crs = ccrs.PlateCarree(), draw_labels=True,
                   linewidth=0.25, color='black', alpha=0.5, linestyle='dotted')
                   
    if dateline_crossing == True:
        gl = ax.gridlines(xlocs=longitude_labels_dateline, draw_labels=True,
                   linewidth=1.0, color='white', alpha=0.5, linestyle='--')

    gl.top_labels = False
    gl.right_labels = False
    gl.xlines = True
    gl.ylines = True

    if catalog == 'LosAngeles':
        gl.xlocator = mticker.FixedLocator([-112,-114,-116, -118, -120, -122])
    
    if catalog == 'Tokyo':
        gl.xlocator = mticker.FixedLocator([132,134,136, 138, 140, 142, 144, 146])
#    gl.xlocator = mticker.FixedLocator([left_side, right_side])

#    gl.xlocator = mticker.FixedLocator([left_side, right_side])

    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    
    #   -----------------------------------------
    #   Put california faults on the map
    
    input_file_name = './California_Faults.txt'
    input_file  =   open(input_file_name, 'r')
    
    for line in input_file:
        items = line.strip().split()
        number_points = int(len(items)/2)
        
        for i in range(number_points-1):
            x = [float(items[2*i]),float(items[2*i+2])]
            y = [float(items[2*i+1]), float(items[2*i+3])]
            ax.plot(x,y,'r-', lw=0.55, zorder=2)
            
    input_file.close()
    #   -----------------------------------------
    #
    #   Read the timeseries data from timeseries.txt  Then use only the data between 
    #       start_date and end_date to contour
    
    time_bins, timeseries = ISRFileMethods.get_timeseries_data(min_mag)
    

    lat_grid,lng_grid,lat_indices,lng_indices     = ISRFileMethods.read_grid_file()
    
    nlat = int( (NELat_local - SWLat_local)/grid_size )
    nlng = int( (NELng_local - SWLng_local)/grid_size)
    
    half_grid_size = grid_size * 0.5
    
    lat_array   =   [SWLat_local + half_grid_size + i*grid_size for i in range(nlat)]
    lng_array   =   [SWLng_local + half_grid_size + i* grid_size for i in range(nlng)]
    
    ntotal = nlat*nlng

    relative_intensity =   np.zeros((nlat,nlng)) #   Define an empty array with nlat rows and nlon columns
    
    for i in range(len(lng_grid)):
        try:
            relative_intensity[lat_index[i]][lng_index[i]] =   math.log(1. + sum(timeseries[i]),10)
        except:
            pass
# 
    SupTitle_text = '$Log_{10}$(Relative Intensity) ' + ' From ' + str(start_year) + ' To ' + str(end_year)
    plt.suptitle(SupTitle_text, fontsize=11)
    
    delta_deg_lat = (NELat_local  - SWLat_local) * 0.5
    delta_deg_lng = (NELng_local  - SWLng_local) * 0.5
#     
    Title_text = 'Within ' + str(delta_deg_lat) + '$^o$ Latitude and ' + str(delta_deg_lng) + '$^o$ Longitude of ' + Location 
    plt.title(Title_text, fontsize=10)
    
    relative_intensity =  scipy.ndimage.zoom(relative_intensity, 15)
    lng_array    =  scipy.ndimage.zoom(lng_array, 15)
    lat_array    =  scipy.ndimage.zoom(lat_array, 15)   
    
    relative_intensity = np.ma.array(relative_intensity, mask = abs(relative_intensity) < 0.75)
    
    im = ax.contourf( lng_array, lat_array, relative_intensity, 10, cmap='rainbow', alpha = 0.50, transform=ccrs.PlateCarree())
    ax.contourf(lng_array, lat_array, relative_intensity, 10, transform=ccrs.PlateCarree(), cmap='rainbow', alpha = 0.50, zorder=3)
    
    plt.colorbar(im)
#     
#     contours = plt.contourf(lng_array, lat_array, eigen_value, 10, cmap='jet', alpha = 0.3, zorder=3)
#     plt.clabel(contours, inline=True, fontsize=6)
#     plt.colorbar()
        
    #   -------------------------------------------------------------

#     time_stamp = str(time_stamp)
#     figure_name = './Data/Eigenvector_' + str(eigen_number) + '_' +  time_stamp + '.png'

    figure_name = './Data/Relative_Intensity_' + '_' + str(start_year) + '_' + str(end_year) + '.png'
    plt.savefig(figure_name,dpi=600)
    
    plt.close()

    #   -------------------------------------------------------------
    
    return None

    ######################################################################
    
 
def map_RTI_time_slice_contours(NELat_local, NELng_local, SWLat_local, SWLng_local, \
            grid_size, min_mag, lower_mag, Location, index_movie, delta_time_interval, catalog,\
            timeseries_EMA_reduced, time_list_reduced, date_bins_reduced, NSTau, NSteps, forecast_interval,lower_cutoff, min_rate,\
            mag_array_large, date_array_large, time_array_large, year_array_large, \
            depth_array_large, lat_array_large, lng_array_large):
            

    #   Note:  eigen_number is the number of the eigenvector as defined in descending
    #       order by eigenvalue
    
    #   Note:  This uses the new Cartopy interface
    #
    #   -----------------------------------------
    #   Set up data to plot
    
    lat, lng, lat_index, lng_index = SEISRFileMethods.read_grid_file()
    
    #
    #   -----------------------------------------
    #   Define plot map
    
    dateline_crossing = False
    
    #   California
    left_long   = SWLng_local
    right_long  = NELng_local - 0.1
    top_lat     = SWLat_local - 0.1
    bottom_lat  = NELat_local
    
    
    longitude_labels = [left_long, right_long]
    longitude_labels_dateline = [left_long, 180, right_long, 360]   #   If the map crosses the dateline
    
    central_long_value = 0
    if dateline_crossing:
        central_long_value = 180
        
    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=central_long_value))
    ax.set_extent([left_long, right_long, bottom_lat, top_lat])

    # Create a feature for States/Admin 1 regions at 1:50m from Natural Earth
    states_provinces = cfeature.NaturalEarthFeature(
        category='cultural',
        name='admin_1_states_provinces_lines',
        scale='50m',
        facecolor='none')

    land_10m = cfeature.NaturalEarthFeature('physical', 'land', '10m',
                                        edgecolor='face',
                                        facecolor='coral')
                                        

                                        
    ocean_10m_3000 = cfeature.NaturalEarthFeature('physical', 'bathymetry_H_3000', '10m',
#                                         edgecolor='black',
                                        facecolor='#0000FF',
                                        alpha = 0.3)
                                        

                                        
    lakes_10m = cfeature.NaturalEarthFeature('physical', 'lakes', '10m',
#                                        edgecolor='black',
                                        facecolor='blue',
                                        alpha = 0.75)
                                        
    rivers_and_lakes = cfeature.NaturalEarthFeature('physical', 'rivers_lakes_centerlines', '10m',
#                                        edgecolor='aqua',
                                        facecolor='blue',
                                        alpha = 0.75)

#     ax.add_feature(ocean_10m_3000)

    ax.add_feature(cfeature.BORDERS, linestyle='-', alpha=.5, linewidth=0.5)
    ax.add_feature(cfeature.LAKES, alpha=0.95)
    ax.add_feature(cfeature.RIVERS, linewidth= 0.5)
    ax.add_feature(cfeature.STATES, edgecolor='gray',linewidth= 0.5)
#     ax.add_feature(states_provinces, edgecolor='gray')
    ax.coastlines(resolution='10m', color='black', linewidth=0.5)
    
#     stamen_terrain = cimgt.StamenTerrain()
    stamen_terrain = cimgt.Stamen('terrain-background')
    
    #   Zoom level should not be set to higher than about 6
#     ax.add_image(stamen_terrain, 6)

    if dateline_crossing == False:
        gl = ax.gridlines(crs = ccrs.PlateCarree(), draw_labels=True,
                   linewidth=0.25, color='black', alpha=0.5, linestyle='dotted')
                   
    if dateline_crossing == True:
        gl = ax.gridlines(xlocs=longitude_labels_dateline, draw_labels=True,
                   linewidth=1.0, color='white', alpha=0.5, linestyle='--')

    gl.top_labels = False
    gl.right_labels = False
    gl.xlines = True
    gl.ylines = True

    if catalog == 'LosAngeles':
        gl.xlocator = mticker.FixedLocator([-112,-114,-116, -118, -120, -122])
    
    if catalog == 'Tokyo':
        gl.xlocator = mticker.FixedLocator([132,134,136, 138, 140, 142, 144, 146])
#    gl.xlocator = mticker.FixedLocator([left_side, right_side])

#    gl.xlocator = mticker.FixedLocator([left_side, right_side])

    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    
    #   -----------------------------------------
    #   Put california faults on the map
    
    input_file_name = './California_Faults.txt'
    input_file  =   open(input_file_name, 'r')
    
    for line in input_file:
        items = line.strip().split()
        number_points = int(len(items)/2)
        
        for i in range(number_points-1):
            x = [float(items[2*i]),float(items[2*i+2])]
            y = [float(items[2*i+1]), float(items[2*i+3])]
#             ax.plot(x,y,'r-', lw=0.55, zorder=2)
            ax.plot(x,y,'-', color='darkgreen',lw=0.55, zorder=2)
            
    input_file.close()
    
    #
    #   -----------------------------------------
    #

    levels_list = arange(0.1,1.1,0.1)
    
#   Define the Logistic Function: Short term EMA divided by Long term EMA

 #    ROC_gridbox_threshold_list = sort_list_EQ_RTI_order(ROC_event_list,  NELat_local, NELng_local, SWLat_local, SWLng_local, \
#             min_mag, index_ROC, timeseries, NSTau)

    ROC_event_list = \
            SEISRCalcMethods.classify_large_earthquakes_grid_boxes(NELat_local, NELng_local, SWLat_local, SWLng_local, \
                        grid_size, index_movie, time_list_reduced, forecast_interval,\
                        mag_array_large, year_array_large, depth_array_large, lat_array_large, lng_array_large) 
                        
    
    ROC_gridbox_threshold_list = \
                SEISRCalcMethods.sort_list_EQ_RTI_order(ROC_event_list,  NELat_local, NELng_local, SWLat_local, SWLng_local,\
                min_mag, index_movie, timeseries_EMA_reduced, NSTau, lower_cutoff)
                
    #   -----------------------------------------
            
    lat_grid,lng_grid,lat_indices,lng_indices     = SEISRFileMethods.read_grid_file()
    
    nlat = int( (NELat_local - SWLat_local)/grid_size )
    nlng = int( (NELng_local - SWLng_local)/grid_size)
    
    half_grid_size = grid_size * 0.5
    
    lat_array   =   [SWLat_local + half_grid_size + i*grid_size for i in range(nlat)]
    lng_array   =   [SWLng_local + half_grid_size + i* grid_size for i in range(nlng)]
    
    ntotal = nlat*nlng

    relative_intensity =   np.zeros((nlat,nlng)) #   Define an empty array with nlat rows and nlon columns
    
    for i in range(len(lng_grid)):
        try:
#             relative_intensity[lat_index[i]][lng_index[i]] =   RTI_list[i]
            relative_intensity[lat_index[i]][lng_index[i]] =   ROC_gridbox_threshold_list[i]
        except:
            pass
# 
#     current_date_dash, current_date = SEISRUtilities.convert_partial_year(time_bins[index_movie])
    

#     SupTitle_text = '$Log_{10}$(Relative Intensity) ' + ' At Time ' + str(round(time_bins[time_index],3))
    SupTitle_text = '$Log_{10}$(1.0 + Relative Total Intensity in %)'
    plt.suptitle(SupTitle_text, fontsize=11)
    
    delta_deg_lat = (NELat_local  - SWLat_local) * 0.5
    delta_deg_lng = (NELng_local  - SWLng_local) * 0.5
#     
    Title_text = 'Within ' + str(delta_deg_lat) + '$^o$ Latitude and ' + str(delta_deg_lng) + '$^o$ Longitude of ' + Location\
            + ' on ' + str(date_bins_reduced[index_movie])
    plt.title(Title_text, fontsize=9)
    
    relative_intensity =  scipy.ndimage.zoom(relative_intensity, 15)
    lng_array    =  scipy.ndimage.zoom(lng_array, 15)  
    lat_array    =  scipy.ndimage.zoom(lat_array, 15)   
    
    #   -----------------------------------------
    #
    #   NOTE:   To find the correct levels, first just use levels=10 or similar.  See what that gives you, then define the
    #           levels_list
    
    #   Set the plot levels.  At the moment, levels are unscaled, so large values saturate
    
    levels_list = arange(0.1,1.1, 0.1)
    
    #   -----------------------------------------
    #
    #   Make the contour plots
    
#     relative_intensity = np.ma.array(relative_intensity, mask = relative_intensity < 0.0)
# 
#     im = ax.contourf( lng_array, lat_array, relative_intensity, levels = 10, extend = "both", cmap='rainbow', alpha = 0.30, \
#         transform=ccrs.PlateCarree())
# 
#     ax.contourf(lng_array, lat_array, relative_intensity, levels = 10, transform=ccrs.PlateCarree(), extend = "both",\
#             cmap='rainbow', alpha = 0.30, zorder=3)
    
    relative_intensity = np.ma.array(relative_intensity, mask = relative_intensity < levels_list[0])
    
    im = ax.contourf( lng_array, lat_array, relative_intensity, levels = levels_list, extend = "both", cmap='rainbow', alpha = 0.25,\
             transform=ccrs.PlateCarree())
    
    ax.contourf(lng_array, lat_array, relative_intensity, levels = levels_list, transform=ccrs.PlateCarree(), extend = "both",\
            cmap='rainbow', alpha = 0.25, zorder=3)
            
    plt.colorbar(im)
    #
    #   -----------------------------------------
    #
    #   Plot large earthquakes that occur within 3 years

    mag_array, date_array, time_array, year_array, depth_array, lat_array, lng_array = \
            SEISRFileMethods.read_regional_catalog(min_mag)
            
    lower_time = time_list_reduced[index_movie] 
    upper_time = lower_time + forecast_interval 
    
    for i in range(len(mag_array)): #   First plot them all as black dots

        
        if float(mag_array[i]) >= lower_mag and float(mag_array[i]) < 4.9 \
                and float(year_array[i]) >= lower_time and float(year_array[i]) < upper_time:
                
            ax.plot(float(lng_array[i]), float(lat_array[i]), 'o', mec='k', mfc='None', mew=0.5, ms = 2, zorder=4)
        
        if float(mag_array[i]) >= 5.0 and float(mag_array[i]) < 5.9 \
                and float(year_array[i]) >= lower_time and float(year_array[i]) < upper_time:
                
            ax.plot(float(lng_array[i]), float(lat_array[i]), 'o', mec='k', mfc='None', mew=0.75, ms=5, zorder=5)
        
        if float(mag_array[i]) >= 6.0 and float(mag_array[i]) < 6.89999  \
                and float(year_array[i]) >= lower_time and float(year_array[i]) < upper_time:
 
            ax.plot(float(lng_array[i]), float(lat_array[i]), 'o', mec='b', mfc='None', mew=0.75, \
                    ms=9, zorder=6)
            
        if float(mag_array[i]) >= 6.89999 and float(year_array[i]) >= lower_time and float(year_array[i]) < upper_time:

            ax.plot(float(lng_array[i]), float(lat_array[i]), 'o', mec='r', mfc='None', mew=0.75,\
                    ms=14, zorder=7)
                    
#     ax.plot(float(lng_array[i])+ 1000., float(lat_array[i])+1000., '.','k', ms=2, zorder=4,\
#                     label= str(lower_mag) + '$ > M \geq $'+str(min_mag))

#     print(min_mag, lower_mag)
                    
    ax.plot(float(lng_array[i])+ 1000., float(lat_array[i])+1000., 'o', mec='k', mfc='None', mew=0.5, ms = 2, \
                    zorder=4, label= str(4.9) + '$ > M \geq $'+str(lower_mag))
                    
    ax.plot(float(lng_array[i])+ 1000., float(lat_array[i])+1000., 'o', mec='k', mfc='None', mew=0.75, ms=5, \
                    zorder=4, label='$5.9 > M \geq $'+str(5.0))
                    
    ax.plot(float(lng_array[i])+1000., float(lat_array[i])+1000., 'o', mec='b', mfc='None', mew=0.75, \
                    ms=9, zorder=4, label='\n$6.9 > M \geq 6.0$')
                    
    ax.plot(float(lng_array[i])+1000., float(lat_array[i])+1000., 'o', mec='r', mfc='None', mew=0.75,\
                    ms=14, zorder=4, label='\n$M \geq 6.9$')
                    
    #
    #   -----------------------------------------
    #
    
    test_time_interval = delta_time_interval/0.07692
    if abs(test_time_interval-1.0) <0.01:
        str_time_interval = '1 Month'
    elif abs(test_time_interval-0.25) < 0.01:
        str_time_interval = '1 Week'
    elif abs(test_time_interval-2.0) < 0.01:
        str_time_interval = '2 Months'
    elif abs(test_time_interval-3.0) < 0.01:
        str_time_interval = '3 Months'
        
    textstr =   '$T_{D}$ (Months): ' + str(NSTau) +\
                '\nTime Step: ' + str_time_interval +\
                '\nGrid Size: ' + str(grid_size) + '$^o$' +\
                '\n$R_{min}$: ' + str(round(min_rate,0)) +\
                '\n$M_{min}$: ' + str(round(min_mag,2))
    
#     textstr =   'Time Step: ' + str_time_interval +\
#                '\n$M_{min}$: ' + str(round(min_mag,2))
 
    #     # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='white', edgecolor = 'gray', alpha=0.75)

    #     # place a text box in upper left in axes coords
    ax.text(0.98, 0.98, textstr, transform=ax.transAxes, fontsize=6,
        verticalalignment='top', horizontalalignment = 'right', bbox=props, linespacing = 1.8)
    
    title_text = 'Future Earthquakes\nWithin $T_W$ = ' + str(forecast_interval) + ' Years'
    leg = ax.legend(loc = 'lower left', title=title_text, fontsize=6)
    leg.set_title(title_text,prop={'size':6})   #   Set the title text font size

    #
    #   -----------------------------------------
    #
    
#     
#     contours = plt.contourf(lng_array, lat_array, eigen_value, 10, cmap='jet', alpha = 0.3, zorder=3)
#     plt.clabel(contours, inline=True, fontsize=6)
#     plt.colorbar()
        
    #   -------------------------------------------------------------

#     time_stamp = str(time_stamp)
#     figure_name = './Data/Eigenvector_' + str(eigen_number) + '_' +  time_stamp + '.png'

    figure_name = './DataMoviesLogRTI/LogRTI_Time_Slice_000' +  str(index_movie) + '.png'
    plt.savefig(figure_name,dpi=300)
    
#     figure_name = './Data/RI_Time_Slice_' + str(NSteps_RI) + '_' + str(current_date_dash) + '.png'
#     plt.savefig(figure_name,dpi=300)
    
#     plt.show()
    
    plt.close()

    #   -------------------------------------------------------------
    
    return None

    ######################################################################

def plot_spatial_ROC_diagram(tpr_list, fpr_list,  NSTau, delta_time_interval,\
        lower_mag, min_rate, min_mag, grid_size, begin_date_of_plot, end_date_of_plot, forecast_interval,\
        delta_deg_lat, delta_deg_lng, Location):

    fig, ax = plt.subplots()
    
    x = list(range(len(tpr_list[0])))
    x = [float(x[i])/float(len(tpr_list[0])) for i in range(len(tpr_list[0]))]
    x.append(1.)
    y=x
    
    skill_score_list = []
    
    tpr_mean_list = [0. for i in range(len(tpr_list[0]))]
    fpr_mean_list = [0. for i in range(len(fpr_list[0]))]
    
    for k in range(len(tpr_list)):
    
        true_positive_rate  = tpr_list[k]
        false_positive_rate = fpr_list[k]
    
        tpr_array = np.array(true_positive_rate)
        fpr_array = np.array(false_positive_rate)
    
        area_trapz = trapz(tpr_array, fpr_array)    #   Works with non-equidistantly tabulated data
#         print("Trap Area = ", area_trapz)
# 
        skill_score = round(area_trapz,3)
        skill_score_list.append(skill_score)
        
        ax.plot(false_positive_rate, true_positive_rate, '-', color='cyan', alpha = 0.5, lw = 1.15, zorder=2)
        
        for j in range(len(tpr_list[0])):
            tpr_mean_list[j] += true_positive_rate[j]
            fpr_mean_list[j] += false_positive_rate[j]
            
    tpr_mean = []
    fpr_mean = []
    number_ROC_curves = len(tpr_list)
    
    for j in range(len(tpr_mean_list)):
        tpr_mean.append(tpr_mean_list[j]/float(number_ROC_curves))
        fpr_mean.append(fpr_mean_list[j]/float(number_ROC_curves))
        
    ax.plot(fpr_mean, tpr_mean, '-', color='r', alpha = 1.0, lw = 1.15, zorder=2)
        
    skill_score_mean = np.mean(skill_score_list)
    skill_score_mean = round(skill_score_mean,3)
    
    skill_score_std   = np.std(skill_score_list)
    skill_score_std   = round(skill_score_std,3)
    
    
#     print('Skill Score Mean: ', skill_score_mean)
#     print('Skill Score Std. Dev.:', skill_score_std)    
    
    ax.plot(x,y,'--', color='k', zorder=1)
        
    ax.grid(True, lw = 0.5, which='major', linestyle='dotted')
    
#     ax.legend(loc = 'lower right', fontsize=8)
        
    SupTitle_text = 'Spatial Receiver Operating Characteristic'
    plt.suptitle(SupTitle_text, fontsize=11)
    
    Title_text = 'Within ' + str(delta_deg_lat) + '$^o$ Latitude and ' + str(delta_deg_lng) + '$^o$ Longitude of ' + Location\
            + ' From ' + str(begin_date_of_plot) + ' to ' + str(end_date_of_plot)
    plt.title(Title_text, fontsize=9)
        
    plt.ylabel('True Positive Rate', fontsize = 10)
    plt.xlabel('False Positive Rate', fontsize = 10)
    
    #
    #   -----------------------------------------
    #

    test_time_interval = delta_time_interval/0.07692
    
    if abs(test_time_interval-1.0) <0.01:
        str_time_interval = '1 Month'
    elif(abs(test_time_interval-0.25) < 0.01):
        str_time_interval = '1 Week'
    elif(abs(test_time_interval-2.0) < 0.01):
        str_time_interval = '2 Months'
    elif(abs(test_time_interval-3.0) < 0.01):
        str_time_interval = '3 Months'
        
#     textstr =   'Skill: ' + str(round(skill_score_mean,3)) + ' $\pm$ ' + str(round(skill_score_std,3)) +\
#                 '$\nT_{D}$ (Months): ' + str(NSTau) +\
#                 '\nTime Step: ' + str_time_interval +\
#                 '\nGrid Size: ' + str(grid_size) + '$^o$' +\
#                 '\n$R_{min}$ = ' + str(round(min_rate,0)) +\
#                 '\n$M_{min}$: ' + str(round(min_mag,2)) +\
#                 '\nNowcast: ' + str(forecast_interval) + ' Years'

    textstr =   'Skill: ' + str(round(skill_score_mean,3)) + '$\pm$' + str(round(skill_score_std,3)) +\
                '\nNowcast: ' + str(forecast_interval) + ' Years' +\
                '\n$T_{D}$ (Months): ' + str(NSTau) +\
                '\nTime Step: ' + str_time_interval +\
                '\nGrid Size: ' + str(round(grid_size,3)) + '$^o$' +\
                '\n$R_{min}$: ' + str(round(min_rate,0)) +\
                '\n$M_{min}$: ' + str(round(min_mag,2))

    
    #     # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='white', edgecolor = 'gray', alpha=0.75)

    #     # place a text box in upper left in axes coords
    ax.text(0.98, 0.02, textstr, transform=ax.transAxes, fontsize=8,
        verticalalignment='bottom', horizontalalignment = 'right', bbox=props, linespacing = 1.8)

    plot_year = begin_date_of_plot[:4]
    figure_name = './Predictions/Spatial_ROC_Mmin' + str(min_mag) + '_Tau' + str(NSTau)  + '_Grid' + str(grid_size) \
            + '_' + plot_year + '.png'
    plt.savefig(figure_name,dpi=300)
    
    plt.close()
        
#     plt.show()

    return
    
        ######################################################################
    
def plot_mean_eqs_timeseries(timeseries, time_bins, date_bins, plot_start_year, mag_large_plot, forecast_interval, \
        NELng_local, SWLng_local, NELat_local, SWLat_local, Location, NSteps, delta_time_interval, min_mag, lower_cutoff, min_rate):
#     
#
#   ------------------------------------------------------------
#
    year_large_eq, mag_large_eq, index_large_eq = SEISRCalcMethods.get_large_earthquakes(mag_large_plot,min_mag)
    
#
#   ------------------------------------------------------------
#

    eqs_list = []
    
    for i in range(len(time_bins)):        #   Over all bins
        eq_sum = 0
        
        for j in range(len(timeseries)):
            eq_sum += timeseries[j][i]
            
        eqs_list.append(eq_sum)
    
    eq_means = []
    for i in range(len(eqs_list)):
        eq_means.append(np.mean(eqs_list[:i]))

#     first_values = int(forecast_interval * 13)
#     eq_means = eq_means[:-first_values]
#     time_bins = time_bins[:-first_values]
    
    fig, ax = plt.subplots()
    
    ax.plot(time_bins, eq_means, linestyle='-', lw=1.0, color='b', zorder=3, label='Mean $M$ > ' + str(min_mag))

    xmin, xmax = plt.xlim()
    ymin, ymax = plt.ylim()
    

   #   -------------------------------------------------------------
   
    year_large_eq, mag_large_eq, index_large_eq = \
            SEISRCalcMethods.adjust_year_times(year_large_eq, mag_large_eq, index_large_eq, time_bins, time_bins[0])

    for i in range(len(year_large_eq)):

        if float(mag_large_eq[i]) >= 6.0 and float(mag_large_eq[i]) < 6.89999:
            x_eq = [year_large_eq[i], year_large_eq[i]]
            y_eq = [ymin,eq_means[index_large_eq[i] ]]
            
            ax.plot(x_eq, y_eq, linestyle='dotted', color='k', lw=0.7, zorder=2)
            
    ax.plot(x_eq,y_eq, linestyle='dotted', color='k', lw=0.7, zorder=2, label = '6.9 $>$ M $\geq$ 6.0')
            
    for i in range(len(year_large_eq)):

        if float(mag_large_eq[i]) >= 6.89999:
            x_eq = [year_large_eq[i], year_large_eq[i]]
            y_eq = [ymin,eq_means[index_large_eq[i] ]]
            
            ax.plot(x_eq, y_eq, linestyle='--', color='r', lw=0.7, zorder=2)
            
    ax.plot(x_eq,y_eq, linestyle='--', color='r', lw=0.7, zorder=2, label='M $\geq$ 6.9')

    #   -------------------------------------------------------------
    
    min_plot_line = [ymin for i in range(len(time_bins))]
    ax.fill_between(time_bins, min_plot_line, eq_means, color='c', alpha=0.1, zorder=0)
    
#     plt.gca().invert_yaxis()
            
    ax.grid(True, lw = 0.5, which='major', linestyle='dotted', axis = 'both')
    
    ax.legend(loc = 'lower right', fontsize=10)
    
    #     
    #   ------------------------------------------------------------
    #
            
    test_time_interval = delta_time_interval/0.07692
    if abs(test_time_interval-1.0) <0.01:
        str_time_interval = '1 Month'
    elif abs(test_time_interval-0.25) < 0.01:
        str_time_interval = '1 Week'
    elif abs(test_time_interval-2.0) < 0.01:
        str_time_interval = '2 Months'
    elif abs(test_time_interval-3.0) < 0.01:
        str_time_interval = '3 Months'
        
    textstr =   'Time Step: ' + str_time_interval +\
                '\n$M_{min}$: ' + str(round(min_mag,2))

# 
#     # these are matplotlib.patch.Patch properties
#     props = dict(boxstyle='round', facecolor='white', edgecolor = 'gray', alpha=0.5)
# 
# #     # place a text box in upper left in axes coords
#     ax.text(0.975, 0.035, textstr, transform=ax.transAxes, fontsize=12,
#         verticalalignment='bottom', horizontalalignment = 'right', bbox=props, linespacing = 1.8)

#   ------------------------------------------------------------
#

    ax.minorticks_on()
    
    delta_deg_lat = (NELat_local  - SWLat_local) * 0.5
    delta_deg_lng = (NELng_local  - SWLng_local) * 0.5
    
    SupTitle_text = 'Mean Number of Small Earthquakes vs. Time'

    plt.suptitle(SupTitle_text, fontsize=14, y = 0.98)
    
    Title_text = 'Within ' + str(delta_deg_lat) + '$^o$ Latitude and ' + str(delta_deg_lng) + '$^o$ Longitude of ' + Location
            
    plt.title(Title_text, fontsize=10)
    
    plt.ylabel('Mean Monthly Number', fontsize = 12)
    plt.xlabel('Time (Year)', fontsize = 12)
    
#     data_string_title = 'EMA' + '_FI' + str(forecast_interval) + '_TTI' + str(test_time_interval) + \
#             '_NSTP' + str(NSteps) + '_MM' + str(min_mag) + '_CF' + str(lower_cutoff) 

#     figure_name = './Data/SEISR_' + data_string_title + '_' + str(plot_start_year) + '.png'
#     plt.savefig(figure_name,dpi=600)
#     
    plt.show()
#     matplotlib.pyplot.close('all')
    plt.close('all')

    return 
    
    ######################################################################
    

    
def plot_test():

    print('Hello2')

    fig = plt.figure()        #   Define large figure and thermometer - 4 axes needed

    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    x = list(range(10))
    y=x
    
    ax1.plot(x,y)    
    
    y = [-x[i] for i in range(len(x))]
    
    ax2.plot(x,y)
    
    plt.show()
    
    plt.close()
    
    return
    
    ######################################################################
 