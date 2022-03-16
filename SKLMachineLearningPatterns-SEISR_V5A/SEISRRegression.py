#!/opt/local/bin python

    #   TimeSeriesForecasting.0.py    -   This version will use the correlations to form time series,
    #                                       then do time series forecasting
    #
    #   This is an implementation of the code at:
    #
    #   https://machinelearningmastery.com/random-forest-for-time-series-forecasting/
    #
    #   This version steps through time, updating the eigenpatterns on each time step.  Only 1 test feature
    #       vector is used on each time step.
    #
    #   This version differs from versions 4.* in that the we use all the data to build
    #       the pattern vectors and step through forecasts 1 week at a time and
    #       accumulate statistics
    #
    #   Python code to use Scikit_Learn to identify earthquake alerts
    #
    #   In this code, we compute eigenvalues from the entire data set.  Then we
    #       we use the train_test_split() method to build a test data set.  But
    #       note that the feature vectors and the eigenvectors of the sets are not
    #       independent.
    #
    #       So in this code, the test feature vectors and the eigenvectors are NOT
    #       independent.
    #
    #   This code downloads data from the USGS web site.
    #
    #   This code was written on a Mac using Macports python.  A list of the ports needed to run the code are available at:
    #       https://www.dropbox.com/s/8wr5su8d7l7a30z/myports-wailea.txt?dl=0
    
    #   ---------------------------------------------------------------------------------------
    
    # =========================================================
    # Linear Regression Example
    # =========================================================
    # This example uses the only the first feature of the `diabetes` dataset, in
    # order to illustrate a two-dimensional plot of this regression technique. The
    # straight line can be seen in the plot, showing how linear regression attempts
    # to draw a straight line that will best minimize the residual sum of squares
    # between the observed responses in the dataset, and the responses predicted by
    # the linear approximation.
    # 
    # The coefficients, the residual sum of squares and the coefficient
    # of determination are also calculated.
    
    # Code source: Jaques Grobler
    # License: BSD 3 clause
    
    #  https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html#sphx-glr-auto-examples-linear-model-plot-ols-py
    
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
import matplotlib.pyplot as plt
import math

import SKLCalcMethods
import SKLFileMethods
import SKLPlotMethods
import SKLUtilities

from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

    ######################################################################
    
def find_time_next_large_eq(times_window, values_window, year_large_eq, mag_large_eq, index_large_eq, lower_cutoff_date):

    time_to_next_eq =   []
    value_next_eq   =   []
    
    print(year_large_eq)
    print('')

    for i in range (len(times_window)):
        if float(times_window[i]) <= float(year_large_eq[-1:][0]) and float(times_window[i]) >= lower_cutoff_date:
    
            working_list = []
        
            for j in range(len(year_large_eq)):
                working_list.append(year_large_eq[j] - times_window[i])
        
            min_time_interval = min([k for k in working_list if k > 0]) 
    
#             print(times_window[i],min_time_interval)
#             print('')
    
            time_to_next_eq.append(round(min_time_interval,4))
            value_next_eq.append(values_window[i])

    return  time_to_next_eq, value_next_eq
    
    ######################################################################

def linear_regression(X_train, y_train, mag_large, lower_cutoff_date, data_string_title):

    
#     Load the correlation dataset
#     diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)
# 
#     Use only one feature
#     diabetes_X = diabetes_X[:, np.newaxis, 2]
# 
#     Split the data into training/testing sets
#     diabetes_X_train = diabetes_X[:-20]
#     diabetes_X_test = diabetes_X[-20:]
# 
#     Split the targets into training/testing sets
#     diabetes_y_train = diabetes_y[:-20]
#     diabetes_y_test = diabetes_y[-20:]

    X_train_array = np.reshape(X_train,(-1,1))
    
    
# 
#     # Create linear regression object
    regr = linear_model.LinearRegression()
# 
#     # Train the model using the training sets
    regr.fit(X_train_array, y_train)
# 
#     # Make predictions using the testing set
# #     y_pred = regr.predict(X_test)
    y_pred = regr.predict(X_train_array)
# 
#     # The coefficients
#     print('Coefficients: \n', regr.coef_)
#     
#     # The mean squared error
# #     print('Mean squared error: %.2f'% mean_squared_error(y_test, y_pred))
#     print('Mean squared error: %.2f'% mean_squared_error(y_train, y_pred))
#     
#     # The coefficient of determination: 1 is perfect prediction
# #     print('Coefficient of determination: %.2f'% r2_score(y_test, y_pred))
#     print('Coefficient of determination: %.2f'% r2_score(y_train, y_pred))
# 
#     # Plot outputs
# #     plt.scatter(diabetes_X_test, diabetes_y_test,  color='black')
# #     plt.plot(diabetes_X_test, diabetes_y_pred, color='blue', linewidth=3)

#     print('X_train',   X_train)

    plt.plot(X_train, y_train,  'k.', ms = 4)
    plt.plot(X_train, y_pred, color='blue', linewidth=2)
    
    plt.ylabel('Time to Next Large EQ (Years)', fontsize = 12)
    plt.xlabel('Correlation Value', fontsize = 12)
    
    SupTitle_text = 'Time to Next M$\geq$' + str(mag_large) + ' vs. Correlation since ' + str(lower_cutoff_date)

    plt.suptitle(SupTitle_text, fontsize=12, y = 0.96)
    
#     Title_text = 'Forecast: ' + str(round(forecast_interval,1)) + ' Years; ' + 'Longitude(' + str(lngmin) + '$^o$,' + \
#             str(lngmax) + '$^o$); Latitude(' + str(latmin) + '$^o$,' + str(latmax) + '$^o$); NSteps: ' + str(NSteps)
            
#     Title_text = 'Within ' + str(delta_deg_lat) + '$^o$ Latitude and ' + str(delta_deg_lng) + '$^o$ Longitude of ' + Location
#             
#     plt.title(Title_text, fontsize=9)

#     plt.xticks(())
#     plt.yticks(())

    data_string_title_reduced = data_string_title[15:]

    figure_name = './Predictions/Regression_' + data_string_title_reduced + '.png'
    plt.savefig(figure_name,dpi=600)

    plt.show()
    
    return

    ######################################################################
    
def read_input_data(input_file_name):

    input_file = open(input_file_name,'r')
    
#     number_lines = 0
#     for line in input_file:
#         number_lines += 1
        
    values_window       =   []
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

    return values_window, times_window, eqs_window, NELng_local, SWLng_local, NELat_local, SWLat_local, Grid, Location
    
    ######################################################################
    
def get_large_earthquakes(mag_large):

    mag_array, date_array, time_array, year_array, depth_array, lat_array, lng_array = \
            SKLFileMethods.read_regional_catalog()
            
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

if __name__ == '__main__':

    data_string_title = 'confusion_data_M2.99_SF4_TD1_SI13_CF0.5_G0.33_ER[0, 151]'
    input_file_name   = 'Data/' + data_string_title + '.csv'
    input_file_name   = data_string_title + '.csv'
    
    mag_large = 7.0
    lower_cutoff_date = 1980.0
    
    year_large_eq, mag_large_eq, index_large_eq = get_large_earthquakes(mag_large)

    values_window, times_window, eqs_window, NELng_local, SWLng_local, NELat_local, SWLat_local, Grid, Location = \
        read_input_data(input_file_name)
        
    time_next_eq, value_next_eq = find_time_next_large_eq(times_window, values_window, \
            year_large_eq, mag_large_eq, index_large_eq, lower_cutoff_date)
    
#     print('values: ', value_next_eq)
#     print('')
#     print('time_next_eq: ', time_next_eq)
#     print()
    
    linear_regression(value_next_eq, time_next_eq, mag_large, lower_cutoff_date, data_string_title)
    
#     print(time_to_next_eq)



