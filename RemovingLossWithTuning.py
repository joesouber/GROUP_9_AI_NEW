# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 17:36:50 2023

@author: safij
"""

#%% IMPORTS

import random
import matplotlib.pyplot as plt
import numpy as np
import csv
import pandas as pd
import seaborn as sn
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error as mse
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_percentage_error

from Pipeline_attempts import FormatData, Removeoutliars, OneHotEncode, LevyMedian, TrainTestVal, MinMax
from Pipeline_attempts import setup_x_y, RandomForestRegressionModel2, RemoveHighResidualsRandom, RandomForestRegressionModel3, RemoveHighResidualsRandom2
from Pipeline_attempts import XGBR_model, RemoveHighResidualsXGBR, XGBR_model2, RemoveHighResidualsXGBR2

#%% REMOVE ERRORS FROM RANDOM FOREST REGRESSOR
t = RandomForestRegressionModel2()

ns = []
random_r2_new_set = []
random_mape_new_set = []
random_mse_new_set = []
X_train, y_train, X_test, y_test, X_val, y_val = setup_x_y()

for i in range(30):
  n = i*5
  ns.append(n)
  t = RandomForestRegressionModel2()
  tt = RemoveHighResidualsRandom(t, n)
  random_r2_new_set.append(tt[0])
  random_mape_new_set.append(tt[1])
  random_mse_new_set.append(tt[2])


#combine ns and random_r2

random_r2_data = list(zip(random_r2_new_set, ns))
sorted_random_r2_data = sorted(random_r2_data, reverse=True)


random_mape_data = list(zip(random_mape_new_set, ns))
sorted_random_mape_data = sorted(random_mape_data, reverse=False)


random_mse_data = list(zip(random_mse_new_set, ns))
sorted_random_mse_data = sorted(random_mse_data, reverse=False)


print(sorted_random_r2_data[0], sorted_random_mape_data[0], sorted_random_mse_data[0])


t = RandomForestRegressionModel2()
x = ["original model", "remove top 40", "remove top 95"]
y_r2 = [t[0], random_r2_data[8][0], random_r2_data[19][0]]
y_mape = [t[2], random_mape_data[8][0], random_mape_data[19][0]]
y_mse = [t[3], random_mse_data[8][0], random_mse_data[19][0]]

fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(9, 12))

# Plot the data on each subplot and set the title
axes[0].bar(x,y_r2)
axes[0].set_title('R SQUARED')
#axes[0].set_xlabel('Number of error points removed')
axes[0].set_ylabel('R squared score')
axes[0].set_ylim(0.7, 0.8)

#axes[0].set_yscale("log")

axes[1].bar(x, y_mape)
axes[1].set_title('MAPE')
#axes[1].set_xlabel('Number of error points removed')
axes[1].set_ylabel('MAPE score')
axes[1].set_ylim(2.0, 2.2)
#axes[1].set_yscale("log")


axes[2].bar(x, y_mse)
axes[2].set_title('MSE')
#axes[2].set_xlabel('Number of error points removed')
axes[2].set_ylabel('MSE score multiplied by 10^7')
axes[2].set_ylim(70000000, 80000000)
#axes[2].set_yscale("log")

# Set the overall title
fig.suptitle('Random Forest Regression')

# Show the plot
plt.show()

#%% NOW WITH HYPERPARAMETER TUNING

t = RandomForestRegressionModel3()

ns = []
random_r2_new_set1 = []
random_mape_new_set1 = []
random_mse_new_set1 = []
X_train, y_train, X_test, y_test, X_val, y_val = setup_x_y()


for i in range(30):
  n = i*5
  ns.append(n)
  t = RandomForestRegressionModel3()
  tt = RemoveHighResidualsRandom2(t, n)
  random_r2_new_set1.append(tt[0])
  random_mape_new_set1.append(tt[1])
  random_mse_new_set1.append(tt[2])



#combine ns and random_r2

random_r2_data1 = list(zip(random_r2_new_set1, ns))
sorted_random_r2_data1 = sorted(random_r2_data1, reverse=True)


random_mape_data1 = list(zip(random_mape_new_set1, ns))
sorted_random_mape_data1 = sorted(random_mape_data1, reverse=False)


random_mse_data1 = list(zip(random_mse_new_set1, ns))
sorted_random_mse_data1 = sorted(random_mse_data1, reverse=False)


print(sorted_random_r2_data1[0], sorted_random_mape_data1[0], sorted_random_mse_data1[0])



t = RandomForestRegressionModel3()
x1 = ["original model", "remove top 95", "remove top 140"]
y1_r2 = [t[0], random_r2_data1[19][0], random_r2_data1[28][0]]
y1_mape = [t[2], random_mape_data1[19][0], random_mape_data1[28][0]]
y1_mse = [t[3], random_mse_data1[19][0], random_mse_data1[28][0]]

fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(9, 12))

# Plot the data on each subplot and set the title
axes[0].bar(x1,y1_r2)
axes[0].set_title('R SQUARED')
#axes[0].set_xlabel('Number of error points removed')
axes[0].set_ylabel('R squared score')
axes[0].set_ylim(0.72, 0.76)

#axes[0].set_yscale("log")

axes[1].bar(x1, y1_mape)
axes[1].set_title('MAPE')
#axes[1].set_xlabel('Number of error points removed')
axes[1].set_ylabel('MAPE score')
axes[1].set_ylim(2.3, 2.5)
#axes[1].set_yscale("log")


axes[2].bar(x1, y1_mse)
axes[2].set_title('MSE')
#axes[2].set_xlabel('Number of error points removed')
axes[2].set_ylabel('MSE score multiplied by 10^7')
axes[2].set_ylim(75000000, 80000000)
#axes[2].set_yscale("log")

# Set the overall title
fig.suptitle('Random Forest Regression with hyperparameter tuning')

# Show the plot
plt.show()

#%% combine plots

t = RandomForestRegressionModel2()
t1 = RandomForestRegressionModel3()

x2 = ["original model", "remove top 40", "remove top 95", "remove top 140"]
y_r2 = [t[0], random_r2_data[8][0], random_r2_data[19][0], random_r2_data[28][0]]
y_mape = [t[2], random_mape_data[8][0], random_mape_data[19][0], random_mape_data[28][0]]
y_mse = [t[3], random_mse_data[8][0], random_mse_data[19][0], random_mse_data[28][0]]
y1_r2 = [t1[0], random_r2_data1[8][0], random_r2_data1[19][0], random_r2_data1[28][0]]
y1_mape = [t1[2], random_mape_data1[8][0], random_mape_data1[19][0], random_mape_data1[28][0]]
y1_mse = [t1[3], random_mse_data1[8][0], random_mse_data1[19][0], random_mse_data1[28][0]]



fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(8, 12))
width = 0.3

# Plot the data on each subplot and set the title
axes[0].bar(x2,y_r2, width, label = "default")
axes[0].bar([x2 + width for x2 in range(len(y1_r2))],y1_r2, width, label = "with tuning")
axes[0].set_title('R SQUARED')
#axes[0].set_xlabel('Number of error points removed')
axes[0].set_ylabel('R squared score')
axes[0].set_ylim(0.72, 0.76)
axes[0].legend()

axes[1].bar(x2, y_mape, width, label = "default")
axes[1].bar([x2 + width for x2 in range(len(y1_mape))], y1_mape, width, label = "with tuning")
axes[1].set_title('MAPE')
#axes[1].set_xlabel('Number of error points removed')
axes[1].set_ylabel('MAPE score')
axes[1].set_ylim(2.0, 2.7)
axes[1].legend()
#axes[1].set_yscale("log")

axes[2].bar(x2, y_mse, width, label = "default")
axes[2].bar([x2 + width for x2 in range(len(y1_mse))], y1_mse, width, label = "with tuning")
axes[2].set_title('MSE')
#axes[2].set_xlabel('Number of error points removed')
axes[2].set_ylabel('MSE score multiplied by 10^7')
axes[2].set_ylim(74000000, 80000000)
axes[2].legend()
#axes[2].set_yscale("log")

# Set the overall title
fig.suptitle('Random Forest Regression')
# Show the plot
plt.show()

#%% REMOVE ERRORS WITH XGBR REGRESSOR

t = XGBR_model()

ns = []
xgbr_r2_new_set = []
xgbr_mape_new_set = []
xgbr_mse_new_set = []
X_train, y_train, X_test, y_test, X_val, y_val = setup_x_y()


for i in range(30):
  n = i*5
  ns.append(n)
  t = XGBR_model()
  tt = RemoveHighResidualsXGBR(t, n)
  xgbr_r2_new_set.append(tt[0])
  xgbr_mape_new_set.append(tt[1])
  xgbr_mse_new_set.append(tt[2])

#combine ns and random_r2

xgbr_r2_data = list(zip(xgbr_r2_new_set, ns))
sorted_xgbr_r2_data = sorted(xgbr_r2_data, reverse=True)


xgbr_mape_data = list(zip(xgbr_mape_new_set, ns))
sorted_xgbr_mape_data = sorted(xgbr_mape_data, reverse=False)


xgbr_mse_data = list(zip(xgbr_mse_new_set, ns))
sorted_xgbr_mse_data = sorted(xgbr_mse_data, reverse=False)


print(sorted_xgbr_r2_data[0], sorted_xgbr_mape_data[0], sorted_xgbr_mse_data[0])
## plot bar of normal vs remove 30

t = XGBR_model()
x = ["original model", "remove top 5", "remove top 145"]
y_r2 = [t[0], xgbr_r2_data[1][0], xgbr_r2_data[29][0]]
y_mape = [t[2], xgbr_mape_data[1][0], xgbr_mape_data[29][0]]
y_mse = [t[3], xgbr_mse_data[1][0], xgbr_mse_data[29][0]]

fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(9, 12))

# Plot the data on each subplot and set the title
axes[0].bar(x,y_r2)
axes[0].set_title('R SQUARED')
#axes[0].set_xlabel('Number of error points removed')
axes[0].set_ylabel('R squared score')
axes[0].set_ylim(0.6, 0.8)

#axes[0].set_yscale("log")

axes[1].bar(x, y_mape)
axes[1].set_title('MAPE')
#axes[1].set_xlabel('Number of error points removed')
axes[1].set_ylabel('MAPE score')
axes[1].set_ylim(3.0, 3.3)
#axes[1].set_yscale("log")


axes[2].bar(x, y_mse)
axes[2].set_title('MSE')
#axes[2].set_xlabel('Number of error points removed')
axes[2].set_ylabel('MSE score multiplied by 10^7')
axes[2].set_ylim(80000000, 85000000)
#axes[2].set_yscale("log")

# Set the overall title
fig.suptitle('Extreme Gradient Boosting')

# Show the plot
plt.show()


#%% NOW WITH HYPERPARAMETER TUNING

t = XGBR_model2()


ns = []
xgbr_r2_new_set1 = []
xgbr_mape_new_set1 = []
xgbr_mse_new_set1 = []
X_train, y_train, X_test, y_test, X_val, y_val = setup_x_y()


for i in range(30):
  n = i*5
  ns.append(n)
  t = XGBR_model2()
  tt = RemoveHighResidualsXGBR2(t, n)
  xgbr_r2_new_set1.append(tt[0])
  xgbr_mape_new_set1.append(tt[1])
  xgbr_mse_new_set1.append(tt[2])

#combine ns and random_r2

xgbr_r2_data1 = list(zip(xgbr_r2_new_set1, ns))
sorted_xgbr_r2_data1 = sorted(xgbr_r2_data1, reverse=True)


xgbr_mape_data1 = list(zip(xgbr_mape_new_set1, ns))
sorted_xgbr_mape_data1 = sorted(xgbr_mape_data1, reverse=False)


xgbr_mse_data1 = list(zip(xgbr_mse_new_set1, ns))
sorted_xgbr_mse_data1 = sorted(xgbr_mse_data1, reverse=False)


print(sorted_xgbr_r2_data1[0], sorted_xgbr_mape_data1[0], sorted_xgbr_mse_data1[0])
t1 = XGBR_model2()
x1 = ["original model", "remove top 45", "remove top 70"]
y1_r2 = [t1[0], xgbr_r2_data[9][0], xgbr_r2_data[14][0]]
y1_mape = [t1[2], xgbr_mape_data[9][0], xgbr_mape_data[14][0]]
y1_mse = [t1[3], xgbr_mse_data[9][0], xgbr_mse_data[14][0]]

fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(9, 12))

# Plot the data on each subplot and set the title
axes[0].bar(x1,y1_r2)
axes[0].set_title('R SQUARED')
#axes[0].set_xlabel('Number of error points removed')
axes[0].set_ylabel('R squared score')
axes[0].set_ylim(0.4, 0.8)

#axes[0].set_yscale("log")

axes[1].bar(x1, y1_mape)
axes[1].set_title('MAPE')
#axes[1].set_xlabel('Number of error points removed')
axes[1].set_ylabel('MAPE score')
axes[1].set_ylim(1.4, 3.3)
#axes[1].set_yscale("log")


axes[2].bar(x1, y1_mse)
axes[2].set_title('MSE')
#axes[2].set_xlabel('Number of error points removed')
axes[2].set_ylabel('MSE score multiplied by 10^7')
axes[2].set_ylim(80000000, 160000000)
#axes[2].set_yscale("log")

# Set the overall title
fig.suptitle('Extreme Gradient Boosting with hyperparameter tuning')

# Show the plot
plt.show()

#%% PLOT BOTH


t = XGBR_model()
t1 = XGBR_model2()

x2 = ["original model", "remove top 5", "remove top 45", "remove top 70", "remove top 145"]
y_r2 = [t[0], xgbr_r2_data[1][0], xgbr_r2_data[9][0], xgbr_r2_data[14][0], xgbr_r2_data[29][0]]
y_mape = [t[2], xgbr_mape_data[1][0], xgbr_mape_data[9][0], xgbr_mape_data[14][0], xgbr_mape_data[29][0]]
y_mse = [t[3], xgbr_mse_data[1][0], xgbr_mse_data[9][0], xgbr_mse_data[14][0], xgbr_mse_data[29][0]]
y1_r2 = [t1[0], xgbr_r2_data1[1][0], xgbr_r2_data1[9][0], xgbr_r2_data1[14][0],xgbr_r2_data1[29][0]]
y1_mape = [t1[2], xgbr_mape_data1[1][0], xgbr_mape_data1[9][0], xgbr_mape_data1[14][0], xgbr_mape_data1[29][0]]
y1_mse = [t1[3], xgbr_mse_data1[1][0], xgbr_mse_data1[9][0], xgbr_mse_data1[14][0], xgbr_mse_data1[29][0]]



fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(8, 12))
width = 0.3

# Plot the data on each subplot and set the title
axes[0].bar(x2,y_r2, width, label = "default")
axes[0].bar([x2 + width for x2 in range(len(y1_r2))],y1_r2, width, label = "with tuning")
axes[0].set_title('R SQUARED')
#axes[0].set_xlabel('Number of error points removed')
axes[0].set_ylabel('R squared score')
axes[0].set_ylim(0.4, 0.85)
axes[0].legend()

axes[1].bar(x2, y_mape, width, label = "default")
axes[1].bar([x2 + width for x2 in range(len(y1_mape))], y1_mape, width, label = "with tuning")
axes[1].set_title('MAPE')
#axes[1].set_xlabel('Number of error points removed')
axes[1].set_ylabel('MAPE score')
axes[1].set_ylim(1.3, 3.7)
axes[1].legend()
#axes[1].set_yscale("log")

axes[2].bar(x2, y_mse, width, label = "default")
axes[2].bar([x2 + width for x2 in range(len(y1_mse))], y1_mse, width, label = "with tuning")
axes[2].set_title('MSE')
#axes[2].set_xlabel('Number of error points removed')
axes[2].set_ylabel('MSE score multiplied by 10^7')
axes[2].set_ylim(70000000, 170000000)
axes[2].legend()
#axes[2].set_yscale("log")

# Set the overall title
fig.suptitle('Extreme Gradient Boosting')
# Show the plot
plt.show()
