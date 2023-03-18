# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 20:10:28 2023

@author: safij
"""


#%% IMPORTS
import random
import matplotlib.pyplot as plt
import numpy as np
import csv
import pandas as pd
import seaborn as sn
#from google.colab import drive
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error as mse
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import classification_report
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_absolute_percentage_error
from scipy.stats import randint as sp_randint
import numpy as np
from sklearn.preprocessing import MinMaxScaler
#drive.mount('/content/drive')


car_data = 'Documents\INTRO_TO_AI\car_price_prediction.csv'
#car_data = pd.read_csv(car_data, header = 0, skiprows=0, low_memory=False)

df = pd.read_csv(car_data, header = 0, skiprows=0, low_memory=False)
#%% Processing Functions 

def FormatData(df):
    df.replace('-',np.nan, inplace=True) #replaces '-' with Nan values
    df['Levy'] = df['Levy'].astype('float64')
    df['Mileage'] = df['Mileage'].str.extract('(\d+)').astype('int64') # This is going to remove the 'km' in the mileage
    #car_data['Mileage'] = car_data['Mileage'].astype('int64')
    df['Leather interior'] = df['Leather interior'].replace({'Yes': True, 'No': False})#replace 'Leather interior yes/no with T/F
    df['Wheel'] = df['Wheel'].replace({'Right-hand drive': True, 'Left wheel': False})#replace 'Wheel' Right-hand drive/left wheel with T/F
    df['Turbo'] = df['Engine volume'].str.contains('Turbo') #place turbo in separate new column with T/F.
    df['Engine volume'] = df['Engine volume'].str.extract(r'(\d+\.\d+|\d+)').astype(float) # remove turbo from engine type, 
    df['Engine volume'] = df['Engine volume'].astype('float64')
    df['Doors'].replace({'04-May':4, '02-Mar':2, '>5':5}, inplace=True) #replace doors dates with 2,4,5
    df = df.drop('ID', axis=1) #drops a column with not relevant information
    return(df)

def Removeoutliars(df):
  high = 250000
  low = 100
  df = df[df['Price'] <= high]
  df = df[df['Price'] >= low]
  return df

def OneHotEncode(df):
  df = pd.get_dummies(df, columns = ['Manufacturer','Model','Category','Fuel type','Gear box type','Drive wheels','Color','Leather interior','Wheel','Turbo']) #one hot encoding the data
  return df


#-#-#-#-#- |LEVY STUFF| -#-#-#-#-#
def LevyMedian(X_train,X_val,X_test):
  X_train = pd.DataFrame(X_train)
  X_val = pd.DataFrame(X_val)
  X_test = pd.DataFrame(X_test)
  medianval = X_train[0].median()
  X_train[0] = X_train[0].fillna(X_train[0].median())
  X_val[0] = X_val[0].fillna(medianval)
  X_test[0] = X_test[0].fillna(medianval)
  return X_train,X_val, X_test


def TrainTestVal(df):
  X = df.drop('Price',axis=1)
  y = df['Price']
  X = np.array(X)
  y = np.array(y)
  X_train, X_testval, y_train, y_testval = train_test_split(X, y, test_size=0.5, random_state = 42)
  XB = np.array(X_testval)
  yB = np.array(y_testval)
  X_val, X_test, y_val, y_test = train_test_split(XB, yB, test_size=0.5, random_state = 42)
  return X_train, y_train, X_test, y_test, X_val, y_val


def MinMax(X_train,X_val,X_test):
  X_train = pd.DataFrame(X_train)
  X_val = pd.DataFrame(X_val)
  X_test = pd.DataFrame(X_test)
  sc = MinMaxScaler()
  sc.fit(X_train)
  sc.transform(X_train)
  sc.transform(X_val)
  sc.transform(X_test)
  return X_train,X_val, X_test

def setup_x_y():
    df = pd.read_csv(car_data, header = 0, skiprows=0, low_memory=False)
    a = FormatData(df)
    b = OneHotEncode(a)
    c = Removeoutliars(b)
    X_train, y_train, X_test, y_test, X_val, y_val = TrainTestVal(c)
    X_train,X_val,X_test = LevyMedian(X_train,X_val,X_test)
    X_train,X_val, X_test = MinMax(X_train,X_val,X_test)
    return X_train, y_train, X_test, y_test, X_val, y_val

def RandomForestRegressionModel4():
        X_train, y_train, X_test, y_test, X_val, y_val = setup_x_y()
        random = RandomForestRegressor()
        random.fit(X_train, y_train)
        y_pred = random.predict(X_test)
        random_r2 = r2_score(y_test, y_pred)
        random_mape = mean_absolute_percentage_error(y_test, y_pred)
        random_mse = mse(y_test, y_pred)
        return random_r2, y_pred, random_mape, random_mse
    
def RandomForestRegressionModel5():
        X_train, y_train, X_test, y_test, X_val, y_val = setup_x_y()
        random = RandomForestRegressor(n_estimators=138, max_depth = 645, min_samples_split=7, min_samples_leaf=2)
        random.fit(X_train, y_train)
        y_pred = random.predict(X_test)
        random_r2 = r2_score(y_test, y_pred)
        random_mape = mean_absolute_percentage_error(y_test, y_pred)
        random_mse = mse(y_test, y_pred)
        return random_r2, y_pred, random_mape, random_mse


def RemoveHighResidualsRandom3(t, n):
  X_train, y_train, X_test, y_test, X_val, y_val = setup_x_y()
  y_pred = t[1]
  res = y_test - y_pred
  sq_res = res**2
  sorted_loss_indices = np.argsort(sq_res)[::-1][:n]
  X_train_new =  X_train.drop(X_train.index[sorted_loss_indices])
  y_train_new = np.delete(y_train, sorted_loss_indices)
  random_model_new = RandomForestRegressor()
  random_model_new.fit(X_train_new, y_train_new)
  y_pred_new = random_model_new.predict(X_test)
  random_r2_new = r2_score(y_test, y_pred_new)
  random_mape_new = mean_absolute_percentage_error(y_test, y_pred_new)
  random_mse_new = mse(y_test, y_pred_new)
  return random_r2_new, random_mape_new, random_mse_new


def RemoveHighResidualsRandom4(t, n):
  X_train, y_train, X_test, y_test, X_val, y_val = setup_x_y()
  y_pred = t[1]
  res = y_test - y_pred
  sq_res = res**2
  sorted_loss_indices = np.argsort(sq_res)[::-1][:n]
  X_train_new =  X_train.drop(X_train.index[sorted_loss_indices])
  y_train_new = np.delete(y_train, sorted_loss_indices)
  random_model_new = RandomForestRegressor(n_estimators=138, max_depth = 645, min_samples_split=7, min_samples_leaf=2)
  random_model_new.fit(X_train_new, y_train_new)
  y_pred_new = random_model_new.predict(X_test)
  random_r2_new = r2_score(y_test, y_pred_new)
  random_mape_new = mean_absolute_percentage_error(y_test, y_pred_new)
  random_mse_new = mse(y_test, y_pred_new)
  return random_r2_new, random_mape_new, random_mse_new


t = RandomForestRegressionModel4()
t1 = RandomForestRegressionModel5()

#top40
r2top40 =  RemoveHighResidualsRandom3(t, 40)[0]
r2top40_1 =  RemoveHighResidualsRandom4(t, 40)[0]
mapetop40 =  RemoveHighResidualsRandom3(t, 40)[1]
mapetop40_1 =  RemoveHighResidualsRandom4(t, 40)[1]
msetop40 =  RemoveHighResidualsRandom3(t, 40)[2]
msetop40_1 =  RemoveHighResidualsRandom4(t, 40)[2]

#top95
r2top95 =  RemoveHighResidualsRandom3(t, 95)[0]
r2top95_1 =  RemoveHighResidualsRandom4(t, 95)[0]
mapetop95 =  RemoveHighResidualsRandom3(t, 95)[1]
mapetop95_1 =  RemoveHighResidualsRandom4(t, 95)[1]
msetop95 =  RemoveHighResidualsRandom3(t, 95)[2]
msetop95_1 =  RemoveHighResidualsRandom4(t, 95)[2]


#top140
r2top140 =  RemoveHighResidualsRandom3(t, 140)[0]
r2top140_1 =  RemoveHighResidualsRandom4(t, 140)[0]
mapetop140 =  RemoveHighResidualsRandom3(t, 140)[1]
mapetop140_1 =  RemoveHighResidualsRandom4(t, 140)[1]
msetop140 =  RemoveHighResidualsRandom3(t, 140)[2]
msetop140_1 =  RemoveHighResidualsRandom4(t, 140)[2]


x2 = ["original model", "remove top 40", "remove top 95", "remove top 140"]
y_r2 = [t[0], r2top40, r2top95, r2top140]
y_mape = [t[2], mapetop40, mapetop95, mapetop140]
y_mse = [t[3], msetop40, msetop95, msetop140]
y1_r2 = [t1[0], r2top40_1, r2top95_1, r2top140_1]
y1_mape = [t1[2], mapetop40_1, mapetop95_1, mapetop140_1]
y1_mse = [t1[3], msetop40_1, msetop95_1, msetop140_1]


fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(8, 12))
width = 0.3

# Plot the data on each subplot and set the title
axes[0].bar(x2,y_r2, width, label = "default")
axes[0].bar([x2 + width for x2 in range(len(y1_r2))],y1_r2, width, label = "with tuning")
axes[0].set_title('R SQUARED')
#axes[0].set_xlabel('Number of error points removed')
axes[0].set_ylabel('R squared score')
axes[0].set_ylim(0.68, 0.76)
axes[0].legend()

axes[1].bar(x2, y_mape, width, label = "default")
axes[1].bar([x2 + width for x2 in range(len(y1_mape))], y1_mape, width, label = "with tuning")
axes[1].set_title('MAPE')
#axes[1].set_xlabel('Number of error points removed')
axes[1].set_ylabel('MAPE score')
axes[1].set_ylim(1.6, 2.5)
axes[1].legend()
#axes[1].set_yscale("log")

axes[2].bar(x2, y_mse, width, label = "default")
axes[2].bar([x2 + width for x2 in range(len(y1_mse))], y1_mse, width, label = "with tuning")
axes[2].set_title('MSE')
#axes[2].set_xlabel('Number of error points removed')
axes[2].set_ylabel('MSE score multiplied by 10^7')
axes[2].set_ylim(81000000, 88000000)
axes[2].legend()
#axes[2].set_yscale("log")

# Set the overall title
fig.suptitle('Random Forest Regression')
# Show the plot
plt.show()


#%%

def XGBR_model3():
    X_train, y_train, X_test, y_test, X_val, y_val = setup_x_y()
    xgbr_model =XGBRegressor()
    xgbr_model.fit(X_train, y_train)
    y_pred = xgbr_model.predict(X_test)
    xgbr_r2 = r2_score(y_test, y_pred)
    xgbr_mape = mean_absolute_percentage_error(y_test, y_pred)
    xgbr_mse = mse(y_test, y_pred)
    return xgbr_r2, y_pred, xgbr_mape, xgbr_mse



def XGBR_model4():
    X_train, y_train, X_test, y_test, X_val, y_val = setup_x_y()
    xgbr_model =XGBRegressor(max_depth = 63, learning_rate = 0.01, gamma = 0.0, subsample = 1.0, colsample_bytree = 0.998)
    xgbr_model.fit(X_train, y_train)
    y_pred = xgbr_model.predict(X_test)
    xgbr_r2 = r2_score(y_test, y_pred)
    xgbr_mape = mean_absolute_percentage_error(y_test, y_pred)
    xgbr_mse = mse(y_test, y_pred)
    return xgbr_r2, y_pred, xgbr_mape, xgbr_mse


def RemoveHighResidualsXGBR3(t, n):
  X_train, y_train, X_test, y_test, X_val, y_val = setup_x_y()
  y_pred = t[1]
  res = y_test - y_pred
  sq_res = res**2
  sorted_loss_indices = np.argsort(sq_res)[::-1][:n]
  X_train_new =  X_train.drop(X_train.index[sorted_loss_indices])
  y_train_new = np.delete(y_train, sorted_loss_indices)
  xgbr_model_new = XGBRegressor()
  xgbr_model_new.fit(X_train_new, y_train_new)
  y_pred_new = xgbr_model_new.predict(X_test)
  xgbr_r2_new = r2_score(y_test, y_pred_new)
  xgbr_mape_new = mean_absolute_percentage_error(y_test, y_pred_new)
  xgbr_mse_new = mse(y_test, y_pred_new)
  return xgbr_r2_new, xgbr_mape_new, xgbr_mse_new


def RemoveHighResidualsXGBR4(t, n):
  X_train, y_train, X_test, y_test, X_val, y_val = setup_x_y()
  y_pred = t[1]
  res = y_test - y_pred
  sq_res = res**2
  sorted_loss_indices = np.argsort(sq_res)[::-1][:n]
  X_train_new =  X_train.drop(X_train.index[sorted_loss_indices])
  y_train_new = np.delete(y_train, sorted_loss_indices)
  xgbr_model_new = XGBRegressor(max_depth = 63, learning_rate = 0.01, gamma = 0.0, subsample = 1.0, colsample_bytree = 0.998)
  xgbr_model_new.fit(X_train_new, y_train_new)
  y_pred_new = xgbr_model_new.predict(X_test)
  xgbr_r2_new = r2_score(y_test, y_pred_new)
  xgbr_mape_new = mean_absolute_percentage_error(y_test, y_pred_new)
  xgbr_mse_new = mse(y_test, y_pred_new)
  return xgbr_r2_new, xgbr_mape_new, xgbr_mse_new



t = XGBR_model3()
t1 = XGBR_model4()

x2 = ["original model", "remove top 5", "remove top 45", "remove top 70", "remove top 145"]

#top5
r2top5 =  RemoveHighResidualsXGBR3(t, 5)[0]
r2top5_1 =  RemoveHighResidualsXGBR4(t, 5)[0]
mapetop5 =  RemoveHighResidualsXGBR3(t, 5)[1]
mapetop5_1 =  RemoveHighResidualsXGBR4(t, 5)[1]
msetop5 =  RemoveHighResidualsXGBR3(t, 5)[2]
msetop5_1 =  RemoveHighResidualsXGBR4(t, 5)[2]

#top45
r2top45 =  RemoveHighResidualsXGBR3(t, 45)[0]
r2top45_1 =  RemoveHighResidualsXGBR4(t, 45)[0]
mapetop45 =  RemoveHighResidualsXGBR3(t, 45)[1]
mapetop45_1 =  RemoveHighResidualsXGBR4(t, 45)[1]
msetop45 =  RemoveHighResidualsXGBR3(t, 45)[2]
msetop45_1 =  RemoveHighResidualsXGBR4(t, 45)[2]


#top70
r2top70 =  RemoveHighResidualsXGBR3(t, 70)[0]
r2top70_1 =  RemoveHighResidualsXGBR4(t, 70)[0]
mapetop70 =  RemoveHighResidualsXGBR3(t, 70)[1]
mapetop70_1 =  RemoveHighResidualsXGBR4(t, 70)[1]
msetop70 =  RemoveHighResidualsXGBR3(t, 70)[2]
msetop70_1 =  RemoveHighResidualsXGBR4(t, 70)[2]

#top145
r2top145 =  RemoveHighResidualsXGBR3(t, 145)[0]
r2top145_1 =  RemoveHighResidualsXGBR4(t, 145)[0]
mapetop145 =  RemoveHighResidualsXGBR3(t, 145)[1]
mapetop145_1 =  RemoveHighResidualsXGBR4(t, 145)[1]
msetop145 =  RemoveHighResidualsXGBR3(t, 145)[2]
msetop145_1 =  RemoveHighResidualsXGBR4(t, 145)[2]


x2 = ["original model", "remove top 5", "remove top 45", "remove top 70", "remove top 145"]
y_r2 = [t[0], r2top5, r2top45, r2top70, r2top145]
y_mape = [t[2], mapetop5, mapetop45, mapetop70, mapetop145]
y_mse = [t[3], msetop5, msetop45, msetop70, msetop145]
y1_r2 = [t1[0], r2top5_1, r2top45_1, r2top70_1, r2top145_1]
y1_mape = [t1[2], mapetop5_1, mapetop45_1, mapetop70_1, mapetop145_1]
y1_mse = [t1[3], msetop5_1, msetop45_1, msetop70_1, msetop145_1]



fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(8, 12))
width = 0.3

# Plot the data on each subplot and set the title
axes[0].bar(x2,y_r2, width, label = "default")
axes[0].bar([x2 + width for x2 in range(len(y1_r2))],y1_r2, width, label = "with tuning")
axes[0].set_title('R SQUARED')
#axes[0].set_xlabel('Number of error points removed')
axes[0].set_ylabel('R squared score')
axes[0].set_ylim(0.4, 0.73)
axes[0].legend()

axes[1].bar(x2, y_mape, width, label = "default")
axes[1].bar([x2 + width for x2 in range(len(y1_mape))], y1_mape, width, label = "with tuning")
axes[1].set_title('MAPE')
#axes[1].set_xlabel('Number of error points removed')
axes[1].set_ylabel('MAPE score')
axes[1].set_ylim(1.0, 3.4)
axes[1].legend()
#axes[1].set_yscale("log")

axes[2].bar(x2, y_mse, width, label = "default")
axes[2].bar([x2 + width for x2 in range(len(y1_mse))], y1_mse, width, label = "with tuning")
axes[2].set_title('MSE')
#axes[2].set_xlabel('Number of error points removed')
axes[2].set_ylabel('MSE score multiplied by 10^7')
axes[2].set_ylim(85000000, 173000000)
axes[2].legend()
#axes[2].set_yscale("log")

# Set the overall title
fig.suptitle('Extreme Gradient Boosting')
# Show the plot
plt.show()