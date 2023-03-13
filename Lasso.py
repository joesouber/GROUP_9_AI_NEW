#%% IMPORTS

import random
import matplotlib.pyplot as plt
import numpy as np
import csv
import pandas as pd
import seaborn as sn
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Lasso
import scipy.optimize
#from google.colab import drive
#import elastic net
from sklearn.linear_model import ElasticNet
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error as mse
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasRegressor
#import conv2d, flatten, maxpooling2d from keras.layers
from keras.layers import Conv2D, Flatten, MaxPooling2D
import time
#drive.mount('/content/drive')

# %% IMPORT DATA

df = pd.read_csv('car_price_prediction.csv', header = 0, skiprows = 0, low_memory = 'false')
df  = Removeoutliars(OneHotEncode(FormatData(df)))

X_train, y_train, X_test, y_test, X_val, y_val = TrainTestVal(df)
X_train, X_val = LevyMedian(X_train, X_val)
X_train, X_val = MinMax(X_train, X_val)
#%% LASSO REGRESSION
# build a lasso regression around x and y and find the coefficients of each feature
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)
# print the coefficients of each feature
k = lasso.coef_

#%%
# now drop the features with coefficients of zero from the dataset
X_train = X_train.drop(X_train.columns[np.where(k == 0)], axis=1)
X_val = X_val.drop(X_val.columns[np.where(k == 0)], axis=1)

#%%
# now we can run all the models below on the new dataset

#RandomForestRegressionModel(X_train, y_train, X_val, y_val) 0.743022
#GradientBoostingRegressionModel(X_train, y_train, X_val, y_val) 0.57603
#ExtremeGradientBoostingRegressionModel(X_train, y_train, X_val, y_val) 0.42457
RidgeRegressionModel(X_train, y_train, X_val, y_val)
LinearRegressionModel(X_train, y_train, X_val, y_val) 
LassoRegressionModel(X_train, y_train, X_val, y_val) 0.42457
ElasticNetRegressionModel(X_train, y_train, X_val, y_val)



#%% FUNCTIONS
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

def FeatureReduction(df):
  df = df.drop(['Levy'], axis = 1)
  df = df.drop(['Color'], axis = 1)
  df = df.drop(['Drive wheels'], axis = 1)
  return df

def Removeoutliars(df):
  high = 250000
  low = 100
  df = df[df['Price'] <= high]
  df = df[df['Price'] >= low]
  return df

def OneHotEncode(df):
  df = pd.get_dummies(df, columns = ['Manufacturer','Model','Category','Fuel type','Gear box type', 'Drive wheels', 'Leather interior','Turbo', 'Color']) #one hot encoding the data
  return df

#-#-#-#-#- |LEVY STUFF| -#-#-#-#-#
def LevyMedian(X_train,X_val):
  X_train = pd.DataFrame(X_train)
  X_val = pd.DataFrame(X_val)
  medianval = X_train[0].median()
  X_train[0] = X_train[0].fillna(X_train[0].median())
  X_val[0] = X_val[0].fillna(medianval)
  return X_train,X_val

def LevyMean(X_train,X_val):
  X_train = pd.DataFrame(X_train)
  X_val = pd.DataFrame(X_val)
  meanval = X_train[0].mean()
  X_train[0] = X_train[0].fillna(X_train[0].mean())
  X_val[0] = X_val[0].fillna(meanval)
  return X_train,X_val

def DtrOnLevy(X_train_F,X_val_F):
    nan_df = X_train_F[X_train['Levy'].isnull()]   # get rows with NaN values in 'Levy' column
    non_nan_df = X_train_F.dropna(subset=['Levy'])   # get rows without NaN values in 'Levy' column
    X_train, X_test, y_train, y_test = train_test_split(non_nan_df.drop('Levy', axis=1), non_nan_df['Levy'], test_size=0.33, random_state=42)
    reg = DecisionTreeRegressor(random_state=42)
    reg.fit(X_train, y_train)
    nan_pred = reg.predict(nan_df.drop('Levy', axis=1))
    nan_df['Levy'] = nan_pred
    return pd.concat([non_nan_df, nan_df])

def DtrOnLevyBoth(train_df, test_df):
    train_df = pd.DataFrame(train_df)
    test_df = pd.DataFrame(test_df)
    nan_train_df = train_df[train_df[0].isnull()]  # get rows with NaN values in 'Levy' column from train dataframe
    non_nan_train_df = train_df.dropna(subset=[0])  # get rows without NaN values in 'Levy' column from train dataframe
    X_train, X_test, y_train, y_test = train_test_split(non_nan_train_df.drop(0, axis=1), non_nan_train_df[0], test_size=0.33, random_state=42)
    reg = DecisionTreeRegressor(random_state=42)
    reg.fit(X_train, y_train)
    nan_pred = reg.predict(nan_train_df.drop(0, axis=1))
    nan_train_df[0] = nan_pred
    nan_test_df = test_df[test_df[0].isnull()]  # get rows with NaN values in 'Levy' column from test dataframe
    nan_test_pred = reg.predict(nan_test_df.drop(0, axis=1))
    nan_test_df[0] = nan_test_pred
    train_df = pd.concat([non_nan_train_df, nan_train_df])
    test_df.update(nan_test_df)
    return train_df, test_df



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


def MinMax(X_train,X_val):
  X_train = pd.DataFrame(X_train)
  X_val = pd.DataFrame(X_val)
  sc = MinMaxScaler()
  sc.fit(X_train)
  sc.transform(X_train)
  sc.transform(X_val)
  return X_train,X_val

# %%

# Create a Linear Regression Model
def LinearRegressionModel(X_train, y_train, X_val, y_val):
        # create a linear regression model object
        linear = LinearRegression()
        # fit the model to the training data
        linear.fit(X_train, y_train)
        # predict on the test data
        y_pred = linear.predict(X_val)
        # evaluate the performance of the model on the test data
        r2 = r2_score(y_val, y_pred)
        print('We have predicted the price with an accuracy of',r2,'on the val set')
        #return r2 and rmse_scores and y_pred
        return r2




def RandomForestRegressionModel(X_train, y_train, X_val, y_val):

        # create a random forest regression model object
        #random = RandomForestRegressor()
        random = RandomForestRegressor(max_depth = 220,max_features = None, min_samples_leaf =1, min_samples_split=4,n_estimators=100)   #||SPECIFIC FOR PIPELINE ONE||
        # fit the model to the training data
        random.fit(X_train, y_train)
        # predict on the test data
        y_pred = random.predict(X_val)
        # evaluate the performance of the model on the test data
        r2 = r2_score(y_val, y_pred)
        print('We have predicted the price with an accuracy of',r2,'on the val set')
        #return r2 and rmse_scores and y_pred
        return r2

def RidgeRegressionModel(X_train, y_train, X_test, y_test):
    # find an optimal alpha
    alpha = []
    r2 = []
    step = 0.1
    values = [i for i in range(int(0/step), int(10/step))]
    for value in values:
        alpha.append(value*step)
        ridge_model = Ridge(alpha=value*step)
        ridge_model.fit(X_train, y_train)
        r2.append(ridge_model.score(X_test, y_test))
    optimal_alpha = alpha[r2.index(max(r2))]
    # create a ridge regression model object
    ridge = Ridge(alpha=optimal_alpha)
    # fit the model to the training data
    ridge.fit(X_train, y_train)
    # predict on the test data
    y_pred = ridge.predict(X_test)
    # evaluate the performance of the model on the test data
    r2 = r2_score(y_test, y_pred)
    print('We have predicted the price with an accuracy of',r2,'on the test set')
    # perform cross validation
    #scores = cross_val_score(ridge, X_train, y_train, scoring='neg_mean_squared_error', cv=5)
    #rmse_scores = np.sqrt(-scores)
    #print("Ridge regression model mean cross validation: ", rmse_scores.mean())
    #return r2 and rmse_scores and y_pred
    return r2

def ElasticNetRegressionModel(X_train, y_train, X_test, y_test):

    alpha = []
    r2 = []

    step = 0.1
    values = [i for i in range(int(0.0001/step), int(10/step))]
    for value in values:
        alpha.append(value*step)
        elastic_model = ElasticNet(alpha=value*step)
        elastic_model.fit(X_train, y_train)
        r2.append(elastic_model.score(X_test, y_test))

    optimal_alpha = alpha[r2.index(max(r2))]

    # create a elastic net regression model object
    elastic = ElasticNet(alpha=optimal_alpha)

    # fit the model to the training data
    elastic.fit(X_train, y_train)

    # predict on the test data
    y_pred = elastic.predict(X_test)

    # evaluate the performance of the model on the test data
    r2 = r2_score(y_test, y_pred)

    print('We have predicted the price with an accuracy of',r2,'on the test set')

    # perform cross validation
    scores = cross_val_score(elastic, X_train, y_train, scoring='neg_mean_squared_error', cv=5)
    rmse_scores = np.sqrt(-scores)
    print("Elastic net regression model mean cross validation: ", rmse_scores.mean())

    #return r2 and rmse_scores and y_pred
    return r2, rmse_scores, y_pred

def LassoRegressionModel(X_train, y_train, X_test, y_test):

    # optimizing the value of alpha
    alpha = []
    r2 = []

    step = 0.1
    values = [i for i in range(int(0.0001/step), int(10/step))]

    for value in values:

        alpha.append(value*step)
        lasso_model = Lasso(alpha=value*step)
        lasso_model.fit(X_train, y_train)
        r2.append(lasso_model.score(X_test, y_test))

    optimal_alpha = alpha[r2.index(max(r2))]
    print('The optimal value of alpha is', optimal_alpha)

    # Building a lasso model
    lasso_model = Lasso(optimal_alpha)
    lasso_model.fit(X_train, y_train)
    # producing r2 score
    lasso_score = lasso_model.score(X_test, y_test)
    print('Lasso model produces an accuracy of', lasso_score)
    #cross validation
    scores = cross_val_score(lasso_model, X_train, y_train, scoring='neg_mean_squared_error', cv=5)
    lasso_rmse_scores = np.sqrt(-scores)
    print(lasso_rmse_scores)
    print("Lasso regression model mean cross validation :", lasso_rmse_scores.mean())

    return lasso_score, lasso_rmse_scores

def PolynomialRegressionModel(X_train, y_train, X_test, y_test):

    # optimizing the degree of polynomial regression
    degree = []
    r2 = []

    for value in range(1, 10):
        degree.append(value)
        poly = PolynomialFeatures(degree=value)
        X_train_poly = poly.fit_transform(X_train)
        poly.fit(X_train_poly, y_train)
        lin2 = LinearRegression()
        lin2.fit(X_train_poly, y_train)
        r2.append(lin2.score(poly.fit_transform(X_test), y_test))

    optimal_degree = degree[r2.index(max(r2))]

    # Building a polynomial regression model
    poly = PolynomialFeatures(degree=optimal_degree)
    X_train_poly = poly.fit_transform(X_train)

    poly.fit(X_train_poly, y_train)
    lin2 = LinearRegression()
    lin2.fit(X_train_poly, y_train)
    # create a prediction
    y_pred = lin2.predict(poly.fit_transform(X_test))
    # producing r2 score
    poly_score = lin2.score(poly.fit_transform(X_test), y_test)
    print('Polynomial regression model produces an accuracy of', poly_score)
    #cross validation
    scores = cross_val_score(lin2, X_train_poly, y_train, scoring='neg_mean_squared_error', cv=5)
    poly_rmse_scores = np.sqrt(-scores)
    print(poly_rmse_scores)
    print("Polynomial regression model mean cross validation :", poly_rmse_scores.mean())

    return poly_score, poly_rmse_scores, y_pred

def ExtremeGradientBoostingRegressionModel(X_train, y_train, X_test, y_test):

    # Building a gradient boosting model
    gradient_boosting_model = XGBRegressor()
    gradient_boosting_model.fit(X_train, y_train)
    # producing r2 score
    gradient_boosting_score = gradient_boosting_model.score(X_test, y_test)
    print('Extreme Gradient boosting model produces an accuracy of', gradient_boosting_score)
    #cross validation
    scores = cross_val_score(gradient_boosting_model, X_train, y_train, scoring='neg_mean_squared_error', cv=5)
    gradboost_rmse_scores = np.sqrt(-scores)
    print(gradboost_rmse_scores)
    print("Extreme Gradient boosting model mean cross validation :", gradboost_rmse_scores.mean())

    return gradient_boosting_score, gradboost_rmse_scores, y_pred

def GradientBoostingRegressionModel(X_train, y_train, X_test, y_test):

    # Building a gradient boosting model
    gradient_boosting_model = GradientBoostingRegressor()
    gradient_boosting_model.fit(X_train, y_train)
    # producing r2 score
    gradient_boosting_score = gradient_boosting_model.score(X_test, y_test)
    print('Gradient boosting model produces an accuracy of', gradient_boosting_score)
    #cross validation
    scores = cross_val_score(gradient_boosting_model, X_train, y_train, scoring='neg_mean_squared_error', cv=5)
    gradboost_rmse_scores = np.sqrt(-scores)
    print(gradboost_rmse_scores)
    print("Gradient boosting model mean cross validation :", gradboost_rmse_scores.mean())

    return gradient_boosting_score, gradboost_rmse_scores
