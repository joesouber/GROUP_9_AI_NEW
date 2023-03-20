#%% IMPORTS

import random
import matplotlib.pyplot as plt
import numpy as np
import csv
import pandas as pd
import seaborn as sn
from google.colab import drive
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, mean_absolute_percentage_error, r2_score, classification_report

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
from scipy.stats import randint as sp_randint
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score

df = pd.read_csv('car_price_prediction', header = 0, skiprows=0, low_memory=False)

# Functions 
def LinearRegressionModel(X_train, y_train, X_test, y_test):
    # create a linear regression model object
    lr = LinearRegression()

    # fit the model to the training data
    lr.fit(X_train, y_train)

    # predict on the test data
    y_pred = lr.predict(X_test)

    # evaluate the performance of the model on the test data
    mpe_score = mean_absolute_percentage_error(y_val, y_pred)

    print('We have predicted the price with an accuracy of', mpe_score,'on the test set')


    #return r2 and rmse_scores and y_pred
    return mpe_score





def RandomForestRegressionModel(X_train, y_train, X_test, y_test):

        # create a random forest regression model object
        random = RandomForestRegressor()

        # fit the model to the training data
        random.fit(X_train, y_train)

        # predict on the test data
        y_pred = random.predict(X_test)

        # evaluate the performance of the model on the test data
        mape = mean_absolute_percentage_error(y_test, y_pred)

        print('We have predicted the price with an accuracy of',mape,'on the test set')

        return mape


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
    mape = mean_absolute_percentage_error(y_test, y_pred)
    print('We have predicted the price with an accuracy of',mape,'on the test set')
    # perform cross validation
    #scores = cross_val_score(ridge, X_train, y_train, scoring='neg_mean_squared_error', cv=5)
    #rmse_scores = np.sqrt(-scores)
    #print("Ridge regression model mean cross validation: ", rmse_scores.mean())
    #return r2 and rmse_scores and y_pred
    return mape, optimal_alpha


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




#%%
# build a extra tree regressor model
def XGBRmodel(X_train, y_train, X_test, y_test):
    xgbr_model =XGBRegressor()
    xgbr_model.fit(X_train, y_train)
    y_pred = xgbr_model.predict(X_test)
    xgbr_r2 = r2_score(y_test, y_pred)
    xgbr_mape = mean_absolute_percentage_error(y_test, y_pred)
    xgbr_mse = mse(y_test, y_pred)
    return xgbr_r2, xgbr_mape, xgbr_mse, xgbr_model
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


# Caclulating spearmen coefficients 
correlations = {}
for feature in df.columns:
     if feature != 'Price':
         corr = df[feature].corr(df['Price'], method='spearman')
         correlations[feature] = corr

 # Convert correlations dictionary to DataFrame
correlations_df = pd.DataFrame.from_dict(correlations, orient='index', columns=['Spearman Correlation'])

 # Now find all the features with coefficient less than zero
# correlations_df = correlations_df[correlations_df['Spearman Correlation'] < 0]
# # Create heatmap

sn.heatmap(correlations_df, annot=True, cmap='coolwarm')
 #%%
 #Dropping features


# dropping features = 0
 df = df.drop(['Color', 'Drive wheels', 'ID', 'Doors', 'Category', 'Cylinders', 'Airbags', 'Manufacturer', 'Model', ''], axis=1)
#%%
df  = Removeoutliars(OneHotEncode(FormatData(df)))

X_train, y_train, X_test, y_test, X_val, y_val = TrainTestVal(df)
X_train, X_val = LevyMedian(X_train, X_val)
X_train, X_val = MinMax(X_train, X_val)
#%%
LinearRegressionModel(X_train, y_train, X_val, y_val)
 non dropped
# #%%
RandomForestRegressionModel(X_train, y_train, X_val, y_val)

#%%
XGBRmodel(X_train, y_train, X_val, y_val)

RidgeRegressionModel(X_train, y_train, X_val, y_val)

# LASSO REDUCTION
#df  = Removeoutliars(OneHotEncode(FormatData(df)))

X_train, y_train, X_test, y_test, X_val, y_val = TrainTestVal(df)
X_train, X_val = LevyMedian(X_train, X_val)
X_train, X_val = MinMax(X_train, X_val)
#%% LASSO REGRESSION
# build a lasso regression around x and y and find the coefficients of each feature
lasso = Lasso(alpha=2.7)
lasso.fit(X_train, y_train)
# print the coefficients of each feature
k = lasso.coef_

#%%
# now drop the features with coefficients of zero from the dataset
X_train = X_train.drop(X_train.columns[np.where(k == 0)], axis=1)
X_val = X_val.drop(X_val.columns[np.where(k == 0)], axis=1)
