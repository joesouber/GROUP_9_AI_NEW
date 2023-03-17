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
# import extra trees regressor
from sklearn.ensemble import ExtraTreesRegressor
#import conv2d, flatten, maxpooling2d from keras.layers
from keras.layers import Conv2D, Flatten, MaxPooling2D
import time
df = pd.read_csv('car_price_prediction.csv', header = 0, skiprows = 0, low_memory = 'false')


# %% PLOT THE CORRELATION CHART
correlations = {}
for feature in df.columns:
    if feature != 'Price':
        corr = df[feature].corr(df['Price'], method='spearman')
        correlations[feature] = corr

# Convert correlations dictionary to DataFrame
correlations_df = pd.DataFrame.from_dict(correlations, orient='index', columns=['Spearman Correlation'])

# Now find all the features with coefficient less than zero
correlations_df = correlations_df[correlations_df['Spearman Correlation'] < 0]
# Create heatmap
sn.heatmap(correlations_df, annot=True, cmap='coolwarm')
#%%
#Drop those features with coefficient less than zero from the dataset
df = df.drop(['Airbags', 'Cylinders', 'Fuel type', 'Wheel', 'Category', 'Manufacturer'], axis=1)
#%%
df  = Removeoutliars(OneHotEncode(FormatData(df)))

X_train, y_train, X_test, y_test, X_val, y_val = TrainTestVal(df)
X_train, X_val = LevyMedian(X_train, X_val)
X_train, X_val = MinMax(X_train, X_val)
#%%
# now run the other functions
LinearRegressionModel(X_train, y_train, X_val, y_val)
# 0.405819 Normal
# 0.405819 Feature Reduction Spearmen Correlation

#%%
RandomForestRegressionModel(X_train, y_train, X_val, y_val)
# 0.741503 Normal
#  0.7409460 Feature Reduction Spearmen Correlation
#%%
GradientBoostingRegressionModel(X_train, y_train, X_val, y_val)
# 0.575701 Normal
# 0.5754290 Feature Reduction Spearmen Correlation
#%%
ExtremeGradientBoostingRegressionModel(X_train, y_train, X_val, y_val)


#%%
RidgeRegressionModel(X_train, y_train, X_val, y_val)
# 0.427921 Normal
# 0.4279215 Feature Reduction Spearmen Correlation
#%%
LassoRegressionModel(X_train, y_train, X_val, y_val)
# 0.424604

#%%
ElasticNetRegressionModel(X_train, y_train, X_val, y_val)
# 0.4092784 Normal

#%%
ExtraTreesRegressorModel(X_train, y_train, X_val, y_val)
# 0.7200097 Normal



























#%% FUNCTIONS
def FormatData(df):
    df.replace('-',np.nan, inplace=True) #replaces '-' with Nan values
    df['Levy'] = df['Levy'].astype('float64')
    df['Mileage'] = df['Mileage'].str.extract('(\d+)').astype('int64') # This is going to remove the 'km' in the mileage
    #car_data['Mileage'] = car_data['Mileage'].astype('int64')
    df['Leather interior'] = df['Leather interior'].replace({'Yes': True, 'No': False})#replace 'Leather interior yes/no with T/F
    #df['Wheel'] = df['Wheel'].replace({'Right-hand drive': True, 'Left wheel': False})#replace 'Wheel' Right-hand drive/left wheel with T/F
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
  df = pd.get_dummies(df, columns = ['Model','Gear box type', 'Drive wheels', 'Leather interior','Turbo', 'Color']) #one hot encoding the data
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

#%% 
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

#%%
# build a extra tree regressor model
def ExtraTreesRegressorModel(X_train, y_train, X_test, y_test):
  
      # Building a extra tree regressor model
      extra_tree_regressor_model = ExtraTreesRegressor()
      extra_tree_regressor_model.fit(X_train, y_train)
      # producing r2 score
      extra_tree_regressor_score = extra_tree_regressor_model.score(X_test, y_test)
      print('Extra tree regressor model produces an accuracy of', extra_tree_regressor_score)
      #cross validation
      scores = cross_val_score(extra_tree_regressor_model, X_train, y_train, scoring='neg_mean_squared_error', cv=5)
      extra_tree_regressor_rmse_scores = np.sqrt(-scores)
      print(extra_tree_regressor_rmse_scores)
      print("Extra tree regressor model mean cross validation :", extra_tree_regressor_rmse_scores.mean())
  
      return extra_tree_regressor_score, extra_tree_regressor_rmse_scores
# %%
