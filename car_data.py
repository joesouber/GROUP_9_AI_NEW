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
#from xgboost import XGBRegressor
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

#%%
# PUT THE CAR_PRICE_PREDICTION FILE IN THE SAME FOLDER AS THE AI FOLDER
#car_data = '/content/drive/MyDrive/AI shared drive/car_price_prediction.csv'
car_data = pd.read_csv('car_price_prediction.csv', header = 0, skiprows=0, low_memory=False)

#CLEANING

car_data.replace('-',np.nan, inplace=True) #replaces '-' with Nan values
car_data['Levy'] = car_data['Levy'].astype('float64')

car_data['Mileage'] = car_data['Mileage'].str.extract('(\d+)').astype('int64') # This is going to remove the 'km' in the mileage
#car_data['Mileage'] = car_data['Mileage'].astype('int64')

car_data['Leather interior'] = car_data['Leather interior'].replace({'Yes': True, 'No': False})#replace 'Leather interior yes/no with T/F

car_data['Wheel'] = car_data['Wheel'].replace({'Right-hand drive': True, 'Left wheel': False})#replace 'Wheel' Right-hand drive/left wheel with T/F

car_data['Turbo'] = car_data['Engine volume'].str.contains('Turbo') #place turbo in separate new column with T/F.

car_data['Engine volume'] = car_data['Engine volume'].str.extract(r'(\d+\.\d+|\d+)').astype(float) # remove turbo from engine type,
car_data['Engine volume'] = car_data['Engine volume'].astype('float64')
car_data['Doors'].replace({'04-May':4, '02-Mar':2, '>5':5}, inplace=True) #replace doors dates with 2,4,5

car_data = car_data.drop('ID', axis=1) #drops a column with not relevant information

car_data = car_data.drop(16983)

car_data
#%%


#make a new variable that is the price in increasing order
car_sorted = car_data.sort_values('Price')

percent = np.linspace(0,100, len(car_sorted['Price']))
plt.plot(car_sorted['Price'], percent)
plt.show()

# do the same above but with only the first third of the data
car_sorted = car_sorted.iloc[:int(len(car_sorted['Price'])/6)]

plt.plot(car_sorted['Price'], percent[0:int(len(car_sorted['Price']))])
plt.show()

# do the exact same with mileage
car_sorted = car_data.sort_values('Mileage')

percent = np.linspace(0,100, len(car_sorted['Mileage']))
plt.plot(car_sorted['Mileage'], percent)
plt.show()

# do the same above but with only the first third of the data
car_sorted = car_sorted.iloc[:int(len(car_sorted['Mileage'])/6)]

plt.plot(car_sorted['Mileage'], percent[0:int(len(car_sorted['Mileage']))])
plt.show()

# do the exact same with levy
car_sorted = car_data.sort_values('Levy')

percent = np.linspace(0,100, len(car_sorted['Levy']))
plt.plot(car_sorted['Levy'], percent)
plt.show()

# do the same above but with only the first third of the data
car_sorted = car_sorted.iloc[:int(len(car_sorted['Levy'])/6)]

plt.plot(car_sorted['Levy'], percent[0:int(len(car_sorted['Levy']))])
plt.show()


# %%
## Removes outliers

def detect_outliers(df, features, threshold=1.5):
    """
    Detects outliers in a DataFrame based on the interquartile range (IQR) method.

    Parameters:
        df (pandas.DataFrame): The DataFrame to search for outliers in.
        features (list): A list of column names to search for outliers in.
        threshold (float): The number of IQRs from the median to consider a value an outlier. Default is 1.5.

    Returns:
        A list of the row indices where outliers were found.
    """
    outlier_indices = []

    for feature in features:
        values = df[feature].values
        q1, q3 = np.percentile(values, [25, 75])
        iqr = q3 - q1
        outlier_step = iqr * threshold
        lower_bound = q1 - outlier_step
        upper_bound = q3 + outlier_step
        outliers = np.where((values < lower_bound) | (values > upper_bound))[0]
        outlier_indices.extend(outliers)

    return outlier_indices

features = ['Price', 'Levy', 'Mileage']
outliers = detect_outliers(car_data,features, 1.5)
car_data_cleaned = car_data.drop(car_data.loc[outliers].index,axis=0)

plt.boxplot(car_data_cleaned['Price'], notch=None, vert=None, patch_artist=None, widths=None)
max(car_data_cleaned['Price'])


car_data_cleaned
#%% REMOVING NAN'S IN LEVY COLUMN

#replace Nan values in levy colum using
car_data_cleaned = pd.get_dummies(car_data_cleaned, columns = ['Manufacturer','Model','Category','Fuel type','Gear box type','Drive wheels','Color','Leather interior','Wheel','Turbo']) #one hot encoding the data

Nanloc = car_data_cleaned['Levy'].isnull() #finding where the nans are true = Nan, false = no Nan
nan_indices = car_data_cleaned[Nanloc].index    #turning into index

df_with_nans = car_data_cleaned.loc[nan_indices]   #splitting dataset up into nan and no nan
df_without_nans = car_data_cleaned.loc[~Nanloc]

X_train, X_test, y_train, y_test = train_test_split(df_without_nans.drop('Levy', axis=1), df_without_nans['Levy'], test_size=0.33, random_state=42) #splitting up dataset in test/train and droppping levy as we're predicting this

# create a decision tree regressor object
reg = DecisionTreeRegressor(random_state=42)

# fit the regressor to the training data
reg.fit(X_train, y_train)

# predict on the test data
y_pred = reg.predict(X_test)

# evaluate the performance of the model on the test data
r2 = r2_score(y_test, y_pred)

print('We have predicted the levy price with an accuracy of',r2,'on the test set')


#Now we get a r2 score of 98.5.3% with a test size of 33%. Use this predictor on the Nan dataset.
df_with_nans = df_with_nans.drop('Levy', axis=1)

Nan_pred = reg.predict(df_with_nans)
df_with_nans['Levy'] = Nan_pred

Cleaned_and_final_data = pd.concat([df_with_nans, df_without_nans])

#%% SPLITTING INTO TEST AND TRAIN

#Cleaned_and_final_data_one_hot = pd.get_dummies(Cleaned_and_final_data, columns = ['Manufacturer','Model','Category','Fuel type','Gear box type','Drive wheels','Color','Leather interior','Wheel','Turbo'])
#hashed out^^ as used above before predicting levy
X = Cleaned_and_final_data.drop('Price',axis=1)
y = Cleaned_and_final_data['Price']
X = np.array(X)
y = np.array(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state = 42)
#%%GRIDSEARCH TO OPTIMISE HYPERPARAMETERS

# Define the range of hyperparameters to test
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_features': ['auto', 'sqrt'],
    'max_depth': [10, 20, 30, None]
}

# Create an instance of Random Forest regressor
rf = RandomForestRegressor()

# Use GridSearchCV to search over the range of hyperparameters
grid_search = GridSearchCV(rf, param_grid, cv=5, n_jobs=-1)

# Fit the GridSearchCV object with your training data and labels
grid_search.fit(X_train, y_train)

# Print out the best hyperparameters and best score
print(f"Best hyperparameters: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_:.4f}")

# Use the best hyperparameters to fit the model on the entire training set
rf_best = RandomForestRegressor(**grid_search.best_params_)
rf_best.fit(X_train, y_train)

# Make predictions on the test set
#y_pred = rf_best.predict(X_test)



#%%SUPERVISED LEARNING
#Linear regression model
# create a linear regression model function

def LinearRegressionModel(X_train, y_train, X_test, y_test):
    # create a linear regression model object
    lr = LinearRegression()

    # fit the model to the training data
    lr.fit(X_train, y_train)

    # predict on the test data
    y_pred = lr.predict(X_test)

    # evaluate the performance of the model on the test data
    r2 = r2_score(y_test, y_pred)

    print('We have predicted the price with an accuracy of',r2,'on the test set')

    # perform cross validation
    scores = cross_val_score(lr, X_train, y_train, scoring='neg_mean_squared_error', cv=5)
    rmse_scores = np.sqrt(-scores)
    print("Linear regression model mean cross validation: ", rmse_scores.mean())

    #return r2 and rmse_scores and y_pred
    return r2, rmse_scores, y_pred


# linear_model = LinearRegression()
# linear_model.fit(X_train, y_train)
# linear_regression_score = linear_model.score(X_test, y_test)
# print('Linear regression model produces an accuracy of', linear_regression_score)
# #cross validation
# scores = cross_val_score(linear_model, X_train, y_train, scoring='neg_mean_squared_error', cv=5)
# linear_rmse_scores = np.sqrt(-scores)
# print(linear_rmse_scores)
# print("Linear regression model mean cross validation: ", linear_rmse_scores.mean())

#%%
# Ridge Regression Model
# Need to find the right value of the hyperparameter alpha
# finding the optimal value of alpha by plotting the r2 score against alpha
# alpha = []
# r2 = []

# step = 0.1
# values = [i for i in range(int(0/step), int(10/step))]

# for value in values:
#     print(value * step)
#     alpha.append(value*step)
#     ridge_model = Ridge(alpha=value*step)
#     ridge_model.fit(X_train, y_train)
#     r2.append(ridge_model.score(X_test, y_test))

# optimal_alpha = alpha[r2.index(max(r2))]
# print('The optimal value of alpha is', optimal_alpha)

# create a ridge regression model function
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
    scores = cross_val_score(ridge, X_train, y_train, scoring='neg_mean_squared_error', cv=5)
    rmse_scores = np.sqrt(-scores)
    print("Ridge regression model mean cross validation: ", rmse_scores.mean())


    #return r2 and rmse_scores and y_pred
    return r2, rmse_scores, y_pred

## residual score shows points with the highest loss
## testing the function on the linear model, seems to work

def ResidualScores(X_test, y_test, y_pred):
  residuals = y_test - y_pred
  squared_residuals = residuals**2
  n = 10
  top_indices = np.argsort(squared_residuals)[::-1][:n]
  print(top_indices) # these are the top 10 data points with the highest error.
  top_data_points = X_test[top_indices]
  print(top_data_points)  # then what do i do with this
  plt.scatter(y_pred[top_indices], residuals[top_indices])
  plt.xlabel('Predicted Values')
  plt.ylabel('Residuals')
  plt.title('Residual Plot')
  plt.show()
  return residuals, squared_residuals, top_indices, top_data_points

# just to test
ResidualScores(X_test, y_test, LinearRegressionModel(X_train, y_train, X_test, y_test)[2])

# ridge_regressor = Ridge(optimal_alpha)
# ridge_regressor.fit(X_train, y_train)

# ridge_regressor_score = ridge_regressor.score(X_test, y_test)
# print('Ridge regression model produces an accuracy of', ridge_regressor_score)
# #cross validation
# scores = cross_val_score(ridge_regressor, X_train, y_train, scoring='neg_mean_squared_error', cv=5)
# ridge_rmse_scores = np.sqrt(-scores)
# print(ridge_rmse_scores)
# print("Ridge regression model mean cross validation: ", ridge_rmse_scores.mean())


#%% Logistic Regression Model
# logistic_model = LogisticRegression(random_state=1, max_iter=1000)
# logistic_model.fit(X_train, y_train)
# logistic_regression_score = logistic_model.score(X_test, y_test)
# print('Logistic regression model produces an accuracy of', logistic_regression_score)

#%%
# # Building a elastic net regression model
# alpha = []
# r2 = []

# step = 0.1
# values = [i for i in range(int(0.0001/step), int(10/step))]
# for value in values:
#     alpha.append(value*step)
#     elastic_model = ElasticNet(alpha=value*step)
#     elastic_model.fit(X_train, y_train)
#     r2.append(elastic_model.score(X_test, y_test))

# Building a elastic net regression model function
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

# optimal_alpha = alpha[r2.index(max(r2))]
# print('The optimal value of alpha is', optimal_alpha)

# # Building a elastic net regression model
# elastic_model = ElasticNet(optimal_alpha)
# elastic_model.fit(X_train, y_train)
# # producing r2 score
# elastic_score = elastic_model.score(X_test, y_test)
# print('Elastic net model produces an accuracy of', elastic_score)
# #cross validation
# scores = cross_val_score(elastic_model, X_train, y_train, scoring='neg_mean_squared_error', cv=5)
# elastic_rmse_scores = np.sqrt(-scores)
# print(elastic_rmse_scores)
# print("Elastic net regression model mean cross validation :", elastic_rmse_scores.mean())

#%%
# #Random forest
# random_model = RandomForestRegressor()
# random_model.fit(X_train, y_train)
# y_pred = random_model.predict(X_test)
# random_forest_score = r2_score(y_test,y_pred)
# print('Random forest regression model produces an accuracy of', random_forest_score)
# #cross validation
# scores = cross_val_score(random_model, X_train, y_train, scoring='neg_mean_squared_error', cv=5)
# random_rmse_scores = np.sqrt(-scores)
# print(random_rmse_scores)
# print("Random forest regression model mean cross validation: ", random_rmse_scores.mean())

# Building a random forest regression model function
def RandomForestRegressionModel(X_train, y_train, X_test, y_test):

        # create a random forest regression model object
        random = RandomForestRegressor()

        # fit the model to the training data
        random.fit(X_train, y_train)

        # predict on the test data
        y_pred = random.predict(X_test)

        # evaluate the performance of the model on the test data
        r2 = r2_score(y_test, y_pred)

        print('We have predicted the price with an accuracy of',r2,'on the test set')

        # perform cross validation
        scores = cross_val_score(random, X_train, y_train, scoring='neg_mean_squared_error', cv=5)
        rmse_scores = np.sqrt(-scores)
        print("Random forest regression model mean cross validation: ", rmse_scores.mean())
        #return r2 and rmse_scores and y_pred
        return r2, rmse_scores, y_pred

#%%

# optimizing the degree of polynomial regression
# This doesn't work for some reason in VS code, but it works in Jupyter notebook
# degree = []
# r2 = []

# for value in range(1, 10):
#     degree.append(value)
#     poly = PolynomialFeatures(degree=value)
#     X_train_poly = poly.fit_transform(X_train)
#     poly.fit(X_train_poly, y_train)
#     lin2 = LinearRegression()
#     lin2.fit(X_train_poly, y_train)
#     r2.append(lin2.score(poly.fit_transform(X_test), y_test))

# optimal_degree = degree[r2.index(max(r2))]
# print('The optimal degree of polynomial regression is', optimal_degree)


# Building a polynomial regression model
# poly = PolynomialFeatures(degree=2)
# X_train_poly = poly.fit_transform(X_train)

# poly.fit(X_train_poly, y_train)
# lin2 = LinearRegression()
# lin2.fit(X_train_poly, y_train)
# # producing r2 score
# poly_score = lin2.score(poly.fit_transform(X_test), y_test)
# print('Polynomial regression model produces an accuracy of', poly_score)
# #cross validation
# scores = cross_val_score(lin2, X_train_poly, y_train, scoring='neg_mean_squared_error', cv=5)
# poly_rmse_scores = np.sqrt(-scores)
# print(poly_rmse_scores)
# print("Polynomial regression model mean cross validation :", poly_rmse_scores.mean())

# building a polynomial regression model function with optimising degree
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
    print('The optimal degree of polynomial regression is', optimal_degree)

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


#%%
# #Decision tree
# Dtree_model = DecisionTreeRegressor(random_state=42)
# Dtree_model.fit(X_train, y_train)
# y_pred = Dtree_model.predict(X_test)
# decision_tree_score = r2_score(y_test, y_pred)
# print('Decision tree model produces an accuracy of', decision_tree_score)
# #cross validation
# scores = cross_val_score(Dtree_model, X_train, y_train, scoring='neg_mean_squared_error', cv=5)
# tree_rmse_scores = np.sqrt(-scores)
# print(tree_rmse_scores)
# print("Decision tree regression model mean cross validation :", tree_rmse_scores.mean())

# Building a decision tree regression model function
def DecisionTreeRegressionModel(X_train, y_train, X_test, y_test):

            # create a decision tree regression model object
            Dtree_model = DecisionTreeRegressor(random_state=42)

            # fit the model to the training data
            Dtree_model.fit(X_train, y_train)

            # predict on the test data
            y_pred = Dtree_model.predict(X_test)

            # evaluate the performance of the model on the test data
            r2 = r2_score(y_test, y_pred)

            print('We have predicted the price with an accuracy of',r2,'on the test set')

            # perform cross validation
            scores = cross_val_score(Dtree_model, X_train, y_train, scoring='neg_mean_squared_error', cv=5)
            rmse_scores = np.sqrt(-scores)
            print("Decision tree regression model mean cross validation: ", rmse_scores.mean())

            return r2, rmse_scores, y_pred


#%%
# Building a lasso model
# alpha = []
# r2 = []

# step = 0.1
# values = [i for i in range(int(0.0001/step), int(10/step))]

# for value in values:
#     print(value * step)
#     alpha.append(value*step)
#     lasso_model = Lasso(alpha=value*step)
#     lasso_model.fit(X_train, y_train)
#     r2.append(lasso_model.score(X_test, y_test))

# optimal_alpha = alpha[r2.index(max(r2))]
# print('The optimal value of alpha is', optimal_alpha)

# # Building a lasso model
# lasso_model = Lasso(optimal_alpha)
# lasso_model.fit(X_train, y_train)
# # producing r2 score
# lasso_score = lasso_model.score(X_test, y_test)
# print('Lasso model produces an accuracy of', lasso_score)
# #cross validation
# scores = cross_val_score(lasso_model, X_train, y_train, scoring='neg_mean_squared_error', cv=5)
# lasso_rmse_scores = np.sqrt(-scores)
# print(lasso_rmse_scores)
# print("Lasso regression model mean cross validation :", lasso_rmse_scores.mean())

# Building a lasso model function
def LassoRegressionModel(X_train, y_train, X_test, y_test):

    # optimizing the value of alpha
    alpha = []
    r2 = []

    step = 0.1
    values = [i for i in range(int(0.0001/step), int(10/step))]

    for value in values:
        print(value * step)
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

    return lasso_score, lasso_rmse_scores, y_pred
#%%
# # Building a gradient boosting model
# gradient_boosting_model = GradientBoostingRegressor()
# gradient_boosting_model.fit(X_train, y_train)
# # producing r2 score
# gradient_boosting_score = gradient_boosting_model.score(X_test, y_test)
# print('Gradient boosting model produces an accuracy of', gradient_boosting_score)
# #cross validation
# scores = cross_val_score(gradient_boosting_model, X_train, y_train, scoring='neg_mean_squared_error', cv=5)
# gradboost_rmse_scores = np.sqrt(-scores)
# print(gradboost_rmse_scores)
# print("Gradient boosting model mean cross validation :", gradboost_rmse_scores.mean())

# Building a gradient boosting model function
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

    return gradient_boosting_score, gradboost_rmse_scores, y_pred
#%% XGBR
#extreme gradient boosting
# extreme_gradient_boost =XGBRegressor()
# extreme_gradient_boost.fit(X_train, y_train)
# y_pred=extreme_gradient_boost.predict(X_test)
# extreme_gradient_boosting_score = extreme_gradient_boost.score(X_test, y_test)
# #xgbr_r2score = r2_score(y_test, y_pred)
# print('Extreme gradient boosting model produces an accuracy of', extreme_gradient_boosting_score)
# #print('Extreme gradient boosting model produces an r2 score of', xgbr_r2score)
# #cross validation
# scores = cross_val_score(extreme_gradient_boost, X_train, y_train, scoring='neg_mean_squared_error', cv=5)
# extreme_rmse_scores = np.sqrt(-scores)
# print(extreme_rmse_scores)
# print("Extreme gradient boosting model mean cross validation :", extreme_rmse_scores.mean())

#%%
# Plotting the r2 scores
scores = [linear_regression_score, ridge_regressor_score, random_forest_score, decision_tree_score, lasso_score, gradient_boosting_score, extreme_gradient_boosting_score]
names = ['Linear Regression', 'Ridge Regression', 'Random Forest', 'Decision Tree', 'Lasso', 'Gradient Boosting', 'Extreme Gradient Boosting']
# plt.figure()
# plt.bar(names, scores)
# plt.title('Accuracy of different models')
# plt.xticks(names, rotation = "vertical")
# plt.xlabel('Model')
# plt.ylabel('Accuracy')
# plt.show()

# making an r2 plotting function
def r2_plotting(names, scores):
    plt.figure()
    plt.bar(names, scores)
    plt.title('Accuracy of different models')
    plt.xticks(names, rotation = "vertical")
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.show()




#plotting cross validation scores
# cross_val_scores = [linear_rmse_scores.mean(), ridge_rmse_scores.mean(), random_rmse_scores.mean(), tree_rmse_scores.mean(), lasso_rmse_scores.mean(), gradboost_rmse_scores.mean(), extreme_rmse_scores.mean()]
# plt.figure()
# plt.bar(names, cross_val_scores)
# plt.title('Cross validation of different models')
# plt.xticks(names, rotation = "vertical")
# plt.ylim(0, 10000)
# plt.show()

# making a cross validation plotting function
def cross_val_plotting(names, cross_val_scores):
    plt.figure()
    plt.bar(names, cross_val_scores)
    plt.title('Cross validation of different models')
    plt.xticks(names, rotation = "vertical")
    plt.ylim(0, 10000)
    plt.show()

#%% DEEP LEARNING
# start_time = time.time()

# # Define the number of input features
# input_dims = X_train.shape[1]

# # Define the neural network architecture
# model = Sequential()
# model.add(Dense(512, input_dim=input_dims, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(256, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(32, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(1, activation='softmax'))

# # Compile the model
# model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy','mse','mae','mape'])

# # Train the model
# model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# #evaluate the model
# model.evaluate(X_test,y_test,verbose = 2)

# end_time = time.time()
# elapsed_time = end_time - start_time
# print("Elapsed time: ", elapsed_time, " seconds")

# creating a DNN function
def DNN(X_train, y_train, X_test, y_test):
    start_time = time.time()

    # Define the number of input features
    input_dims = X_train.shape[1]

    # Define the neural network architecture
    model = Sequential()
    model.add(Dense(512, input_dim=input_dims, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='softmax'))

    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy','mse','mae','mape'])

    # Train the model
    model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

    #evaluate the model
    model.evaluate(X_test,y_test,verbose = 2)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Elapsed time: ", elapsed_time, " seconds")

    return model
