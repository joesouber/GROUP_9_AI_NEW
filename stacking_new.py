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
from sklearn.model_selection import cross_val_score
drive.mount('/content/drive')


 #%% PUT THE CAR_PRICE_PREDICTION FILE IN THE SAME FOLDER AS THE AI FOLDER
  #%% PUT THE CAR_PRICE_PREDICTION FILE IN THE SAME FOLDER AS THE AI FOLDER
  car_data = '/content/drive/MyDrive/AI shared drive/car_price_prediction.csv'
  car_data = pd.read_csv(car_data, header = 0, skiprows=0, low_memory=False)
  
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

FormatData(car_data)

## Removes outliars 
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

#plt.boxplot(car_data_cleaned['Price'], notch=None, vert=None, patch_artist=None, widths=None)

# Deciding the inital threshold to be 1% of dataset size
length_data= car_data_cleaned.shape[0]
threshold = length_data*0.005
print ('The minimum count threshold is: '+str(threshold))
# Apply the count threshold to all the categorical values
obj_columns = list(car_data_cleaned.select_dtypes(include=['object']).columns)    # Get a list of all the columns' names with object dtype
car_data_cleaned = car_data_cleaned.apply(lambda x: x.mask(x.map(x.value_counts())<threshold, 'RARE') if x.name in obj_columns else x)

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

def LevyMedian(df):
  df['Levy'] = df['Levy'].fillna(df['Levy'].median())
  return df
  
LevyMedian(car_data_cleaned)

#Cleaned_and_final_data_one_hot = pd.get_dummies(Cleaned_and_final_data, columns = ['Manufacturer','Model','Category','Fuel type','Gear box type','Drive wheels','Color','Leather interior','Wheel','Turbo'])
#hashed out^^ as used above before predicting levy 
X = Cleaned_and_final_data.drop('Price',axis=1)
y = Cleaned_and_final_data['Price']
X = np.array(X)
y = np.array(y)
X_train, X_testval, y_train, y_testval = train_test_split(X, y, test_size=0.5, random_state = 42)
XB = np.array(X_testval)
yB = np.array(y_testval)
X_val, X_test, y_val, y_test = train_test_split(XB, yB, test_size=0.5, random_state = 42)

def cross_validation(regression_model, X_train, y_train):
  scores = cross_val_score(regression_model, X_train, y_train, scoring = "neg_mean_squared_error", cv = 5)
  rmse_scores = np.sqrt(-scores)
  print(rmse_scores)
  return rmse_scores

  linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
linear_regression_score = linear_model.score(X_test, y_test)
print('Linear regression model produces an accuracy of', linear_regression_score)

## working on this - finding data points with highest loss.
## still working on it
def ResidualScores(X_test, y_test, y_pred):
  residuals = y_test - y_pred
  squared_residuals = residuals**2
  n = 10
  top_indices = np.argsort(squared_res)[::-1][:n]
  print(top_indices) # these are the top 10 data points with the highest error.
  top_data_points = X_test[top_indices]
  print(top_data_points)  # then what do i do with this
  plt.scatter(y_pred[top_indices], residuals[top_indices])
  plt.xlabel('Predicted Values')
  plt.ylabel('Residuals')
  plt.title('Residual Plot')
  plt.show()
  return residuals, squared_residuals, top_indices, top_data_points


y_pred = linear_model.predict(X_test)
residuals = y_test - y_pred
squared_res = residuals**2
n = 10
top_indices = np.argsort(squared_res)[::-1][:n]
print(top_indices) # these are the top 10 data points with the highest error.
top_data_points = X_test[top_indices]
print(top_data_points)  # then what do i do with this

#scatter plot residuals against y_pred
plt.scatter(y_pred[top_indices], residuals[top_indices])
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.show()
#if you plot the top residuals, they are at either lower predicted values (< 15000) or at really high predicted values

#leverage
# Calculate leverage scores
XTX_inv = np.linalg.inv(np.dot(X_train.T, X_train))
H = np.dot(np.dot(X_train, XTX_inv), X_train.T)
leverage_scores = np.diag(H)
#print(leverage_scores)

top_lev_indices = np.argsort(leverage_scores)[::-1][:n]
print(top_lev_indices) # these are the top 10 data points with the highest error.
top_lev_points = X_train[top_indices]
print(top_lev_points)
plt.scatter(y_train[top_lev_indices], leverage_scores[top_lev_indices])
plt.ylabel('Leverage Scores')
plt.xlabel('Predicted Values')
plt.title('Scatter plot of predicted values vs. leverage scores')
plt.show()

# Ridge Regression Model
# Need to find the right value of the hyperparameter alpha
# finding the optimal value of alpha by plotting the r2 score against alpha
alpha = []
r2 = []
step = 0.1
values = [i for i in range(int(0/step), int(10/step))]

mape=[]

for value in values:
    alpha.append(value*step)
    ridge_model = Ridge(alpha=value*step)
    ridge_model.fit(X_train, y_train)
    r2.append(ridge_model.score(X_test, y_test))

optimal_alpha = alpha[r2.index(max(r2))]

ridge_regressor = Ridge(optimal_alpha)
ridge_regressor.fit(X_train, y_train)

ridge_regressor_score = ridge_regressor.score(X_test, y_test)
print('Ridge regression model produces an accuracy of', ridge_regressor_score)

#cross validation
cross_validation(ridge_regressor, X_train, y_train)

#Random forest 
random_model = RandomForestRegressor(random_state=42)
random_model.fit(X_train, y_train)
y_pred = random_model.predict(X_test)
random_forest_score = r2_score(y_test,y_pred)
print('Random forest regression model produces an accuracy of', random_forest_score)

#cross validation
cross_validation(random_model, X_train, y_train)

#Decision tree 
Dtree_model = DecisionTreeRegressor(random_state=42)
Dtree_model.fit(X_train, y_train)
y_pred = Dtree_model.predict(X_test)
decision_tree_score = r2_score(y_test, y_pred)
print('Decision tree model produces an accuracy of', decision_tree_score)

#cross validation
cross_validation(Dtree_model, X_train, y_train)

# Building a lasso model 
from sklearn.linear_model import Lasso
alpha = []
r2 = []

mape=[]

step = 0.1
values = [i for i in range(int(0.0001/step), int(10/step))]

for value in values:
    #print(value * step)
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
cross_validation(lasso_model, X_train, y_train)

#extreme gradient boosting
extreme_gradient_boost =XGBRegressor()
extreme_gradient_boost.fit(X_train, y_train)
y_pred=extreme_gradient_boost.predict(X_test)
extreme_gradient_boosting_score = extreme_gradient_boost.score(X_test, y_test)
#xgbr_r2score = r2_score(y_test, y_pred)
print('Extreme gradient boosting model produces an accuracy of', extreme_gradient_boosting_score)
#print('Extreme gradient boosting model produces an r2 score of', xgbr_r2score)

#cross validation
cross_validation(extreme_gradient_boost, X_train, y_train)

def MAPE(Y_actual, Y_predicted):
  mape = np.mean(np.abs((Y_actual-Y_predicted)/Y_actual))*100
  return mape


from sklearn.ensemble import StackingRegressor


model_types=[['linear', linear_model], ['ridge', ridge_model], ['rf', random_model], ['decision', Dtree_model], ['lasso', lasso_model], ['xgbr', extreme_gradient_boost]]

res = [(a, b) for idx, a in enumerate(model_types) for b in model_types[idx + 1:]]
#print(res) 

stack=[]
bar_chart_labels=[]

for [a,b] in res:
#  # print(estimator)

  estimators =[a,b]
  stack_reg = StackingRegressor(estimators=estimators, final_estimator=RandomForestRegressor(random_state=42))
  #print("estimators:", estimators)
  bar_chart_labels.append(estimators)
# #Fit the StackingRegressor on the training data
  regression = stack_reg.fit(X_train, y_train)
# #Predict target varaible on the test data using the StackingRegressor
  y_pred = stack_reg.predict(X_test)

# #Evaluate the performance of the StackingRegressor using R-squared 
  r2 = r2_score(y_test, y_pred)
  stack.append(r2)
  
  regression_mape = MAPE(y_test, y_pred)
  print('final_estimator=rf', a, b, 'R-squared:', r2, 'MAPE:', regression_mape)


import matplotlib.pyplot as plt

bar_chart_labels=['linear and ridge', 'linear and random forest', 'linear and decision', 'linear and lasso', 'linear and xgbr', 'ridge and random forest', 'ridge and decision', 'ridge and lasso', 'ridge and xgbr', 'random forest and decision', 'random forest and lasso', 'random forest and xgbr','decision and lasso','decision and xgbr','lasso and xgbr']

print(bar_chart_labels)
x_pos = [i for i, _ in enumerate(bar_chart_labels)]

plt.bar(x_pos, r2, color='green')
plt.xlabel("the combination of estimators")
plt.ylabel("r2 value")
plt.title("stacking for final_estimator as random forest")

plt.xticks(x_pos, bar_chart_labels, rotation='vertical')

plt.show()


plt.bar(x_pos, regression_mape, color='blue')
plt.xlabel("the combination of estimators")
plt.ylabel("MAPE value")
plt.title("stacking for final_estimator as random forest")

plt.xticks(x_pos, bar_chart_labels, rotation='vertical')

plt.show()

#final_estimator=Ridge
model_types=[['linear', linear_model], ['ridge', ridge_model], ['rf', random_model], ['decision', Dtree_model], ['lasso', lasso_model], ['xgbr', extreme_gradient_boost]]

res = [(a, b) for idx, a in enumerate(model_types) for b in model_types[idx + 1:]]
#print(res) 

stack=[]

for [a,b] in res:
#  # print(estimator)

  estimators =[a,b]
  stack_reg = StackingRegressor(estimators=estimators, final_estimator=Ridge(optimal_alpha))
  #print("estimators:", estimators)

# #Fit the StackingRegressor on the training data
  regression = stack_reg.fit(X_train, y_train)
# #Predict target varaible on the test data using the StackingRegressor
  y_pred = stack_reg.predict(X_test)

# #Evaluate the performance of the StackingRegressor using R-squared 
  r2 = r2_score(y_test, y_pred)
  
  regression_mape = MAPE(y_test, y_pred)

import matplotlib.pyplot as plt

bar_chart_labels=['linear and ridge', 'linear and random forest', 'linear and decision', 'linear and lasso', 'linear and xgbr', 'ridge and random forest', 'ridge and decision', 'ridge and lasso', 'ridge and xgbr', 'random forest and decision', 'random forest and lasso', 'random forest and xgbr','decision and lasso','decision and xgbr','lasso and xgbr']

print(bar_chart_labels)
x_pos = [i for i, _ in enumerate(bar_chart_labels)]

plt.bar(x_pos, r2, color='green')
plt.xlabel("combination of estimators")
plt.ylabel("r2 value")
plt.title("stacking for final_estimator as ridge")

plt.xticks(x_pos, bar_chart_labels, rotation='vertical')

plt.show()

plt.bar(x_pos, regression_mape, color='blue')
plt.xlabel("combination of estimators")
plt.ylabel("MAPE value")
plt.title("stacking for final_estimator as ridge")

plt.xticks(x_pos, bar_chart_labels, rotation='vertical')

plt.show()

#final_estimator=DecisionTreeRegressor
model_types=[['linear', linear_model], ['ridge', ridge_model], ['rf', random_model], ['decision', Dtree_model], ['lasso', lasso_model], ['xgbr', extreme_gradient_boost]]

res = [(a, b) for idx, a in enumerate(model_types) for b in model_types[idx + 1:]]
#print(res) 

stack=[]

for [a,b] in res:
#  # print(estimator)

  estimators =[a,b]
  stack_reg = StackingRegressor(estimators=estimators, final_estimator=DecisionTreeRegressor(random_state=42))
  #print("estimators:", estimators)

# #Fit the StackingRegressor on the training data
  regression = stack_reg.fit(X_train, y_train)
# #Predict target varaible on the test data using the StackingRegressor
  y_pred = stack_reg.predict(X_test)

# #Evaluate the performance of the StackingRegressor using R-squared 
  r2 = r2_score(y_test, y_pred)
  
  regression_mape = MAPE(y_test, y_pred)
  print('final_estimator=DecisionTreeRegressor', a, b, 'R-squared:', r2, 'MAPE:', regression_mape)

 import matplotlib.pyplot as plt


bar_chart_labels=['linear and ridge', 'linear and random forest', 'linear and decision', 'linear and lasso', 'linear and xgbr', 'ridge and random forest', 'ridge and decision', 'ridge and lasso', 'ridge and xgbr', 'random forest and decision', 'random forest and lasso', 'random forest and xgbr','decision and lasso','decision and xgbr','lasso and xgbr']

print(bar_chart_labels)
x_pos = [i for i, _ in enumerate(bar_chart_labels)]

plt.bar(x_pos, r2, color='green')
plt.xlabel("combination of estimators")
plt.ylabel("r2 value")
plt.title("stacking for final_estimator as decision tree")

plt.xticks(x_pos, bar_chart_labels, rotation='vertical')

plt.show()

plt.bar(x_pos, regression_mape, color='blue')
plt.xlabel("combination of estimators")
plt.ylabel("MAPE value")
plt.title("stacking for final_estimator as decision tree")

plt.xticks(x_pos, bar_chart_labels, rotation='vertical')

plt.show()

#final_estimator=LinearRegression
model_types=[['linear', linear_model], ['ridge', ridge_model], ['rf', random_model], ['decision', Dtree_model], ['lasso', lasso_model], ['xgbr', extreme_gradient_boost]]

res = [(a, b) for idx, a in enumerate(model_types) for b in model_types[idx + 1:]]
#print(res) 

stack=[]

for [a,b] in res:
#  # print(estimator)

  estimators =[a,b]
  stack_reg = StackingRegressor(estimators=estimators, final_estimator=LinearRegression())
  #print("estimators:", estimators)

# #Fit the StackingRegressor on the training data
  regression = stack_reg.fit(X_train, y_train)
# #Predict target varaible on the test data using the StackingRegressor
  y_pred = stack_reg.predict(X_test)

# #Evaluate the performance of the StackingRegressor using R-squared 
  r2 = r2_score(y_test, y_pred)
  
  regression_mape = MAPE(y_test, y_pred)
  print('final_estimator=LinearRegression', a, b, 'R-squared:', r2, 'MAPE:', regression_mape)


import matplotlib.pyplot as plt


bar_chart_labels=['linear and ridge', 'linear and random forest', 'linear and decision', 'linear and lasso', 'linear and xgbr', 'ridge and random forest', 'ridge and decision', 'ridge and lasso', 'ridge and xgbr', 'random forest and decision', 'random forest and lasso', 'random forest and xgbr','decision and lasso','decision and xgbr','lasso and xgbr']

print(bar_chart_labels)
x_pos = [i for i, _ in enumerate(bar_chart_labels)]

plt.bar(x_pos, r2, color='green')
plt.xlabel("combination of estimators")
plt.ylabel("r2 value")
plt.title("stacking for final_estimator as linear regression")

plt.xticks(x_pos, bar_chart_labels, rotation='vertical')

plt.show()

plt.bar(x_pos, regression_mape, color='blue')
plt.xlabel("combination of estimators")
plt.ylabel("MAPE value")
plt.title("stacking for final_estimator as linear regression")

plt.xticks(x_pos, bar_chart_labels, rotation='vertical')

plt.show()

#final_estimator=Lasso
model_types=[['linear', linear_model], ['ridge', ridge_model], ['rf', random_model], ['decision', Dtree_model], ['lasso', lasso_model], ['xgbr', extreme_gradient_boost]]

res = [(a, b) for idx, a in enumerate(model_types) for b in model_types[idx + 1:]]
#print(res) 

stack=[]

for [a,b] in res:
#  # print(estimator)

  estimators =[a,b]
  stack_reg = StackingRegressor(estimators=estimators, final_estimator=Lasso(optimal_alpha))
  #print("estimators:", estimators)

# #Fit the StackingRegressor on the training data
  regression = stack_reg.fit(X_train, y_train)
# #Predict target varaible on the test data using the StackingRegressor
  y_pred = stack_reg.predict(X_test)

# #Evaluate the performance of the StackingRegressor using R-squared 
  r2 = r2_score(y_test, y_pred)
  
  regression_mape = MAPE(y_test, y_pred)
  print('final_estimator=Lasso', a, b, 'R-squared:', r2, 'MAPE:', regression_mape)

bar_chart_labels=['linear and ridge', 'linear and random forest', 'linear and decision', 'linear and lasso', 'linear and xgbr', 'ridge and random forest', 'ridge and decision', 'ridge and lasso', 'ridge and xgbr', 'random forest and decision', 'random forest and lasso', 'random forest and xgbr','decision and lasso','decision and xgbr','lasso and xgbr']

print(bar_chart_labels)
x_pos = [i for i, _ in enumerate(bar_chart_labels)]

plt.bar(x_pos, r2, color='green')
plt.xlabel("combination of estimators")
plt.ylabel("r2 value")
plt.title("stacking for final_estimator as lasso")

plt.xticks(x_pos, bar_chart_labels, rotation='vertical')

plt.show()

plt.bar(x_pos, regression_mape, color='blue')
plt.xlabel("combination of estimators")
plt.ylabel("MAPE value")
plt.title("stacking for final_estimator as lasso")

plt.xticks(x_pos, bar_chart_labels, rotation='vertical')

plt.show()

#final_estimator=XGBRegressor
model_types=[['linear', linear_model], ['ridge', ridge_model], ['rf', random_model], ['decision', Dtree_model], ['lasso', lasso_model], ['xgbr', extreme_gradient_boost]]

res = [(a, b) for idx, a in enumerate(model_types) for b in model_types[idx + 1:]]
#print(res) 

stack=[]

for [a,b] in res:
#  # print(estimator)

  estimators =[a,b]
  stack_reg = StackingRegressor(estimators=estimators, final_estimator=XGBRegressor())
  #print("estimators:", estimators)

# #Fit the StackingRegressor on the training data
  regression = stack_reg.fit(X_train, y_train)
# #Predict target varaible on the test data using the StackingRegressor
  y_pred = stack_reg.predict(X_test)

# #Evaluate the performance of the StackingRegressor using R-squared 
  r2 = r2_score(y_test, y_pred)
  stack.append(r2)
  
  regression_mape = MAPE(y_test, y_pred)
  print('final_estimator=XGBRegressor', a, b, 'R-squared:', r2, 'MAPE:', regression_mape)

bar_chart_labels=['linear and ridge', 'linear and random forest', 'linear and decision', 'linear and lasso', 'linear and xgbr', 'ridge and random forest', 'ridge and decision', 'ridge and lasso', 'ridge and xgbr', 'random forest and decision', 'random forest and lasso', 'random forest and xgbr','decision and lasso','decision and xgbr','lasso and xgbr']

print(bar_chart_labels)
x_pos = [i for i, _ in enumerate(bar_chart_labels)]

plt.bar(x_pos, r2, color='green')
plt.xlabel("combination of estimators")
plt.ylabel("r2 value")
plt.title("stacking for final_estimator as xgbr")

plt.xticks(x_pos, bar_chart_labels, rotation='vertical')

plt.show()

plt.bar(x_pos, regression_mape, color='blue')
plt.xlabel("combination of estimators")
plt.ylabel("MAPE value")
plt.title("stacking for final_estimator as xgbr")

plt.xticks(x_pos, bar_chart_labels, rotation='vertical')

plt.show()  