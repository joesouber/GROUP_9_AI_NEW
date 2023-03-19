#%% IMPORTS

import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error
import random
import matplotlib.pyplot as plt
import numpy as np
import csv
import pandas as pd
import seaborn as sn
from google.colab import drive
from sklearn.impute import KNNImputer
from sklearn.ensemble import StackingRegressor
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
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MinMaxScaler
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
#car_data = 'Documents\INTRO_TO_AI\car_price_prediction.csv'
df = pd.read_csv(car_data, header = 0, skiprows=0, low_memory=False)  
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


X_train, y_train, X_test, y_test, X_val, y_val = setup_x_y()

def RandomForestRegressionModel(X_train, y_train, X_test, y_test):
    random_model = RandomForestRegressor()
    random_model.fit(X_train, y_train)
    y_pred = random_model.predict(X_test)
    random_r2 = r2_score(y_test, y_pred)
    random_mape = mean_absolute_percentage_error(y_test, y_pred)
    random_mse = mse(y_test, y_pred)
    return random_r2, random_mape, random_mse, random_model
    
def LinearRegressionModel(X_train, y_train, X_test, y_test):
    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)
    y_pred = linear_model.predict(X_test)
    linear_r2 = r2_score(y_test, y_pred)
    linear_mape = mean_absolute_percentage_error(y_test, y_pred)
    linear_mse = mse(y_test, y_pred)
    return linear_r2, linear_mape, linear_mse, linear_model
     


def LassoRegressionModel(X_train, y_train, X_test, y_test):
    lasso_model = Lasso(alpha = 2.7)
    lasso_model.fit(X_train, y_train)
    y_pred = lasso_model.predict(X_test)
    lasso_r2 = r2_score(y_test, y_pred)
    lasso_mape = mean_absolute_percentage_error(y_test, y_pred)
    lasso_mse = mse(y_test, y_pred)
    return lasso_r2, lasso_mape, lasso_mse, lasso_model


def RidgeRegressionModel(X_train, y_train, X_test, y_test):
    ridge_model = Ridge(alpha = 1.0)
    ridge_model.fit(X_train, y_train)
    y_pred = ridge_model.predict(X_test)
    ridge_r2 = r2_score(y_test, y_pred)
    ridge_mape = mean_absolute_percentage_error(y_test, y_pred)
    ridge_mse = mse(y_test, y_pred)
    return ridge_r2, ridge_mape, ridge_mse, ridge_model

def GradBoostModel(X_train, y_train, X_test, y_test):
    gbr_model = GradientBoostingRegressor()
    gbr_model.fit(X_train, y_train)
    y_pred = gbr_model.predict(X_test)
    gbr_r2 = r2_score(y_test, y_pred)
    gbr_mape = mean_absolute_percentage_error(y_test, y_pred)
    gbr_mse = mse(y_test, y_pred)
    return gbr_r2, gbr_mape, gbr_mse, gbr_model
    
def XGBRmodel(X_train, y_train, X_test, y_test):
    xgbr_model =XGBRegressor()
    xgbr_model.fit(X_train, y_train)
    y_pred = xgbr_model.predict(X_test)
    xgbr_r2 = r2_score(y_test, y_pred)
    xgbr_mape = mean_absolute_percentage_error(y_test, y_pred)
    xgbr_mse = mse(y_test, y_pred)
    return xgbr_r2, xgbr_mape, xgbr_mse, xgbr_model

def DecisionTreeModel(X_train, y_train, X_test, y_test):
    dtree_model = DecisionTreeRegressor()
    dtree_model.fit(X_train, y_train)
    y_pred = dtree_model.predict(X_test)
    dtree_r2 = r2_score(y_test, y_pred)
    dtree_mape = mean_absolute_percentage_error(y_test, y_pred)
    dtree_mse = mse(y_test, y_pred)
    return dtree_r2, dtree_mape, dtree_mse, dtree_model


def MAPE(y_test, y_pred):
  mape = mean_absolute_percentage_error(y_test, y_pred)
  return mape



linear_model = LinearRegressionModel(X_train, y_train, X_test, y_test)[3]
ridge_model = RidgeRegressionModel(X_train, y_train, X_test, y_test)[3]
random_model = RandomForestRegressionModel(X_train, y_train, X_test, y_test)[3]
Dtree_model = DecisionTreeModel(X_train, y_train, X_test, y_test)[3]
lasso_model = LassoRegressionModel(X_train, y_train, X_test, y_test)[3]
extreme_gradient_boost = XGBRmodel(X_train, y_train, X_test, y_test)[3]

model_types=[['linear', linear_model], ['ridge', ridge_model], ['rf', random_model], ['decision', Dtree_model], ['lasso', lasso_model], ['xgbr', extreme_gradient_boost]]

res = [(a, b) for idx, a in enumerate(model_types) for b in model_types[idx + 1:]]
#print(res) 

stack=[]
mape_scores = []
bar_chart_labels=[]

for [a,b] in res:
#  # print(estimator)

  estimators =[a,b]
  stack_reg = StackingRegressor(estimators=estimators, final_estimator=RandomForestRegressor(random_state=42))

  bar_chart_labels.append(estimators)
  regression = stack_reg.fit(X_train, y_train)
  y_pred = stack_reg.predict(X_test) 
  r2 = r2_score(y_test, y_pred)
  stack.append(r2)
  regression_mape = MAPE(y_test, y_pred)
  mape_scores.append(regression_mape)
  
 #print('final_estimator=rf', a, b, 'R-squared:', r2, 'MAPE:', regression_mape)



bar_chart_labels=['linear and ridge', 'linear and random forest', 'linear and decision', 'linear and lasso', 'linear and xgbr', 'ridge and random forest', 'ridge and decision', 'ridge and lasso', 'ridge and xgbr', 'random forest and decision', 'random forest and lasso', 'random forest and xgbr','decision and lasso','decision and xgbr','lasso and xgbr']

print(bar_chart_labels)
x_pos = [i for i, _ in enumerate(bar_chart_labels)]

plt.bar(x_pos, mape_scores, color='rebeccapurple')
#plt.xlabel("the combination of estimators")
plt.ylabel("MAPE")
plt.title("stacking with final_estimator as random forest regressor")
plt.xticks(x_pos, bar_chart_labels, rotation= 45)

plt.show()



#final_estimator=Ridge
model_types=[['linear', linear_model], ['ridge', ridge_model], ['rf', random_model], ['decision', Dtree_model], ['lasso', lasso_model], ['xgbr', extreme_gradient_boost]]

res = [(a, b) for idx, a in enumerate(model_types) for b in model_types[idx + 1:]]
#print(res) 

stack2=[]
mape_scores2 = []

for [a,b] in res:
#  # print(estimator)

  estimators =[a,b]
  stack_reg = StackingRegressor(estimators=estimators, final_estimator=Ridge(1.0))

  regression = stack_reg.fit(X_train, y_train)

  y_pred = stack_reg.predict(X_test)

  r2 = r2_score(y_test, y_pred)
  stack2.append(r2)
  
  regression_mape = MAPE(y_test, y_pred)
  mape_scores2.append(regression_mape)


bar_chart_labels=['linear and ridge', 'linear and random forest', 'linear and decision', 'linear and lasso', 'linear and xgbr', 'ridge and random forest', 'ridge and decision', 'ridge and lasso', 'ridge and xgbr', 'random forest and decision', 'random forest and lasso', 'random forest and xgbr','decision and lasso','decision and xgbr','lasso and xgbr']

print(bar_chart_labels)
x_pos = [i for i, _ in enumerate(bar_chart_labels)]


plt.bar(x_pos, mape_scores2, color='rebeccapurple')
#plt.xlabel("the combination of estimators")
plt.ylabel("MAPE")
plt.title("stacking with final_estimator as ridge regression")
plt.xticks(x_pos, bar_chart_labels, rotation= 45)

plt.show()

#%%
# #final_estimator=DecisionTreeRegressor
# model_types=[['linear', linear_model], ['ridge', ridge_model], ['rf', random_model], ['decision', Dtree_model], ['lasso', lasso_model], ['xgbr', extreme_gradient_boost]]

# res = [(a, b) for idx, a in enumerate(model_types) for b in model_types[idx + 1:]]
# #print(res) 

# stack=[]

# for [a,b] in res:
# #  # print(estimator)

#   estimators =[a,b]
#   stack_reg = StackingRegressor(estimators=estimators, final_estimator=DecisionTreeRegressor(random_state=42))
#   #print("estimators:", estimators)

# # #Fit the StackingRegressor on the training data
#   regression = stack_reg.fit(X_train, y_train)
# # #Predict target varaible on the test data using the StackingRegressor
#   y_pred = stack_reg.predict(X_test)

# # #Evaluate the performance of the StackingRegressor using R-squared 
#   r2 = r2_score(y_test, y_pred)
  
#   regression_mape = MAPE(y_test, y_pred)
#   print('final_estimator=DecisionTreeRegressor', a, b, 'R-squared:', r2, 'MAPE:', regression_mape)




# bar_chart_labels=['linear and ridge', 'linear and random forest', 'linear and decision', 'linear and lasso', 'linear and xgbr', 'ridge and random forest', 'ridge and decision', 'ridge and lasso', 'ridge and xgbr', 'random forest and decision', 'random forest and lasso', 'random forest and xgbr','decision and lasso','decision and xgbr','lasso and xgbr']

# print(bar_chart_labels)
# x_pos = [i for i, _ in enumerate(bar_chart_labels)]

# plt.bar(x_pos, r2, color='green')
# plt.xlabel("combination of estimators")
# plt.ylabel("r2 value")
# plt.title("stacking for final_estimator as decision tree")

# plt.xticks(x_pos, bar_chart_labels, rotation='vertical')

# plt.show()

# plt.bar(x_pos, regression_mape, color='blue')
# plt.xlabel("combination of estimators")
# plt.ylabel("MAPE value")
# plt.title("stacking for final_estimator as decision tree")

# plt.xticks(x_pos, bar_chart_labels, rotation='vertical')

# plt.show()

# #final_estimator=LinearRegression
# model_types=[['linear', linear_model], ['ridge', ridge_model], ['rf', random_model], ['decision', Dtree_model], ['lasso', lasso_model], ['xgbr', extreme_gradient_boost]]

# res = [(a, b) for idx, a in enumerate(model_types) for b in model_types[idx + 1:]]
# #print(res) 

# stack=[]

# for [a,b] in res:
# #  # print(estimator)

#   estimators =[a,b]
#   stack_reg = StackingRegressor(estimators=estimators, final_estimator=LinearRegression())
#   #print("estimators:", estimators)

# # #Fit the StackingRegressor on the training data
#   regression = stack_reg.fit(X_train, y_train)
# # #Predict target varaible on the test data using the StackingRegressor
#   y_pred = stack_reg.predict(X_test)

# # #Evaluate the performance of the StackingRegressor using R-squared 
#   r2 = r2_score(y_test, y_pred)
  
#   regression_mape = MAPE(y_test, y_pred)
#   print('final_estimator=LinearRegression', a, b, 'R-squared:', r2, 'MAPE:', regression_mape)


# import matplotlib.pyplot as plt


# bar_chart_labels=['linear and ridge', 'linear and random forest', 'linear and decision', 'linear and lasso', 'linear and xgbr', 'ridge and random forest', 'ridge and decision', 'ridge and lasso', 'ridge and xgbr', 'random forest and decision', 'random forest and lasso', 'random forest and xgbr','decision and lasso','decision and xgbr','lasso and xgbr']

# print(bar_chart_labels)
# x_pos = [i for i, _ in enumerate(bar_chart_labels)]

# plt.bar(x_pos, r2, color='green')
# plt.xlabel("combination of estimators")
# plt.ylabel("r2 value")
# plt.title("stacking for final_estimator as linear regression")

# plt.xticks(x_pos, bar_chart_labels, rotation='vertical')

# plt.show()

# plt.bar(x_pos, regression_mape, color='blue')
# plt.xlabel("combination of estimators")
# plt.ylabel("MAPE value")
# plt.title("stacking for final_estimator as linear regression")

# plt.xticks(x_pos, bar_chart_labels, rotation='vertical')

# plt.show()

# #final_estimator=Lasso
# model_types=[['linear', linear_model], ['ridge', ridge_model], ['rf', random_model], ['decision', Dtree_model], ['lasso', lasso_model], ['xgbr', extreme_gradient_boost]]

# res = [(a, b) for idx, a in enumerate(model_types) for b in model_types[idx + 1:]]
# #print(res) 

# stack=[]

# for [a,b] in res:
# #  # print(estimator)

#   estimators =[a,b]
#   stack_reg = StackingRegressor(estimators=estimators, final_estimator=Lasso(optimal_alpha))
#   #print("estimators:", estimators)

# # #Fit the StackingRegressor on the training data
#   regression = stack_reg.fit(X_train, y_train)
# # #Predict target varaible on the test data using the StackingRegressor
#   y_pred = stack_reg.predict(X_test)

# # #Evaluate the performance of the StackingRegressor using R-squared 
#   r2 = r2_score(y_test, y_pred)
  
#   regression_mape = MAPE(y_test, y_pred)
#   print('final_estimator=Lasso', a, b, 'R-squared:', r2, 'MAPE:', regression_mape)

# bar_chart_labels=['linear and ridge', 'linear and random forest', 'linear and decision', 'linear and lasso', 'linear and xgbr', 'ridge and random forest', 'ridge and decision', 'ridge and lasso', 'ridge and xgbr', 'random forest and decision', 'random forest and lasso', 'random forest and xgbr','decision and lasso','decision and xgbr','lasso and xgbr']

# print(bar_chart_labels)
# x_pos = [i for i, _ in enumerate(bar_chart_labels)]

# plt.bar(x_pos, r2, color='green')
# plt.xlabel("combination of estimators")
# plt.ylabel("r2 value")
# plt.title("stacking for final_estimator as lasso")

# plt.xticks(x_pos, bar_chart_labels, rotation='vertical')

# plt.show()

# plt.bar(x_pos, regression_mape, color='blue')
# plt.xlabel("combination of estimators")
# plt.ylabel("MAPE value")
# plt.title("stacking for final_estimator as lasso")

# plt.xticks(x_pos, bar_chart_labels, rotation='vertical')

# plt.show()

# #final_estimator=XGBRegressor
# model_types=[['linear', linear_model], ['ridge', ridge_model], ['rf', random_model], ['decision', Dtree_model], ['lasso', lasso_model], ['xgbr', extreme_gradient_boost]]

# res = [(a, b) for idx, a in enumerate(model_types) for b in model_types[idx + 1:]]
# #print(res) 

# stack=[]

# for [a,b] in res:
# #  # print(estimator)

#   estimators =[a,b]
#   stack_reg = StackingRegressor(estimators=estimators, final_estimator=XGBRegressor())
#   #print("estimators:", estimators)

# # #Fit the StackingRegressor on the training data
#   regression = stack_reg.fit(X_train, y_train)
# # #Predict target varaible on the test data using the StackingRegressor
#   y_pred = stack_reg.predict(X_test)

# # #Evaluate the performance of the StackingRegressor using R-squared 
#   r2 = r2_score(y_test, y_pred)
#   stack.append(r2)
  
#   regression_mape = MAPE(y_test, y_pred)
#   print('final_estimator=XGBRegressor', a, b, 'R-squared:', r2, 'MAPE:', regression_mape)

# bar_chart_labels=['linear and ridge', 'linear and random forest', 'linear and decision', 'linear and lasso', 'linear and xgbr', 'ridge and random forest', 'ridge and decision', 'ridge and lasso', 'ridge and xgbr', 'random forest and decision', 'random forest and lasso', 'random forest and xgbr','decision and lasso','decision and xgbr','lasso and xgbr']

# print(bar_chart_labels)
# x_pos = [i for i, _ in enumerate(bar_chart_labels)]

# plt.bar(x_pos, r2, color='green')
# plt.xlabel("combination of estimators")
# plt.ylabel("r2 value")
# plt.title("stacking for final_estimator as xgbr")

# plt.xticks(x_pos, bar_chart_labels, rotation='vertical')

# plt.show()

# plt.bar(x_pos, regression_mape, color='blue')
# plt.xlabel("combination of estimators")
# plt.ylabel("MAPE value")
# plt.title("stacking for final_estimator as xgbr")

# plt.xticks(x_pos, bar_chart_labels, rotation='vertical')

# plt.show()  