#%% IMPORTS

import random
import matplotlib.pyplot as plt
import numpy as np
import csv
import pandas as pd
#import seaborn as sn
#from google.colab import drive
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as mse
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
#drive.mount('/content/drive')


#%% PUT THE CAR_PRICE_PREDICTION FILE IN THE SAME FOLDER AS THE AI FOLDER
car_data = '/content/drive/MyDrive/AI shared drive/car_price_prediction.csv'
car_data = pd.read_csv(car_data, header = 0, skiprows=0, low_memory=False)


car_data.replace('-',np.nan, inplace=True) #replaces '-' with Nan values
car_data['Levy'] = car_data['Levy'].astype('float64')

car_data['Mileage'] = car_data['Mileage'].str.extract('(\d+)').astype(int) # This is going to remove the 'km' in the mileage
car_data['Mileage'] = car_data['Mileage'].astype('int64')

car_data['Leather interior'] = car_data['Leather interior'].replace({'Yes': True, 'No': False})#replace 'Leather interior yes/no with T/F

car_data['Wheel'] = car_data['Wheel'].replace({'Right-hand drive': True, 'Left wheel': False})#replace 'Wheel' Right-hand drive/left wheel with T/F

car_data['Turbo'] = car_data['Engine volume'].str.contains('Turbo') #place turbo in separate new column with T/F.

car_data['Engine volume'] = car_data['Engine volume'].str.extract(r'(\d+\.\d+|\d+)').astype(float) # remove turbo from engine type, 
car_data['Engine volume'] = car_data['Engine volume'].astype('float64')
car_data['Doors'].replace({'04-May':4, '02-Mar':2, '>5':5}, inplace=True) #replace doors dates with 2,4,5

car_data = car_data.drop('ID', axis=1)


car_data
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

plt.boxplot(car_data_cleaned['Price'], notch=None, vert=None, patch_artist=None, widths=None)
max(car_data_cleaned['Price'])


car_data_cleaned
#%%
#making two new dataframes, one dealing with numerical values, other with words.

num_attribs = ['Levy','Prod. year', 'Engine volume','Doors', 'Mileage', 'Cylinders', 'Airbags']
cat_attribs = ['Manufacturer','Category', 'Leather interior', 'Fuel type', 'Gear box type', 'Drive wheels', 'Wheel', 'Color', 'Turbo']

num_cars = car_data_cleaned[num_attribs]
Objective = car_data_cleaned['Price']
cat_cars = car_data_cleaned[cat_attribs]


#%%
## replacing the NAN values in the Levy column with values using Knearest neighbours
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


imputer = KNNImputer(n_neighbors=10)
num_cars = imputer.fit_transform(num_cars)
scaler = StandardScaler()
num_cars = scaler.fit_transform(num_cars)



num_pipeline = Pipeline([
    ('imputer', KNNImputer(n_neighbors=10)),
    ('std_scaler', StandardScaler())])

full_pipeline = ColumnTransformer([
    ('num',num_pipeline, num_attribs),
    ('cat',OneHotEncoder(), cat_attribs)  
])
cars_prepared = full_pipeline.fit_transform(car_data_cleaned)



#attempting to replace Nan values in levy colum using 

#car_data_cleaned.head()

bosh = car_data_cleaned['Levy'].isnull() #finding where the nans are true = Nan, false = no Nan
nan_indices = car_data_cleaned[bosh].index    #turning into index

df_with_nans = car_data_cleaned.loc[nan_indices]   #splitting dataset up into nan and no nan
df_without_nans = car_data_cleaned.loc[~bosh]
NUMBERS =['Price','Levy','Prod. year', 'Engine volume','Doors', 'Mileage', 'Cylinders', 'Airbags']
df_without_nans_NUMBERS = df_without_nans[NUMBERS]
df_with_nans_NUMBERS = df_with_nans[NUMBERS]

X_train, X_test, y_train, y_test = train_test_split(df_without_nans_NUMBERS.drop('Levy', axis=1), df_without_nans_NUMBERS['Levy'], test_size=0.33, random_state=42)

# create a decision tree regressor object
reg = DecisionTreeRegressor(random_state=42)

# fit the regressor to the training data
reg.fit(X_train, y_train)

# predict on the test data
y_pred = reg.predict(X_test)

# evaluate the performance of the model on the test data
r2 = r2_score(y_test, y_pred)

print('We have predicted the levy price with an accuracy of',r2,'on the test set') 


#Now we get a r2 score of 78.3% with a test size of 33%. Use this predictor on the Nan dataset.
df_with_nans_NUMBERS = df_with_nans_NUMBERS.drop('Levy', axis=1)

Nan_pred = reg.predict(df_with_nans_NUMBERS)
df_with_nans['Levy'] = Nan_pred

Cleaned_and_final_data = pd.concat([df_with_nans, df_without_nans])

