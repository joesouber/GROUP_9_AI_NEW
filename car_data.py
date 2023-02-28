#%% IMPORTS

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
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
drive.mount('/content/drive')

#%% PUT THE CAR_PRICE_PREDICTION FILE IN THE SAME FOLDER AS THE AI FOLDER
car_data = '/content/drive/MyDrive/AI shared drive/car_price_prediction.csv'
car_data = pd.read_csv(car_data, header = 0, skiprows=0, low_memory=False)


car_data.replace('-',np.nan, inplace=True) #replaces '-' with Nan values
car_data['Levy'] = car_data['Levy'].astype('float64')

car_data['Mileage'] = car_data['Mileage'].str.extract('(\d+)').astype(int) # This is going to remove the 'km' in the mileage
car_data['Mileage'] = car_data['Mileage'].astype('int64')

car_data['Leather interior'] = car_data['Leather interior'].replace({'Yes': True, 'No': False})#replace 'Leather interior yes/no with T/F

car_data['Turbo'] = car_data['Engine volume'].str.contains('Turbo') #place turbo in separate new column with T/F.

car_data['Engine volume'] = car_data['Engine volume'].str.extract(r'(\d+\.\d+|\d+)').astype(float) # remove turbo from engine type, 
car_data['Engine volume'] = car_data['Engine volume'].astype('float64')
car_data['Doors'].replace({'04-May':4, '02-Mar':2, '>5':5}, inplace=True) #replace doors dates with 2,4,5

car_data

#%%
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



imputer = KNNImputer(n_neighbors=10)
num_cars = imputer.fit_transform(num_cars)
scaler = StandardScaler()
num_cars = scaler.fit_transform(num_cars)


