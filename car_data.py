#%% IMPORTS

import random
import matplotlib.pyplot as plt
import numpy as np
import csv
import pandas as pd

#%% PUT THE CAR_PRICE_PREDICTION FILE IN THE SAME FOLDER AS THE AI FOLDER

car_data = "car_price_prediction.csv"
car_data = pd.read_csv(car_data, header = 0, skiprows=0, low_memory=False)
car_data.head()
car_data.columns

#%% CREATE SUBSETS FOR TRAINING/TESTING
# Identify the unique car manufacturers in the dataset
car_manufacturer = car_data['Manufacturer'].unique()

# Create an empty dictionary to hold the subsets
subsets = {i: pd.DataFrame() for i in range(10)}

# For each make, randomly select a proportionate number of rows to include in each subset
for make in car_manufacturer:
    make_df = car_data[car_data['Manufacturer'] == make]
    make_size = len(make_df)
    indices = np.arange(make_size)
    np.random.shuffle(indices)
    for i in range(10):
        subset_size = make_size // 10
        start = i * subset_size
        end = (i + 1) * subset_size
        subset_indices = indices[start:end]
        subset = make_df.iloc[subset_indices]
        subsets[i] = pd.concat([subsets[i], subset])

# Save each subset to a separate file
for i in range(10):
    subsets[i].to_csv(f'path/to/subset_{i}.csv', index=False)  # maybe need to remove the 'path/to/' to run locally 

#%% DATA EXPLORATION

car_data.shape   #size of the data frame
car_data.info()    #columns and the data type
car_data.isnull().sum()   #any NaN values or empty cells
car_data.select_dtypes(include=['object']).value_counts()   #checking for unique 

#separate object columns 
object_columns = car_data.select_dtypes(include=['object'])
#separate numerical columns
numeric_columns = car_data.select_dtypes(include=['int', 'float']).drop(['Price'], axis = 1)
#create a target column - we are training to predict price
target = car_data.Price