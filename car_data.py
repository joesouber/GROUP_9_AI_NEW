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
<<<<<<< HEAD
#drive.mount('/content/drive')

=======
drive.mount('/content/drive')
>>>>>>> e1c568c740c92cc5c97b8b2e37df0630a1a0c644

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
#attempting to replace Nan values in levy colum using 

#car_data_cleaned.head()

bosh = car_data_cleaned['Levy'].isnull() #finding where the nans are true = Nan, false = no Nan
nan_indices = car_data_cleaned[bosh].index    #turning into index

<<<<<<< HEAD

#%%
## replacing the NAN values in the Levy column with values using Knearest neighbours
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
=======
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

>>>>>>> e1c568c740c92cc5c97b8b2e37df0630a1a0c644

#Now we get a r2 score of 78.3% with a test size of 33%. Use this predictor on the Nan dataset.
df_with_nans_NUMBERS = df_with_nans_NUMBERS.drop('Levy', axis=1)

Nan_pred = reg.predict(df_with_nans_NUMBERS)
df_with_nans['Levy'] = Nan_pred

Cleaned_and_final_data = pd.concat([df_with_nans, df_without_nans])

#%%One Hot Encoding
Cleaned_and_final_data_one_hot = pd.get_dummies(Cleaned_and_final_data, columns = ['Manufacturer','Model','Category','Fuel type','Gear box type','Drive wheels','Color','Leather interior','Wheel','Turbo'])
X = Cleaned_and_final_data_one_hot.drop('Price',axis=1)
y = Cleaned_and_final_data_one_hot['Price']
X = np.array(X)
y = np.array(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state = 1)
#%%Building Classifiers

#Linear regression model 
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
score = linear_model.score(X_test, y_test)
print('Linear regression model produces an accuracy of',score)

#Random forrest lukes method
random_model = RandomForestRegressor()
random_model.fit(X_train, y_train)
y_pred = random_model.predict(X_test)
score = r2_score(y_test,y_pred)
print('Random forrest regression model produces an accuracy of', score)

#Random forrest joes method 

random_model = RandomForestRegressor()
random_model.fit(X_train, y_train)
score = random_model.score(X_test, y_test)
print('Random forrest regression model produces an accuracy of', score)

#Decision tree 
Dtree_model = DecisionTreeRegressor(random_state=42)
Dtree_model.fit(X_train, y_train)
y_pred = Dtree_model.predict(X_test)
score = r2_score(y_test, y_pred)
print('Decision tree model produces an accuracy of',score)

#%%Trying to build DNN

from keras.models import Sequential
from keras.layers import Dense, Dropout
import time

start_time = time.time()

# Define the number of input features
input_dim = X_train.shape[1]

<<<<<<< HEAD
# Define the neural network architecture
model = Sequential()
model.add(Dense(512, input_dim=input_dim, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='linear'))
metrics=['mse', 'mae', 'mape', 'cosine']
# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy','mse','mae','mape'])

# Train the model
Model = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

end_time = time.time()
elapsed_time = end_time - start_time
print("Elapsed time: ", elapsed_time, " seconds")



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

