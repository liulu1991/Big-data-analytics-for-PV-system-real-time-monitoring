from sklearn.svm import SVR
from sklearn import datasets
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
import numpy as np
from sklearn.preprocessing import StandardScaler

dataset = pd.read_csv('weather_pv_2017_without_time.csv',sep=',')
new_dataset = dataset.convert_objects(convert_numeric=True)
print(new_dataset.dtypes)

"""
Standardization
"""

scaler = StandardScaler()
print(scaler.fit(new_dataset))
print(scaler.mean_)
print(scaler.transform(new_dataset))


X = new_dataset[['WindSpeed','Sunshine','Radiation','AirTemperature','RelativeAirHumidity']]
y = new_dataset['SystemProduction']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state=3)

"""
parameter selection
"""
GridSearch = GridSearchCV(SVR(kernel='rbf'),
                   param_grid={"C": [1e2,1e3,1e4],
                               "gamma": [1e-5,1e-4,1e-3]}, scoring = 'explained_variance')
GridSearch.fit(X_train, y_train)

best_parameters = GridSearch.best_params_
print ("The best value of parameters are:", GridSearch.best_params_, "The score of explained variance: ", GridSearch.best_score_)


