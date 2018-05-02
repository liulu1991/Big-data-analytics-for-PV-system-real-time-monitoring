from sklearn.svm import SVR
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn import datasets
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE

dataset = pd.read_csv('weather_pv_2017_without_time.csv',sep=',')
new_dataset = dataset.convert_objects(convert_numeric=True)
print(new_dataset.dtypes)

#Standardization
scaler = StandardScaler()
scaler.fit(new_dataset)
scaler.transform(new_dataset)

X = new_dataset[['WindSpeed','Sunshine','AirPressure','Radiation','AirTemperature','RelativeAirHumidity']]
y = new_dataset['SystemProduction']

#define estimator
estimator = SVR(kernel='linear')

#define feature selector
selector = RFE(estimator, 1, step=1) 
selector = selector.fit(X,y)
print (selector.support_)
print (selector.ranking_)


#estimator = Lasso(alpha=0.1)
#estimator = Ridge(alpha=1.0)
