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
from sklearn.feature_selection import RFECV
from sklearn.model_selection import KFold

dataset = pd.read_csv('weather_pv_2017_without_time.csv',sep=',')
new_dataset = dataset.convert_objects(convert_numeric=True)
print(new_dataset.dtypes)

"""
Standardization
"""

scaler = StandardScaler()
print(scaler.fit(new_dataset))
print(scaler.transform(new_dataset))


X = new_dataset[['WindSpeed','Sunshine','AirPressure','Radiation','AirTemperature','RelativeAirHumidity']]
y = new_dataset['SystemProduction']

#Create the RFE object and compute a cross-validated score.
#svr = SVR(kernel="linear")
lasso = Lasso(alpha=0.1)
#ridge = Ridge(alpha=1.0)
#the "neg_mean_absolute_error" is the mean absolute error 
#rfecv = RFECV(estimator=svr, step=1, cv=KFold(2), scoring='neg_mean_absolute_error')
#rfecv = RFECV(estimator=svr, step=1, cv=KFold(2), scoring='r2')
#rfecv = RFECV(estimator=svr, step=1, cv=KFold(2), scoring='explained_variance')
#rfecv = RFECV(estimator=svr, step=1, cv=KFold(2), scoring='neg_mean_squared_error')
rfecv = RFECV(estimator=lasso, step=1, cv=KFold(2), scoring='r2')
#rfecv = RFECV(estimator=ridge, step=1, cv=KFold(2), scoring='r2')
rfecv.fit(X,y)

print("optimal number of features : %d" % rfecv.n_features_)
print ("The cross validation scores (R2):", rfecv.grid_scores_)


#plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
#plt.ylabel("Cross validation score (MAE)")
plt.ylabel("Cross validation score (r2)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()



