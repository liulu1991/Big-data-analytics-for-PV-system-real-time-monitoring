from sklearn.svm import SVR
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import time

dataset = pd.read_csv('weather_pv_2017_without_time.csv',sep=',')
new_dataset = dataset.convert_objects(convert_numeric=True)
print(new_dataset.dtypes)

"""
Standardization
"""

scaler = StandardScaler()
scaler.fit(new_dataset)
scaler.transform(new_dataset)

X = new_dataset[['WindSpeed','Sunshine','Radiation','AirTemperature','RelativeAirHumidity']]
y = new_dataset['SystemProduction']

svr = SVR(kernel='rbf', C = 1e3, gamma = 1e-4)

"""
Cross validation
"""
predicted = cross_val_predict(svr, X, y, cv = 3)
explained_variance_score = cross_val_score(svr, X, y,cv=3,scoring='explained_variance')
r2 = cross_val_score(svr, X, y, cv=3, scoring='r2')
mean_squared_error = cross_val_score(svr, X, y, cv=3, scoring='neg_mean_squared_error')
print ("EVS_CV:",explained_variance_score.mean())
print ("r2_CV:",r2.mean())
print ("MSE_CV:",mean_squared_error.mean())

"""
 Test/Evaluation
"""
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state=3)
svr.fit(X_train, y_train)
y_pred= svr.predict(X_test)
print ("EVS:", metrics.explained_variance_score(y_test, y_pred))
print ("r2:", metrics.r2_score(y_test, y_pred))
print ("MSE:", metrics.mean_squared_error(y_test, y_pred))

"""
visualization
"""
fig, ax = plt.subplots()
ax.scatter(y,predicted, edgecolors=(0, 0, 0))
ax.plot([y.min(), y.max()], [predicted.min(), predicted.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.savefig("cv_svr.png")

fig, ax = plt.subplots()
ax.scatter(y_test, y_pred, edgecolors=(0, 0, 0))
ax.plot([y_test.min(), y_test.max()], [y_pred.min(), y_pred.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.savefig("test_svr.png")

