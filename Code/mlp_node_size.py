import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score

dataset = pd.read_csv('weather_pv_2017_without_time.csv',sep=',')
new_dataset = dataset.convert_objects(convert_numeric=True)
print(new_dataset.dtypes)

"""
Standardization
"""

scaler = StandardScaler()
scaler.fit(new_dataset)
scaler.transform(new_dataset)

X = new_dataset[['WindSpeed','Sunshine','AirPressure','Radiation','AirTemperature','RelativeAirHumidity']]
y = new_dataset['SystemProduction']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state=3)
  
evs = []
r2 = []
node = []
for i in range(2,12):
    n = i*10
    mlp = MLPRegressor(activation='relu',solver='lbfgs', hidden_layer_sizes=(n), random_state=2)
    mlp.fit(X_train, y_train)
    y_pred= mlp.predict(X_test)
    evs.append(metrics.explained_variance_score(y_test, y_pred))
    r2.append(metrics.r2_score(y_test, y_pred))
    node.append(n)

print("evs:",evs)
print("r2:",r2)

plt.figure()
plt.plot(node, evs, label='evs', marker='*',mec='r',)
plt.plot(node, r2,label='r2', marker='o',mec='y',)
plt.xlabel('node number')
plt.ylabel('score')
plt.legend()
plt.savefig("score_for_nodes.png")

