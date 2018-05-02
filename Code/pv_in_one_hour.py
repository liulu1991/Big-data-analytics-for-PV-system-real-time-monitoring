import pandas as pd
import numpy as np

df = pd.DataFrame(pd.read_csv('2017solaredge_splice.csv',sep=','))
sum = 0
for i in range(len(df.index)):      
    if i%4 == 0:
       df['System Production (W)'][i]= sum + df['System Production (W)'][i]
       sum = 0  
    else:
       sum = sum + df['System Production (W)'][i]

for i in range(len(df.index)):      
    if i%4 != 0:
       df=df.drop([i])
       
df.to_csv('2017pvsolaredge_splice_1hour.csv',index=False)  
