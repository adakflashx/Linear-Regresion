# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 23:07:28 2018

@author: user
"""
# import library
import pandas as pd
import matplotlib.pyplot as plt

# import data
df = pd.read_csv("multiple-linear-regression-dataset.csv",sep = ";")

# plot data
plt.scatter(df.Deneyim,df.maas)
plt.xlabel("Deneyim")
plt.ylabel("maas")
plt.show()

#%% linear regression

# sklearn library
from sklearn.linear_model import LinearRegression

# linear regression model
linear_reg = LinearRegression()

x = df.Deneyim.values.reshape(-1,1)
y = df.maas.values.reshape(-1,1)

linear_reg.fit(x,y)

y_head = linear_reg.predict(x)

plt.plot(x,y_head,color ="red")
#%%
from sklearn.metrics import r2_score
print("r_square:",r2_score(y,y_head))








