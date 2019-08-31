# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 15:38:51 2019

Program: Simple Linear Regression
Author: Pramodkumar Gupta

"""

# Import all libraries
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Read data into pandas dataframe

df=pd.read_csv('Dataset/Position_Salaries.csv')

# Slice the data in indepedent and depedent variables

X=df.iloc[:,1:2].values              
y=df.iloc[:,2].values 


from sklearn.ensemble import RandomForestRegressor
reg = RandomForestRegressor(n_estimators=100,random_state=0)
reg.fit(X,y)

y_pred = reg.predict(np.asarray(6.5).reshape(1,1))


# Plotting the graph
x_grid=np.arange(min(X),max(X),0.01)
X_grid=x_grid.reshape((len(x_grid),1))
plt.scatter(X,y,color='blue')
plt.plot(X_grid,reg.predict(X_grid),color='red' )              
plt.title('Expireince Salary Graph')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()