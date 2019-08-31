# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 13:41:34 2019
Program: Simple Linear Regression
Author: Pramodkumar Gupta

"""

# Import all libraries
import matplotlib.pyplot as plt
import pandas as pd

# Read data into pandas dataframe

df=pd.read_csv('Dataset/Position_Salaries.csv')

# Slice the data in indepedent and depedent variables

X=df.iloc[:,1:2].values              
y=df.iloc[:,2].values 

# Linear Regession classifier 
from sklearn.linear_model import LinearRegression
lr1=LinearRegression()
lr1.fit(X,y)

# Predict using Linear Regression
X_p = lr1.predict(X)

# Polynomial Regression
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=4)
X_poly=poly.fit_transform(X)

lr2=LinearRegression()
lr2.fit(X_poly,y)

# Predict using Polynomial Linear Regression
x_p2=lr2.predict(X_poly) 


# Plotting the graph
plt.scatter(X,y,color='blue')
plt.plot(X,X_p,color='red' )              # Red - Linear Regression
plt.plot(X,x_p2,color='green' )           # Green - Polynomial Regression
plt.title('Expireince Salary Graph')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.figure(figsize=(12,7))
plt.show()
