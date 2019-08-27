# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 09:22:59 2019
Program: Simple Linear Regression
@author: Pramodkumar Gupta
"""

# Step 1: Import Libraries
import pandas as pd
import matplotlib.pyplot as plt

# Step 2: Read DataSet
slrdata = pd.read_csv('Dataset/SimpleLinearRegressionSalary.csv')


# Step 3: Split data based on depedent (y) and indepedent variables(X)
X=slrdata.iloc[:,:-1].values   # -1: Dropping last column
y=slrdata.iloc[:,1:].values

# Step 4: Split dataset into training and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.33,random_state=0)

# Step 5: Train linear Regression Model
from sklearn.linear_model import LinearRegression
simplereg = LinearRegression()
simplereg.fit(X_train,y_train)

y_test_p=simplereg.predict(X_test)


# Step 6: Plot the graph
plt.scatter(X_train,y_train,color='red',cmap='plasma')
plt.plot(X_train,simplereg.predict(X_train))
plt.title('Salary vs Experience Graph')
plt.xlabel('Experience (In Years)')
plt.ylabel('Salary (In $)')
plt.show()
