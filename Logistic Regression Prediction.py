# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 22:15:27 2019

Program: Logistic Regression
Author: Pramodkumar Gupta

"""

# Import all libraries
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings

# Disable all warning to display
warnings.filterwarnings("ignore")

# Read data into pandas dataframe

df=pd.read_csv('Dataset/Social_Network_Ads.csv')

# Slice the data in indepedent and depedent variables
X=df.iloc[:,[2,3]].values
y=df.iloc[:,4].values

# Divide data in train and test
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=0)


# Feature scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)


# Model Building for Logistic Regression 
from sklearn.linear_model import LogisticRegression
model=LogisticRegression(random_state=0)
model.fit(X_train,y_train)
y_pred=model.predict(X_test)

# Evaluation of Model
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
cm=confusion_matrix(y_test,y_pred)
acc=accuracy_score(y_test,y_pred)
report=classification_report(y_test,y_pred)

# Print all Matrix
print('Confusion Matrix: ',cm)
print('')
print('Accuracy (%): ', acc)
print('')
print("Classification Report: ")
print(report)
