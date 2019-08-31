# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 00:12:41 2019

Program: SVM Classifier
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


# Model Building for SVM Classifier 
from sklearn.svm import SVC
model=SVC(kernel='linear', random_state=0)
model.fit(X_train, y_train)

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


# Visualize the test data 
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, model.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('SVM (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()




