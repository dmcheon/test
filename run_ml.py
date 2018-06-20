# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 20:18:21 2018
Run machine learning
@author: dclab
"""

import numpy as np
import pandas as pd
#importing the dataset
l_filename ='total_5_feature'
l = pd.read_csv(l_filename +'.csv', 
                header=None
                )
X = l.iloc[:, :13]
y = l.iloc[:, 13]
X = X.values
y = y.values


"""
# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)
"""
# Fitting SVM to the Training set
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

classifier = SVC(kernel = 'rbf', random_state = 0)
#classifier = KNeighborsClassifier(n_neighbors = 2, metric = 'minkowski', p = 2)
#classifier = GaussianNB()
#classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
#classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix

#K-fold cross validation
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

cv = KFold(n_splits=6, shuffle=True)
scores = np.zeros(6)


for i, (train, test) in enumerate(cv.split(X)):
    X_train = X[train,:]
    y_train = y[train]
    X_test = X[test,:]
    y_test = y[test]
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    print(y_pred)
    print(y_test)
    scores[i] = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print(scores[i])

print(scores.mean())



