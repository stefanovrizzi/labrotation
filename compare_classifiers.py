# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 16:59:40 2019

@author: HIWI
"""

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

#IMPORT 'DATA' FROM RE-LOAD ANALYSIS BEFORE RUNNING THIS FILE

names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(1 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=2),
    #RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
    RandomForestClassifier(),
    #MLPClassifier(alpha=10),
    MLPClassifier(hidden_layer_sizes=(10)),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]

# preprocess dataset, split into training and test part
X_train, X_test, y_train, y_test = \
train_test_split(X, y, test_size=.3)

# iterate over classifiers
for name, clf in zip(names, classifiers):
    
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)

    print (name,':', round(score, 3))
    print ()