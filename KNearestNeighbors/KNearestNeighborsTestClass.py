# -*- coding: utf-8 -*-
"""
Created on Sun May  9 11:19:00 2021

@author: Marcel Pommer
"""

import numpy as np
import warnings
import pandas as pd
from collections import Counter
import sys
sys.path
sys.path.append('C:/Users/marce/Documents/Dokumente/Python Scripts/machine learning\
                projekte/Own algorithms/Machine-Learning-in-Python/KNearestNeighbors')
import os
path = 'C:/Users/marce/Documents/Dokumente/Python Scripts/machine learning projekte/Own algorithms/Machine-Learning-in-Python/KNearestNeighbors'
os.chdir(path)
from KNearestNeighborsClassifier import knearestNeighbors

#%% first a simple task: create data
data = pd.DataFrame(np.array([[5,7,1],
                    [6,5,1],
                    [7,7,1],
                    [1,1,-1],
                    [2,3,-1],
                    [0,1,-1]]) )
# two predictions which obviusly belong to either one group
predictions = pd.DataFrame([[7,8],[0,3]])
# create classifier
clf = knearestNeighbors(data)
# fit to predictions
clf.fit(predictions)
# getConfidence
clf.getConfidence(predictions)
clf.createPlot(predictions)





#%% lets test the data for the iris data set
from sklearn.datasets import load_iris
from sklearn.utils import shuffle

iris_dataset = pd.DataFrame(load_iris().data)
target = load_iris().target
flower_names = load_iris().target_names

iris_dataset.columns = load_iris().feature_names
iris_dataset['target'] = target

iris_dataset.loc[ iris_dataset['target'] ==2, 'target'] = 1
iris_dataset_shuffle = shuffle(iris_dataset)
data_train = iris_dataset_shuffle[:-20]
data_test = iris_dataset_shuffle[-20:]
y_test = data_test['target'].reset_index(drop=True)
X_test = data_test.drop('target', axis = 1)

clf = knearestNeighbors(data_train)
clf.fit(X_test)
clf.getAccuracy(X_test,y_test)

clf.createPlot(X_test)

