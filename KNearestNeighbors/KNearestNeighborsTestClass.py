# -*- coding: utf-8 -*-
"""
Created on Sun May  9 11:19:00 2021

@author: Marcel Pommer
"""

import numpy as np
import warnings
import pandas as pd
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
