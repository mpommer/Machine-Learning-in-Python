# -*- coding: utf-8 -*-
"""
Created on Sun May  9 18:08:36 2021

@author: marce
"""

import numpy as np
import warnings
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

class knearestNeighbors:
    def __init__(self, data):
        self.data = data
        self.colors =['b','g','r','k']
        
    def fit(self, predictions, numberNearestNeighbors = 3):
        data = self.data
        labels = data.iloc[: , -1].drop_duplicates(keep='first', inplace=False)
        data_dict = {x:[] for x in labels}
        for index, i in data.iterrows():
            data_dict[i[data.shape[1]-1]].append(i[:-1].values)
        
        self.data_dict = data_dict
        
        if len(data_dict) >=numberNearestNeighbors:
            warnings.warn('K is set to value less than number of groups')
        
        predicted_group = []
        confidence = []
        # loop over all predictions
        for index in range(len(predictions)):
            predict = predictions.iloc[index].to_numpy()
            
            # get distances 
            distances = []
            for group in data_dict:
                for features in data_dict[group]:
                    # calculate euclidean distance
                    euclidean_distance = np.linalg.norm(np.array(features) - np.array(predict))
                    distances.append([euclidean_distance, group])
               
            # analyze the distance
            votes = [i[1] for i in sorted(distances)[:numberNearestNeighbors]]
            
            vote_result = Counter(votes).most_common(1)[0][0]
            
            predicted_group.append(vote_result)
            confidence.append(Counter(votes).most_common(1)[0][1]/numberNearestNeighbors)
    
        self.confidence = confidence
        self.predictions = predicted_group
        return predicted_group
    
    def getConfidence(self, predictions, numberNearestNeighbors=3, new_prediction = False):
        if new_prediction == True or self.confidence == None:
              self.fit(predictions = predictions, numberNearestNeighbors = numberNearestNeighbors)
        return self.confidence
    
    def getAccuracy(self, predictions, y_test, numberNearestNeighbors=3, new_prediction = False):
        if new_prediction == True or self.predictions == None:
            self.fit(predictions = predictions, numberNearestNeighbors = numberNearestNeighbors)
        
        pred = self.predictions
        correct = [True for i in range(len(pred)) if (pred[i]==y_test[i])]
        
        return len(correct)/len(pred)
    
    
    def createPlot(self, predictions):
        new_dic = self.data_dict
        colors = self.colors
        index = 0
        color_dic = {}
        for k, v in list(new_dic.items()):
            color_dic[k] = colors[index]
            new_dic[colors[index]] = new_dic.pop(k)
            index +=1
       
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(1,1,1)
        for index in new_dic:
            for i in new_dic[index]:
                plt.scatter(i[0],i[1], s= 50, color = index)
        
        for index in range(len(predictions)):
            plt.scatter(predictions.iloc[[index]].values[0][0],predictions.iloc[[index]].values[0][1],
                        s = 100, marker = '*', color = color_dic[self.predictions[index]])
        plt.show()
            
        

