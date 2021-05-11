# -*- coding: utf-8 -*-
"""
Created on Sun May  9 18:08:36 2021
K nearest neighbors classifier in order to classify unknown data
into one of n classes. 
The class can predict, plot those predctions and give the confidence.

@author: Marcel Pommer
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
        '''
        Fits the model.

        Parameters
        ----------
        predictions : TYPE: array or data frame with the same amount 
        of features as the data set.
            DESCRIPTION.
        numberNearestNeighbors : TYPE, integer, usualy between 3 and 7. 
        Should be choosed wisely, such that there is no tie.
            DESCRIPTION. The default is 3.

        Returns
        -------
        predicted_group : the label of the predicted class
            DESCRIPTION.

        '''
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
        '''
        calculates the confidence of the predictions (value between 0 and 1).

        Parameters
        ----------
        predictions : features to predit.
            DESCRIPTION.
        numberNearestNeighbors : TYPE, optional
            DESCRIPTION. The default is 3.
        new_prediction : TYPE, boolean.
            DESCRIPTION. If the predictions are the same which are used before in 
            the fit method, the calculation speed can be increased. The default is False.

        Returns
        -------
        TYPE
            DESCRIPTION.

        '''
        
        if new_prediction == True or self.confidence == None:
              self.fit(predictions = predictions, numberNearestNeighbors = numberNearestNeighbors)
        return self.confidence
    
    def getAccuracy(self, predictions, y_test, numberNearestNeighbors=3, new_prediction = True):
        '''
        Calculates the accuracy in the sense of coorect/total. 
        Not to be confused with the confidence.

        Parameters
        ----------
        predictions : TYPE feature variables
            DESCRIPTION.
        y_test : TYPE correct labels
            DESCRIPTION.
        numberNearestNeighbors : TYPE, optional
            DESCRIPTION. The default is 3.
        new_prediction : TYPE, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        TYPE
            DESCRIPTION.

        '''
        if new_prediction == True or self.predictions == None:
            self.fit(predictions = predictions, numberNearestNeighbors = numberNearestNeighbors)
        
        pred = self.predictions
        correct = [True for i in range(len(pred)) if (pred[i]==y_test[i])]
        
        return len(correct)/len(pred)
    
    
    def createPlot(self, predictions):
        '''
        Creates the plot for the predictions.

        Parameters
        ----------
        predictions : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        
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
            
        

