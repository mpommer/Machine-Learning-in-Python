# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 16:34:44 2021

@author: Marcel Pommer
"""


import abc

class LinearRegression(metaclass=abc.ABCMeta):
    
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        
        
        
    @abc.abstractmethod        
    def performRegression(self):
        '''
        performs the regression algorithm

        '''
        
    @abc.abstractmethod
    def accuracy(self, X, y):
        '''
        returns accuarcy of the model for the data X and labels y.
        '''
        
        
    def returnDataSet(self):
        return self.X_train, self.y_train
        
        
        