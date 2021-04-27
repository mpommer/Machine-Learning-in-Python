# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 16:34:44 2021
Interface for the implementation of linear classifier
@author: Marcel Pommer
"""


import abc

class LinearRegression(metaclass=abc.ABCMeta):
    
    def __init__(self, X_train, y_train):
        '''
        Initialize with data.

        Parameters
        ----------
        X_train : TYPE Data set
            DESCRIPTION.
        y_train : TYPE target vector
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        
        self.X_train = X_train
        self.y_train = y_train
        
        
        
    @abc.abstractmethod        
    def performRegression(self):
        '''
        Performs the regression of the algorithm.

        '''
        
        
        
    def returnDataSet(self):
        '''
        returns the raw data.

        Returns
        -------
        TYPE
            DESCRIPTION.
        TYPE
            DESCRIPTION.

        '''
        return self.X_train, self.y_train
        
        
        