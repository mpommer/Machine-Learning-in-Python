# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 22:07:50 2021

@author: Marcel Pommer
"""

import numpy as np
import math

from LinearClassifier import LinearRegression


class ADALINESigmoid(LinearRegression): 
    def __init__(self, X, y):
        super().__init__(X, y)
        self.numberOfFeatures = X.shape[1]
        self.X = X
        self.y = y
        self.w = [0] * self.numberOfFeatures
        self.b = 0      
        
        
        
        
    def performRegression(self, w_initial= 0, b_initial=0, learning_rate=0.00005, 
        numberOfIterations=10000, min_acc=0.999, test_size = 0.2, countIterations = False):
        '''
        Performs the algorithm.

        Parameters
        ----------
        w_initial : TYPE, two dimensional array.
            DESCRIPTION. Initial value for the slope. The default is [0,0].
        b_initial : TYPE, one dimensional int/double/float
            DESCRIPTION. Initial value for the slope. The default is 0.
        learning_rate : TYPE, double.
            DESCRIPTION. Learning rate, usualy between 0.01 and 0.3. The default is 0.05.
        numberOfIterations : TYPE, integer
            DESCRIPTION. Max Number of Interations. The default is 1 Mio.
        min_acc : TYPE, double.
            DESCRIPTION. As soon as the minimum accuracy is reached the algorithm stops.
            The default is 0.999.

        Returns
        -------
        w : TYPE: Two dimensional double.
            DESCRIPTION. Slope for regression line.
        b : TYPE: Double
            DESCRIPTION. Intercept for regression line.

        '''
        if countIterations:
            count = 0
        X = self.X
        y = self.y
            
        if test_size != 0:
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X,y,shuffle=True, test_size = test_size)
        else:
            X_train = X
            y_train = y
            X_test = X
            y_test = y
        X_train = X_train.reset_index(drop=True)
        X_test = X_test.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)
        y_test = y_test.reset_index(drop=True)

        # initialize w and b
        if w_initial ==0:
            w,b = self.w, b_initial
        else:
            w, b = w_initial, b_initial
        
        

        for i in range(numberOfIterations):
            y_pred_train = self.predictYourselfSigmoid(X_train, w, b)
            y_pred_test = self.predictYourselfSigmoid(X_test, w, b)

        # get accuracy training and test set
            acc_train = np.sum(y_pred_train == y_train)/len(X_train)
            acc_test = np.sum(y_pred_test == y_test)/len(X_test)
        
            if(acc_train> min_acc and acc_test> 0.9):
                break
            if countIterations:
                count += 1

        
            wTimesXPlusB = []
            for i in range(len(X_train)):
                value = 0
                for j in range(len(w)):
                    value += w[j] * X_train.iloc[i,:][j]
                wTimesXPlusB.append(value + b)
            wTimesXPlusB = wTimesXPlusB   
                        
            gradientDecent =  abs(y_train - wTimesXPlusB) 
            testMulti = []
            for j in range(len(w)):
                sum = 0
                for i in range(len(X_train)):
                    sum += (y_train[i] - (4* self.sigmoid(wTimesXPlusB[i])-1))* X_train.iloc[:,j][i]\
                        * self.sigmoidDerivative(wTimesXPlusB[i])*4
                testMulti.append(sum)
                            
            for i in range(len(w)):                    
                w[i] = w[i] + learning_rate * testMulti[i]
            b = b + learning_rate * np.sum(gradientDecent)     
            print("W: ", w)
            print("b: ", b)
        self.w = w
        self.b = b
        if countIterations:
            return w,b, count
        
        return w,b        

        
    def predictYourselfSigmoid(self, X, w,b):
    
        target_predict = []
        for i in range(len(X)):
            sum = 0
            for j in range(len(w)):
                sum += w[j] * X.iloc[i,:][j]
            sum += b
            target_predict.append(np.sign(4*self.sigmoid(sum)-1).astype(int))

        return target_predict
            
            
            
    def sigmoid(self, x):
        sigmoid = lambda x: np.exp(x)/(1+np.exp(x))
        return sigmoid(x)
        
    def sigmoidDerivative(self, x):
        sigmoidDerivative = lambda x: self.sigmoid(x)*(1-self.sigmoid(x))
        return sigmoidDerivative(x)
        
    
    def accuracy(self, X, y):
        y_pred = self.predictYourselfSigmoid(X, self.w, self.b)
        acc = np.sum(y_pred == y)/len(X)
        return acc
        
        
        
        
