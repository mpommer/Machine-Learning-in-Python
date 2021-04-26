# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 17:02:28 2021

@author: Marcel Pommer
"""
import numpy as np
import math

from LinearClassifier import LinearRegression


class ADALINE(LinearRegression): 
    def __init__(self, X, y):
        self.numberOfFeatures = len(X)
        self.X = X
        self.y = y
        self.w = [0] * self.numberOfFeatures
        self.b = 0      
        
        super.__init__(self, X, y)
        
        
    def performRegression(self, w_initial= 0, b_initial=0, learning_rate=0.05, 
        numberOfIterations=100000, min_acc=0.999, test_size = 0.2, countIterations = False):
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

        # compute accuracy
        if w_initial ==0:
            w,b = self.w, b_initial
        else:
            w, b = w_initial, b_initial
    
    
        for i in range(numberOfIterations):
            y_pred_train = self.predictYourself(X_train, w, b)
            y_pred_test = self.predictYourself(X_test, w, b)

        # get accuracy training and test set
            acc_train = np.sum(y_pred_train == y_train)/len(X_train)
            acc_test = np.sum(y_pred_test == y_test)/len(X_test)
        
            if(acc_train> min_acc and acc_test> 0.9):
                break
            if countIterations:
                count += 1
        # update rule
        # sigmoid function
        sigmoid = lambda x: np.exp(x)/(1+np.exp(x))
        sigmoidDerivative = lambda x: sigmoid(x)*(1-sigmoid(x))
        
        wTimesXPlusB = []
        for i in range(len(X_train)):
            value = 0
            for j in range(len(w)):
                value += w[j] * X_train.iloc[i,:][j]
            wTimesXPlusB.append(value + b)

        gradientDecent =  -math.abs(y_train - sigmoid(wTimesXPlusB)) * sigmoidDerivative(wTimesXPlusB)
        
        for j in range(len(gradientDecent)):
            for i in range(self.numberOfFeatures):
                w[i] = w[i] - learning_rate*gradientDecent[j]*w[i]

            
                b = b - learning_rate * gradientDecent[j]*b      

                    
        self.w = w
        self.b = b
        if countIterations:
            return w,b, count
        
        return w,b        
        
        
        def predictYourself(self, X, w,b):

            regLine = lambda x: w[0] * x[0] + w[1]*x[1] + b
    
            target_predict = []
            for i in range(len(X)):
                sum = 0
                for j in range(X.shape[1]):
                    sum += w[j] * X.iloc[i,:][j]
                target_predict.append(np.sign(self.sigmoid(sum)).astype(int))



            return target_predict
            
            
            
        def sigmoid(self, x):
            sigmoid = lambda x: np.exp(x)/(1+np.exp(x))
            return sigmoid
        
        def sigmoidDerivative(self, x):
            sigmoidDerivative = lambda x: sigmoid(x)*(1-sigmoid(x))
            return sigmoidDerivative
        
        
        
        
        
        
        
        
        
        
        
