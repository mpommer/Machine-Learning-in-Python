# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 17:34:53 2021
The class Perceptron performes the Perceptron algorithm. The Perceptron 
algorithm is a classification algorithm with a binary label, preferebly {-1,1} 
and two features.
@author: Marcel Pommer
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Perceptron:    
    
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.w = [0,0]
        self.b = 0
        
        
        
    def predictYourself(self, X_test, w, b): 
        '''
        Offers the option calculate the predictions with user specified 
        slope w and intercept b.

        Parameters
        ----------
        X_test : Data to be tested.
        w : slope (2 dimensional).
        b : intercept.

        Returns
        -------
        target_predict : vector of predictions.

        '''
        regLine = lambda x: w[0] * x[0] + w[1]*x[1] + b
    
        target_predict = []

        for i in range(len(X_test)):
             xA = X_test.iloc[:,0][i]
             yA = X_test.iloc[:,1][i]
             target_predict.append(np.sign(regLine([xA,yA])).astype(int))

       

        return target_predict
    
    
    def linearRegression(self, w_initial=[0,0], b_initial=0, learning_rate=0.05, 
                     numberOfIterations=1000000, min_acc=0.999):
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
        X = self.X
        y = self.y
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X,y,shuffle=True, test_size = 0.2)
    
        X_train = X_train.reset_index(drop=True)
        X_test = X_test.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)
        y_test = y_test.reset_index(drop=True)

        # compute accuracy
        w, b = w_initial, b_initial
    
    
        for i in range(numberOfIterations):
            y_pred_train = self.predictYourself(X_train, w, b)
            y_pred_test = self.predictYourself(X_test, w, b)

        # get accuracy training and test set
            acc_train = np.sum(y_pred_train == y_train)/len(X_train)
            acc_test = np.sum(y_pred_test == y_test)/len(X_test)
        
            if(acc_train> min_acc and acc_test> 0.9):
                break

        # update rule
            delta = y_train - y_pred_train
        
            for j in range(len(delta)):
                w[0] = w[0] + learning_rate*delta[j]*X_train.iloc[:,0][j]
                w[1] = w[1] + learning_rate*delta[j]*X_train.iloc[:,1][j]
            
                b = b + learning_rate * delta[j]      
        
        self.w = w
        self.b = b
        return w,b
    
    
    
    def predictWithModel(self, X):
        '''
        Calculates the predictions with the paramater slope and Intercept defined 
        by the algorithm
        
        Parameters
        ----------
        X_test : Data to be tested.
        w : slope (2 dimensional).
        b : intercept.

        Returns
        -------
        target_predict : vector of predictions.

        '''
        
        w = self.w
        b= self.b
        
        regLine = lambda x: w[0] * x[0] + w[1]*x[1] + b
    
        target_predict = []

        for i in range(len(X)):
             xA = X.iloc[:,0][i]
             yA = X.iloc[:,1][i]
             target_predict.append(np.sign(regLine([xA,yA])).astype(int))
             
        return target_predict
    
    
    def getAccuracy(self, X, y):
        '''
        Calculates the accuracy of the predictions.

        Parameters
        ----------
        X : TYPE, Data Frame with two columns.
            DESCRIPTION.
        y : TYPE target array.
            DESCRIPTION.

        Returns
        -------
        TYPE double
            DESCRIPTION. Accuracy of the algorithm.

        '''
        predictions = self.predictWithModel(X)
        
        return (np.sum(predictions == y)/len(X))
    
    
    def plotPredictions(self, X, xlabel = '', ylabel = '', title = '', 
                        addSeperationLine = False):
        '''
        Scatter plot for the points of the data frame X.

        Parameters
        ----------
        X : TYPE
            DESCRIPTION: Data Frame with two columns.
        xlabel : TYPE, optional
            DESCRIPTION. The default is ''.
        ylabel : TYPE, optional
            DESCRIPTION. The default is ''.
        title : TYPE, optional
            DESCRIPTION. The default is ''.
        addSeperationLine : TYPE, boolean, if True adds seperation line
            DESCRIPTION. The default is False.

        Returns
        -------
        Plot.

        '''
        
        x0, x1, xmin, xmax, ymin, ymax = self.__findZeroInFunction()
        X['target'] = self.predictWithModel(X)
        
        plt.scatter(X.iloc[:,0],X.iloc[:,1], c = X['target'])
        if addSeperationLine:
            plt.plot([x0[0], x1[0]], [x0[1], x1[1]] , '-r', label='Seperation function')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.xlim(xmin,xmax)
        plt.ylim(ymin,ymax)
        plt.show()
        
        
        
        
        
    def __findZeroInFunction(self):
        '''
        Calculates zero point for the regression function

        Returns
        -------
        firstPoint : TYPE double array 2-dim
            DESCRIPTION. first point
        secondPoint : TYPE double arry 2-dim
            DESCRIPTION. second point

        '''
        w = self.w
        b = self.b
        X = self.X
        x_min = min(X.iloc[:,0]) - 1
        x_max = max(X.iloc[:,0]) + 1
        y_min = min(X.iloc[:,1]) - 1
        y_max = max(X.iloc[:,1]) + 1

        x0 = 0
        y0 = -b/w[1]
        x1 = -b/w[0]
        y1 = 0

        if x0 > x1:
            slope = (y0 - y1)/(x0-x1)
        else:
            slope = (y1 - y0)/(x1-x0)
        
        intercept = y0
        function = lambda x: slope * x + intercept
        firstPoint = []
        firstPoint.append(x_min)
        firstPoint.append(function(x_min))
        secondPoint = []
        secondPoint.append(x_max)
        secondPoint.append(function(x_max))        
        
        
              
        return firstPoint, secondPoint, x_min, x_max, y_min, y_max

        
        
        
        
        






