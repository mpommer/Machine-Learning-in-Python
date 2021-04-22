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
                     numberOfIterations=100000, min_acc=0.999, test_size = 0.2):
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
        X_train, X_test, y_train, y_test = train_test_split(X,y,shuffle=True, test_size = test_size)
    
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
        if w == [0,0]:
            self.linearRegression()
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
                        addSeperationLine = False, plotTrueTarget = True):
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

        if plotTrueTarget:
            X['target'] = self.y
        else:
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
        
        
    def plotPredictionsByOwnGuess(self, xlabel = '', ylabel = '', title = '', 
                        addSeperationLine = False, w = [0,0], b = 0, plotTrueTarget = True):
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
        slope : TYPE, array 2dim, float.
            DESCRIPTION. The default is [0,0].
        intercept : TYPE, float.
            DESCRIPTION. The default is 0.

        Returns
        -------
        Plot.

        '''
        
        X = self.X
        x0, x1, xmin, xmax, ymin, ymax = self.__findZeroInFunction(ownW = w, ownB = b)

        if plotTrueTarget:
            X['target'] = self.y
        else:
            X['target'] = self.predictYourself(X, slope, intercept)
        
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
        
        
    def plotPredictionsByOwnGuessUsingLinearFunction(self, xlabel = '', ylabel = '', title = '', 
        addSeperationLine = False, slope = 1, intercept = 0, plotTrueTarget = True):
        '''
        Scatter plot for the points of the data frame X and regression line 
        y = slope * x + b. Enables the user to guess the seperation line with
        slope and intercept instead of the more difficult to under perceptron 
        parameters w and b.

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
        slope : TYPE, float.
            DESCRIPTION. The default is [0,0].
        intercept : TYPE, float.
            DESCRIPTION. The default is 0.

        Returns
        -------
        Plot.

        '''
        
        X = self.X
        function = lambda x: x*slope + intercept
        xmin, xmax, ymin, ymax = self.__findZeroInFunction(linear=True)
        if plotTrueTarget:
            X['target'] = self.y
        else:
            X['target'] = self.computePredictionsLinearFunction(X, slope, intercept)

        
        plt.scatter(X.iloc[:,0],X.iloc[:,1], c = X['target'])
        if addSeperationLine:
            plt.plot([xmin, xmax], [function(xmin), function(xmax)] , '-r', label='Seperation function')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.xlim(xmin,xmax)
        plt.ylim(ymin,ymax)
        plt.show()
        
        
        
    def plotTrainingAndTestWithPerceptron(self,X_training, X_test, y_training, y_test,
                                          xlabel = '', ylabel = '',
            title = '', addSeperationLine = False, plotTrueTarget = True):
 
        '''
        Scatter plot for the points of the data frame X and regression line 
        calculated by perceptron. Dots are seperated by Train and test set.

        Parameters
        ----------
        X_training : TYPE data frame Xx2.
            DESCRIPTION. training data
        X_test : TYPE data frame Xx2.
            DESCRIPTION.Test data
        y_training : TYPE array.
            DESCRIPTION. True labels for training set.
        y_test : TYPE array.
            DESCRIPTION.True label for test set.
        xlabel : TYPE, optional
            DESCRIPTION. Label x-axis. The default is ''.
        ylabel : TYPE, optional
            DESCRIPTION. Label y-axis.The default is ''.
        title : TYPE, optional
            DESCRIPTION. Title of plot. The default is ''.
        addSeperationLine : TYPE, boolean.
            DESCRIPTION. If True, seperation line from perceptron is drawed. 
            The default is False.
        plotTrueTarget : TYPE, optional
            DESCRIPTION. Option to plot predictions or true tragets. The default is True.

        Returns
        -------
        None.

        '''
        
        if plotTrueTarget:
            X_training['target'] = y_training
            X_test['target'] = y_test
        else:
            X_training['target'] = self.predictWithModel(X)
            X_test['target'] = self.predictWithModel(X)
            
        x0, x1, xmin, xmax, ymin, ymax = self.__findZeroInFunction()
        
        plt.scatter(X_training.iloc[:,0],X_training.iloc[:,1], c = X_training['target'],
                    marker = "D", label ="training")
        plt.scatter(X_test.iloc[:,0],X_test.iloc[:,1], c = X_test['target'], 
                    marker = "+", label = "test")
        
        if addSeperationLine:
            plt.plot([x0[0], x1[0]], [x0[1], x1[1]] , '-r', label='Seperation function')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.xlim(xmin,xmax)
        plt.ylim(ymin,ymax)
        plt.show()
        
        
        
        
        
    def __findZeroInFunction(self, ownW = [0,0], ownB = 0, linear = False):
        '''
        Calculates zero point for the regression function

        Returns
        -------
        firstPoint : TYPE double array 2-dim
            DESCRIPTION. first point
        secondPoint : TYPE double arry 2-dim
            DESCRIPTION. second point

        '''
        X = self.X
        x_min = min(X.iloc[:,0]) - 1
        x_max = max(X.iloc[:,0]) + 1
        y_min = min(X.iloc[:,1]) - 1
        y_max = max(X.iloc[:,1]) + 1
        
        if linear:
            return x_min, x_max, y_min, y_max
        
        if ownW != [0,0]:
            w = ownW
        else:
            w = self.w
        if ownB != 0:
            b = ownB
        else:
            b = self.b
        

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

        
    def computePredictionsLinearFunction(self, X_train,slope,intercept):
        '''
        Function to compute predictions for a linear function with slope and 
        intercept given by the user.

        Parameters
        ----------
        X_train : TYPE: Data Frame with two columns.
            DESCRIPTION.
        slope : TYPE: double 
            DESCRIPTION. Slope of the function
        intercept : TYPE: double
            DESCRIPTION. Interception of function.

        Returns
        -------
        target_predict : Array.
            DESCRIPTION. returns the array with the predictions. (0,1)

        '''
        regLine = lambda x: slope * x + intercept
        x1, x2 = 0, 1
        y1, y2 = regLine(0), regLine(1)
    
        v1 = (x2-x1, y2-y1)   # Vector 1
        target_predict = []

        for i in range(len(X_train)):
            xA = X_train.iloc[:,0][i]
            yA = X_train.iloc[:,1][i]
            v2 = (x2-xA, y2-yA)   # Vector 2

            xp = v1[0]*v2[1] - v1[1]*v2[0]  # Cross product

            if xp > 0:
                target_predict.append(1)
            elif xp < 0:
                target_predict.append(0)
            else:
                target_predict.append(1)

        return target_predict        
        
        
        






