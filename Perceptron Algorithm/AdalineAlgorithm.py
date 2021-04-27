# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 17:02:28 2021
ADALINE algorithmus to classify data. The class uses a linear activation function 
f(x) = x.

For more complex activation functions see ADALINESigmoid or ADALINEActivation

@author: Marcel Pommer
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
pd.options.mode.chained_assignment = None 
from LinearClassifier import LinearRegression


class ADALINELinear(LinearRegression): 
    def __init__(self, X, y):
        '''
        

        Parameters
        ----------
        X : TYPE Data
            DESCRIPTION.
        y : TYPE target
            DESCRIPTION.

        Returns
        -------
        None.

        '''
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
        w_initial : TYPE, n dimensional array.
            DESCRIPTION. Initial value for the slope. The default is [0]^n.
        b_initial : TYPE, one dimensional int/double/float
            DESCRIPTION. Initial value for the intercept (bias). The default is 0.
        learning_rate : TYPE, double.
            DESCRIPTION. Learning rate, usualy between 0.000001 and 0.001. The default is 0.00005.
            Caution, if learning rate is to high, the algorithm jumps from peak to peak
            and does not converge!
        numberOfIterations : TYPE, integer
            DESCRIPTION. Max Number of Interations. The default is 10000.
        min_acc : TYPE, double.
            DESCRIPTION. As soon as the minimum accuracy is reached the algorithm stops.
            The default is 0.999.
        test_size : TYPE, double(or int).
            DESCRIPTION. Portion of data used for testing.
        countIterations : TYPE, boolean.
            DESCRIPTION. If set to true will count the number of iterations.
        Returns
        -------
        w : TYPE: N dimensional double.
            DESCRIPTION. Slope for regression line.
        b : TYPE: Double
            DESCRIPTION. Intercept (bias) for regression line.

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
            y_pred_train = self.predictYourselfLinear(X_train, w, b)
            y_pred_test = self.predictYourselfLinear(X_test, w, b)

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
                        
                        
            gradientDecent =  (y_train - wTimesXPlusB) 
            testMulti = []
            for j in range(len(w)):
                sum = 0
                for i in range(len(X_train)):
                    sum += (y_train[i] - wTimesXPlusB[i])* X_train.iloc[:,j][i]
                testMulti.append(sum)
                            
            for i in range(len(w)):                    
                w[i] = w[i] + learning_rate * testMulti[i]
            b = b + learning_rate * np.sum(gradientDecent)     

        self.w = w
        self.b = b
        if countIterations:
            return w,b, count
        
        return w,b        


    def predictYourselfLinear(self, X, w,b):
        '''
        Creats the predictions for w and b defined by the user.

        Parameters
        ----------
        X : TYPE Data frame with N features.
            DESCRIPTION.
        w : TYPE n dm array, slope.
            DESCRIPTION.
        b : TYPE Intercept (bias).
            DESCRIPTION.

        Returns
        -------
        target_predict : TYPE
            DESCRIPTION.

        '''
    
        target_predict = []
        for i in range(len(X)):
            sum = 0
            for j in range(X.shape[1]):
                sum += w[j] * X.iloc[i,:][j]
            sum += b
            target_predict.append(np.sign(sum).astype(int))
        return target_predict     
    
    
        
    
    def accuracy(self, X, y):
        y_pred = self.predictYourselfLinear(X, self.w, self.b)
        acc = np.sum(y_pred == y)/len(X)
        return acc
        
        
        
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
        
        X = X.copy(deep=True)
        x0, x1, xmin,xmax, ymin, ymax = self.__findZeroInFunction(X)
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
        
        
    def __findZeroInFunction(self, ownW = 0, ownB = 0, linear = False):
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
        
        if type(ownW) is int:
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
        
        
        
        
        
