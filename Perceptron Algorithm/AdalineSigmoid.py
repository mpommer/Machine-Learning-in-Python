# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 22:07:50 2021
ADALINE algorithmus to classify data. The class uses a sigmoid activation function 
f(x) = e^/(1+e^)

For more complex activation functions see ADALINEActivation.
For less complex activation functions see ADALINELinear.
@author: Marcel Pommer
"""

import numpy as np
import matplotlib.pyplot as plt

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
            
                        
            gradientDecent =  abs(y_train - (2* self.sigmoid(wTimesXPlusB) -1))\
                * self.sigmoidDerivative(wTimesXPlusB)*2
            testMulti = []
            for j in range(len(w)):
                sum = 0
                for i in range(len(X_train)):
                    sum += (y_train[i] - (2* self.sigmoid(wTimesXPlusB[i])-1))* X_train.iloc[:,j][i]\
                        * self.sigmoidDerivative(wTimesXPlusB[i])*2
                testMulti.append(sum)
                            
            for i in range(len(w)):                    
                w[i] = w[i] + learning_rate * testMulti[i]
            b = b + learning_rate * np.sum(gradientDecent)     

        self.w = w
        self.b = b
        if countIterations:
            return w,b, count
        
        return w,b        

        
    def predictYourselfSigmoid(self, X, w,b):
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
            for j in range(len(w)):
                sum += w[j] * X.iloc[i,:][j]
            sum += b
            target_predict.append(np.sign(2*self.sigmoid(sum)-1).astype(int))

        return target_predict
            
            
            
    def sigmoid(self, x):
        '''
        Sigmoid function.

        Parameters
        ----------
        x : TYPE float/array
            DESCRIPTION.

        Returns
        -------
        TYPE float/array
            DESCRIPTION. returns the value of the sigmoid(x)

        '''        
        
        sigmoid = lambda x: np.exp(x)/(1+np.exp(x))
        return sigmoid(x)
        
    def sigmoidDerivative(self, x):
        '''
        Sigmoid derivative function.

        Parameters
        ----------
        x : TYPE float/array
            DESCRIPTION.

        Returns
        -------
        TYPE float/array
            DESCRIPTION. returns the value of the dsigmoid(x)/dx

        '''
        sigmoidDerivative = lambda x: self.sigmoid(x)*(1-self.sigmoid(x))
        return sigmoidDerivative(x)
        
    
    def accuracy(self, X, y):
        y_pred = self.predictYourselfSigmoid(X, self.w, self.b)
        acc = np.sum(y_pred == y)/len(X)
        return acc
        
    def loss(self):
        '''
        Loss function is: (y-2*sigmoid(w*x+b))^2

        Returns
        -------
        TYPE float
            DESCRIPTION. returns the loss.

        '''
        
        
        X = self.X
        y = self.y
        w = self.w
        b = self.b
        
        wTimesXPlusB = []
        for i in range(len(X)):
             value = 0
             for j in range(len(w)):
                 value += w[j] * X.iloc[i,:][j]
             wTimesXPlusB.append(value + b)
        wTimesXPlusB = 2*self.sigmoid(wTimesXPlusB)-1
                        
        gradientDecent =  (y - wTimesXPlusB) **2 

        return np.sum(gradientDecent)
    
    

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
        plotTrueTarget : TYPE, boolean, specifies if the seperation (different color)
        is according to the true target or the regression target.        
            DESCRIPTION. The default is True.
            
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
        Calculates zero point for the regression function.

        Parameters
        ----------
        ownW : TYPE, n-dim array
            DESCRIPTION. slope
        ownB : TYPE, float
            DESCRIPTION. The default is 0.
        linear : TYPE, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        TYPE
            DESCRIPTION. Boundaries for the plot and 
            two points to plot the line.            

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
                
        
        
