# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 17:34:53 2021

@author: marce
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
        regLine = lambda x: w[0] * x[0] + w[1]*x[1] + b
    
        target_predict = []

        for i in range(len(X_test)):
             xA = X_test.iloc[:,0][i]
             yA = X_test.iloc[:,1][i]
             target_predict.append(np.sign(regLine([xA,yA])).astype(int))

       

        return target_predict
    
    
    def linearRegression(self, w_initial=[0,0], b_initial=0, learning_rate=0.05, 
                     numberOfIterations=1000000, min_acc=0.999):
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
        predictions = self.predictWithModel(X)
        
        return (np.sum(predictions == y)/len(X))
    
    
    def plotPredictions(self, X, xlabel = '', ylabel = '', title = ''):
        
        X['target'] = self.predictWithModel(X)
        
        plt.scatter(X.iloc[:,0],X.iloc[:,1], c = X['target'])
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.show()
        
        
        
        
        






