# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 09:08:25 2021
Test of the class Perceptron which performs the Perceptron algorithm.

@author: Marcel Pommer
"""
#%%
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
import math
import matplotlib.pyplot as plt
import sys
import os
path = 'C:/Users/marce/Documents/Dokumente/Python Scripts/machine learning projekte/Own algorithms/Machine-Learning-in-Python/Perceptron Algorithm'
os.chdir(path)
#pd.options.mode.chained_assignment = None 

#%%
def computePredictions(X_train,slope,intercept):
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



#%%
# I want to change some labels that we have a binary classification problem
iris_dataset = pd.DataFrame(load_iris().data)
target = load_iris().target
flower_names = load_iris().target_names

iris_dataset.columns = load_iris().feature_names
iris_dataset['target'] = target



#%%
plt.scatter(iris_dataset['sepal length (cm)'][0:50],
   iris_dataset['sepal width (cm)'][0:50])
plt.scatter(iris_dataset['sepal length (cm)'][50:100],
   iris_dataset['sepal width (cm)'][50:100])
plt.scatter(iris_dataset['sepal length (cm)'][100:150],
   iris_dataset['sepal width (cm)'][100:150])
plt.legend(['0', '1', '2'])
plt.show()

#%%

# we can see in the plot above that we can seperate between 0("setosa") and not 0 (versicular and verginica)
iris_dataset.loc[ iris_dataset['target'] ==2, 'target'] = 1

plt.scatter(iris_dataset['sepal length (cm)'][0:50],
   iris_dataset['sepal width (cm)'][0:50])
plt.scatter(iris_dataset['sepal length (cm)'][50:150],
   iris_dataset['sepal width (cm)'][50:150])

plt.legend(['0', '1'])
plt.show()



#%%

# we got the right dataset
# lets add some lines
a = 1.5
b = -5
line = lambda x: a * x + b

plt.scatter(iris_dataset['sepal length (cm)'][0:50],
   iris_dataset['sepal width (cm)'][0:50], label = '0', marker = "+")
plt.scatter(iris_dataset['sepal length (cm)'][50:150],
   iris_dataset['sepal width (cm)'][50:150], label = '1', marker = "D")
plt.plot([4,8],[line(4),line(8)], '-r', label='y=2x+1')
plt.legend()
plt.xlim([4,8])
plt.ylim([1,5])
plt.show()

# now see what this regression line would make to our dataset
x1,x2 = 4,8
y1,y2 = line(4), line(8)
regression_line = [{x1,y1}, {x2,y2}]
# first test
xA = iris_dataset['sepal length (cm)'][0]
yA = iris_dataset['sepal width (cm)'][0]
v1 = (x2-x1, y2-y1)   # Vector 1
target_predict = []

for i in range(len(iris_dataset)):
    xA = iris_dataset['sepal length (cm)'][i]
    yA = iris_dataset['sepal width (cm)'][i]
    v2 = (x2-xA, y2-yA)   # Vector 2

    xp = v1[0]*v2[1] - v1[1]*v2[0]  # Cross product

    if xp > 0:
        target_predict.append(1)
    elif xp < 0:
        target_predict.append(0)
    else:
        target_predict.append(1)

iris_dataset['target_predicted'] = target_predict



#%%

# how many mispredictions?
comparison = iris_dataset['target_predicted'] == iris_dataset['target']
print(comparison.value_counts())
# only 2 wrong predictiosn
# we already got a very good estimation










#%%
# lets test the perceptron function

from Perceptron import Perceptron
# change target variable from 0 to -1
iris_dataset.loc[ iris_dataset['target'] ==0, 'target'] = -1

X_train = iris_dataset[['sepal length (cm)','sepal width (cm)']]
y_train = iris_dataset['target']

model = Perceptron(X_train,y_train)

# before I let the algorithm do its magic I want to guess myself
a = 1.5
b = -5
model.plotPredictionsByOwnGuessUsingLinearFunction(slope = a, intercept = b,
                xlabel = 'Sepal width', ylabel = ' sepal length',
    title = 'Binary classification with Perceptron slope = 1.5, b=-5',
                addSeperationLine = True)

#%% next lets try the perceptron algorithm

# first I guess w and b
w = [40,-40]
b = -60
model.plotPredictionsByOwnGuess(w = w,b = b,xlabel = 'Sepal width', ylabel = ' sepal length',
    title = 'Binary classification with Perceptron w=[40,-40], b=-60', addSeperationLine= True)
# pretty bad! But guessing is not easy with those w and b

model.performRegression()
predictions = model.predictWithModel(X_train)

print('Accuracy with the Perceptron model: ', model.getAccuracy(X_train, y_train))

model.plotPredictions(X_train,xlabel = 'Sepal width', ylabel = ' sepal length',
    title = 'Binary classification with Perceptron', addSeperationLine=True)




#%% lets try our model with less training data but independent test data
X = iris_dataset[['sepal length (cm)','sepal width (cm)']]
y = iris_dataset['target']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,shuffle=True, test_size = 0.2)

model.plotTrainingAndTestWithPerceptron(X_train, X_test, y_train, y_test,
                                        addSeperationLine=True)   






#%% ADALINE Linear sctivation function
X = iris_dataset[['sepal length (cm)','sepal width (cm)']]
y = iris_dataset['target']

from AdalineAlgorithm import ADALINELinear

model = ADALINELinear(X,y)
w_initial, b_initial = [0.5,-0.5], -1
model.performRegression(w_initial = w_initial, b_initial = b_initial,
                        learning_rate=0.00005, numberOfIterations=500)
model.accuracy(X,y)
model.plotPredictions(X, addSeperationLine = True)




#%% ADALINE sigmoid activation function
from AdalineSigmoid import ADALINESigmoid
model = ADALINESigmoid(X,y)

model.performRegression(numberOfIterations=1000, learning_rate = 0.0001, printPogress = True)
model.accuracy(X,y)
model.loss()
model.plotPredictions(X, addSeperationLine = True)




#%% Variable activation function
iris_dataset = pd.DataFrame(load_iris().data)
target = load_iris().target
flower_names = load_iris().target_names

iris_dataset.columns = load_iris().feature_names
iris_dataset['target'] = target
iris_dataset.loc[ iris_dataset['target'] ==2, 'target'] = 1
iris_dataset.loc[ iris_dataset['target'] ==0, 'target'] = -1
X = iris_dataset[['sepal length (cm)','sepal width (cm)']]
y = iris_dataset['target']
import matplotlib.animation as animation
import mpmath
mpmath.pretty = True
'''
data = [[1,2,1],[1,1,1],[0,1,1],[7,6,-1],[7,7,-1],[8,6,-1]]
dataframe = pd.DataFrame(data, columns = ['one','two','target'])
y = dataframe['target']
X = dataframe.drop('target', axis = 1)
'''

#function = lambda x: x
function = lambda x: 2*(mpmath.exp(x)/(1+mpmath.exp(x)))-1


from AdalineActivation import ADALINEActivation
model = ADALINEActivation(X,y, function)

model.performRegression(printProgress = True, learning_rate = 0.0005,test_size = 0,
                        numberOfIterations = 100, continue_fit = False, min_acc = 1)

model.plotPredictions(X, addSeperationLine = True)
model.plotEvolutionOfRegLine(X, iterations = 10, updatesPerIteration=50,
                             min_acc = 1)

model.accuracy(X,y)



def update():
    model.performRegression(printProgress = True, learning_rate = 0.0005,test_size = 0,
                        numberOfIterations = 30, continue_fit = True, min_acc = 1)
    x0, x1, y0, y1 = model.pointsRegLine()
    x = np.array([[x0, x1, y0, y1]])
    
    with open("regPoints.txt","w") as f:
        np.savetxt(f,x, fmt="%f")



fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
        
def animate(i):
    graph_data = X
    graph_data['target'] = y
    graph_data2 = open('regPoints.txt','r').read()
    lines = graph_data2.split('\n')
    update()
    
    xs = []
    ys = []
    for d in range(len(graph_data)):
        xs.append(graph_data.iloc[d,:][0])
        ys.append(graph_data.iloc[d,:][1])
    for line in lines:
        if len(line) > 1:
            x0, x1, y0, y1 = line.split(' ')            
            x0, x1, y0, y1 = float(x0), float(x1), float(y0), float(y1)
    print(x0, x1, y0, y1)
    ax1.clear()
    ax1.plot([x0, y0], [x1, y1] , '-r', label='Seperation function')
    ax1.scatter(graph_data.iloc[:,0],graph_data.iloc[:,1], c = graph_data['target'])
        
ani = animation.FuncAnimation(fig, animate, interval = 100, repeat=False)
plt.show()


