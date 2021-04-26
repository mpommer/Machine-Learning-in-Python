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
import matplotlib.pyplot as plt

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

from LinearClassifier import LinearRegression

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


w = [-50,-50]
b = -50
c = w*b
compre = [i*j for i,j in zip(w,X.iloc[0,:][0])]

test = []
for i in range(len(X)):
    value = 0
    for j in range(len(w)):
        value += w[j] * X.iloc[i,:][j]
    test.append(value + b)

np.exp(w)

len(X)
X.shape[1]


def sigmoid(x):
     sigmoid = lambda x: np.exp(x)/(1+np.exp(x))
     value = sigmoid(x)
     return value

        
def sigmoidDerivative(x):
    sigmoidDerivative = lambda x: sigmoid(x)*(1-sigmoid(x))
    return sigmoidDerivative(x)

target_predict = []
for i in range(len(X)):
    sum = 0
    for j in range(X.shape[1]):
        sum += w[j] * X.iloc[i,:][j]
    target_predict.append(np.sign(sigmoid(sum)-0.5).astype(int))

j = 0


print(sigmoid(-10))


X.iloc[:,0]


from AdalineAlgorithm import ADALINE

model = ADALINE(X,y)

model.performRegression()
model.accuracy(X,y)


from AdalineSigmoid import ADALINESigmoid

model = ADALINESigmoid(X,y)

model.performRegression(numberOfIterations=20000)
model.accuracy(X,y)




wTimesXPlusB = []
for i in range(len(X)):
    value = 0
    for j in range(len(w)):
        value += w[j] * X.iloc[i,:][j]
    wTimesXPlusB.append(value + b)

gradientDecent =  -abs(y - sigmoid(wTimesXPlusB)) * sigmoidDerivative(wTimesXPlusB)

np.sum(w)

w = [1,1]
b = 1

for i in range(1000):
    wTimesXPlusB = []
    for i in range(len(X)):
        value = 0
        for j in range(len(w)):
            value += w[j] * X.iloc[i,:][j]
        wTimesXPlusB.append(value + b)
                
                
    gradientDecent =  (y_train - wTimesXPlusB) 
    testMulti = []
    for j in range(2):
        sum = 0
        for i in range(len(X)):
            sum += (y_train[i] - wTimesXPlusB[i])* X.iloc[:,j][i]
        testMulti.append(sum)
                    
    for i in range(2):                    
        w[i] = w[i] + 0.0001 * testMulti[i]
    b = b + 0.0001 * np.sum(gradientDecent)  



target_predict = []
for i in range(len(X)):
    sum = 0
    for j in range(X.shape[1]):
        sum += w[j] * X.iloc[i,:][j]
        sum += b
    target_predict.append(np.sign(sum).astype(int))

















