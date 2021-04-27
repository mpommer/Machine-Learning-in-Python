# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 16:38:05 2021
Kaggle challenge costumer segmentation

@author: Marcel Pommer
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel as SM
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier as KN
from sklearn.gaussian_process import GaussianProcessClassifier as GS
import seaborn as sns

pd.set_option("display.max_columns", 100)
pd.set_option("display.max_rows", 100)
pd.set_option("display.max_colwidth", 100)

#%%
path = 'C:/Users/marce/Documents/Dokumente/Python Scripts/ml 2020/customerGroups.csv'
df = pd.read_csv(path)

# get a first overview over the data (first 5 rows)
df.head()
# what is the structure of the dataset
df.shape
# 71 features incl. label (columns) and 6620 observations (rows)

# Do we have any Nan in the dataset (for all variables)
df.isna().any()
# no variable has a Nan --> complete dataset

# get some information on the data type
df.info()
# all variables are either integers or floats (approx. half/half)

# get some basic information about mean/std ...
df.describe()
# we directly see that the mean of the target variable is close to 1
# which means that we have as many 0 as 2 in the target variable
# to further analyse the distribution I take a look at a bar diagram
x, y , z = np.sum(df['target']==0), np.sum(df['target']==1) ,np.sum(df['target']==2)

plt.bar(x = [0,1,2], height = [x,y,z], color =['r','g','y'])
plt.show()
# --> most frequent number is 1 with 3076





#%%
################## testing  ###########################
# before I go on with any data transformation/analysis I want to conduct 
# a test run using sklearn and check some predictions out
target = df['target']
data = df.drop(labels = 'target', axis = 1)
X_train, X_test, y_train, y_test = train_test_split(data, target,test_size = 0.3)

# I want to try two regressor
# the first regressor is an easy random forest classifier
regressor = RandomForestClassifier(n_estimators=1000, max_depth = 15, random_state=0)
regressor.fit(X_train, y_train)

# accuracy
print("Train Acc: ", regressor.score(X_train,y_train))
print("Test Acc: ", regressor.score(X_test,y_test))
# we already good pretty good predictions
# by chance we would get an accuracy of 0.33 so we barely doubles the precision
# but we can see a overfitting 

# lets see which variables are most important in predicting
selector = SM(estimator= regressor, threshold = 0.01, max_features=19).fit(X_train,y_train)

selection_variables = selector.get_support()
# what happens if we drop the not so relevant labels
X_train_reduced = X_train.loc[:,selection_variables]
X_test_reduced = X_test.loc[:,selection_variables]

regressor.fit(X_train_reduced, y_train)

# accuracy
print("Train Acc: ", regressor.score(X_train_reduced,y_train))
print("Test Acc: ", regressor.score(X_test_reduced,y_test))
# no increase in accuracy on the test data set (but still overfitting)
# intrestingly it seems like in both groups the same features remain
# so it seems like many features barely have influence on the revenue

predictions = regressor.predict(X_test_reduced)
confuse = confusion_matrix(y_test, predictions)
confuse = pd.DataFrame(confuse)
#figure(num=None, figsize=(7, 7))
sns.heatmap(confuse, linewidths=1,annot=True, fmt='.5g', annot_kws={"size": 12},cmap="YlGnBu", 
            yticklabels=['0 True','1 True','2 True'], xticklabels=['0 Predicted','1 Predicted','2 Predicted'])






#%%
# additional i check for the multicoloniarity in the model
from statsmodels.tools.tools import add_constant
from statsmodels.stats.outliers_influence import variance_inflation_factor
#X_vif = add_constant(df)
X_vif = df
vif = pd.Series([variance_inflation_factor(X_vif.values, i) 
               for i in range(X_vif.shape[1])],index=X_vif.columns)
# as before we observe some variables with low multicoll which could be removed
mult_remove = (vif!=np.inf) 
mult_remove[:-1].value_counts()
(mult_remove[:-1] == selection_variables).value_counts()
# we have high similarity in the both methods


# no lets finaly fid the best features explaining target
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
sel_cols = SelectKBest(mutual_info_classif, k=20)
sel_cols.fit(data,target)
arr = np.array(data.columns[sel_cols.get_support()])
# reduce to those variables
data = data[arr]
data.describe()




# lets test again
#%%
X_train, X_test, y_train, y_test = train_test_split(data, target,test_size = 0.3, shuffle = True)

# let us scale all variables this time
from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler(feature_range=(0,1))
X_train = mms.fit_transform(X_train)
X_test = mms.fit_transform(X_test)

# I again start with an easy random forest
regressor = RandomForestClassifier(n_estimators=1000, max_depth = 15, random_state=0)
regressor.fit(X_train, y_train)

# accuracy
print("Train Acc: ", regressor.score(X_train,y_train))
print("Test Acc: ", regressor.score(X_test,y_test))
# fairly the sam result


# creating a heatmap
predictions = regressor.predict(X_test)
confuse = confusion_matrix(y_test, predictions)
confuse = pd.DataFrame(confuse)
#figure(num=None, figsize=(7, 7))
sns.heatmap(confuse, linewidths=1,annot=True, fmt='.5g', annot_kws={"size": 12},cmap="YlGnBu", 
            yticklabels=['0 True','1 True','2 True'], xticklabels=['0 Predicted','1 Predicted','2 Predicted'])






#
'''
# lets try a neural network
from sklearn.neural_network import MLPClassifier
regressor = MLPClassifier()

data = data.drop(labels = 'target', axis = 1)
X_train, X_test, y_train, y_test = train_test_split(data, target,test_size = 0.3, shuffle = True)

df = df.drop(labels = 'target', axis = 1)


X_train, X_test, y_train, y_test = train_test_split(df, target,test_size = 0.2, shuffle = True)


model = regressor.fit(X_train, y_train)

print(model.score(X_train,y_train))
print(model.score(X_test,y_test))

predictions = model.predict(X_test)

confuse = confusion_matrix(y_test, predictions)
confuse = pd.DataFrame(confuse)
#figure(num=None, figsize=(7, 7))
sns.heatmap(confuse, linewidths=1,annot=True, fmt='.5g', annot_kws={"size": 12},cmap="YlGnBu", 
            yticklabels=['0 True','1 True','2 True'], xticklabels=['0 Predicted','1 Predicted','2 Predicted'])

'''










'''

# further I try several model and compare those using pycart
from pycaret import classification
from pycaret.regression import *
data['target'] = target
classification = setup(data=data, target = 'target')

best_model = compare_models()
print(compare_models())



gb = create_model('gbr')

t = gb.predict(X_test)
print(gb.score(X_train,y_train))
'''





#%%
############## Data transformation ##################
# since the first group seems to be more profitable than the second one
df_group1 = df[df['target'] != 2]

# lets try to classify again
target = df_group1['target']
data = df_group1.drop(labels = 'target', axis = 1)
X_train, X_test, y_train, y_test = train_test_split(data, target,test_size = 0.3)

# I want to try two regressor
# the first regressor is an easy random forest classifier
regressor = RandomForestClassifier(n_estimators=2000, max_depth = 20, random_state=0)
regressor.fit(X_train, y_train)

# accuracy
print("Train Acc: ", regressor.score(X_train,y_train))
print("Test Acc: ", regressor.score(X_test,y_test))
# we instantly get a very good prediction on the trainingsdata 
# but a bad one on the test data

# lets draw the confusion matrix
predictions = regressor.predict(X_test)

confuse = confusion_matrix(y_test, predictions)
confuse = pd.DataFrame(confuse)
#figure(num=None, figsize=(7, 7))
sns.heatmap(confuse, linewidths=1,annot=True, fmt='.5g', annot_kws={"size": 12},cmap="YlGnBu", 
            yticklabels=['0 True','1 True','2 True'], xticklabels=['0 Predicted','1 Predicted','2 Predicted'])

# bad prediction

#%%
# lets check for win vs loss
path = 'C:/Users/marce/Documents/Dokumente/Python Scripts/ml 2020/customerGroups.csv'
df = pd.read_csv(path)
dfWinLoss = df
dfWinLoss.loc[ dfWinLoss['target'] ==2, 'target'] = 1


# lets try to classify again
target = dfWinLoss['target']
data = dfWinLoss.drop(labels = 'target', axis = 1)
X_train, X_test, y_train, y_test = train_test_split(data, target,
                                         shuffle = True, test_size = 0.5)

# I want to try two regressor
# the first regressor is an easy random forest classifier
regressor = KN()
regressor.fit(X_train, y_train)

# accuracy
print("Train Acc: ", regressor.score(X_train,y_train))
print("Test Acc: ", regressor.score(X_test,y_test))
# we instantly get a very good prediction on the trainingsdata 
# but a bad one on the test data

# lets draw the confusion matrix
predictions = regressor.predict(X_test)

confuse = confusion_matrix(y_test, predictions)
confuse = pd.DataFrame(confuse)
#figure(num=None, figsize=(7, 7))
sns.heatmap(confuse, linewidths=1,annot=True, fmt='.5g', annot_kws={"size": 12},cmap="YlGnBu", 
            yticklabels=['0 True','1 True','2 True'], xticklabels=['0 Predicted','1 Predicted','2 Predicted'])














#%% test for neural network
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from sklearn.utils import shuffle
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from sklearn.model_selection import train_test_split
import tqdm
from matplotlib import style
import sklearn

#%%
path = 'C:/Users/marce/Documents/Dokumente/Python Scripts/ml 2020/customerGroups.csv'
df = pd.read_csv(path)
y = df['target']
X = df.drop(labels = 'target', axis = 1)

df_group1 = df[df['target'] == 1]
df_group1 = sklearn.utils.shuffle(df_group1)
df_group1 = df_group1[:2000]
df_noGroup1 = df[df['target'] != 1]
newDF = [df_group1,df_noGroup1]
newDF = pd.concat(newDF)
#y = newDF['target']
#X = newDF.drop(labels = 'target', axis = 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)



columns = 70
size_hidden = 100
n_outputs = 1

class Model(nn.Module):
    def __init__(self, input_dim):
        super(Model, self).__init__()
        self.layer1 = nn.Linear(input_dim, 140)
        self.layer2 = nn.Linear(140, 280)
        self.layer3 = nn.Linear(280, 560)
        self.layer4 = nn.Linear(560, 300)
        self.layer5 = nn.Linear(300, 150)
        self.layer6 = nn.Linear(150, 3)
        
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = F.relu(self.layer4(x))
        x = F.relu(self.layer5(x))
        x = F.softmax(self.layer6(x), dim=1)
        return x

        
# call the class
model = Model(70)
print(Model)

optimizer = t.optim.Adam(model.parameters(), lr=0.001)
criterion = t.nn.CrossEntropyLoss()


X_train = Variable(t.tensor(X_train.values)).float()
y_train = Variable(t.tensor(y_train.values)).long()
X_test  = Variable(t.tensor(X_test.values)).float()
y_test  = Variable(t.tensor(y_test.values)).long()

#%%
'''
batch_size = 10
batch_no = len(X_train) // batch_size
num_epochs = 10000
running_loss = 0.0
loss_list = []
accuracy_list = []

for epoch in range(num_epochs):
    y_pred = model(X_train)
    loss = criterion(y_pred, y_train)
    loss_list.append(loss.item())
        
    # Zero gradients
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
        
    with t.no_grad():
        y_pred = model(X_test)
        correct = (t.argmax(y_pred, dim=1) == y_test).type(t.FloatTensor)
        accuracy_list.append(correct.mean())
'''
def fwd_pass(X,y, trainirenAllow = False):
    correct = 0
    if trainirenAllow:
        # set gradient to 0
        model.zero_grad()
    output = model(X)
    _, pred = t.max(output.data, 1)
    correct = (pred == y).sum().item()
    acc = correct/len(X) 
    loss = criterion(output, y)
    
    if trainirenAllow:
        loss.backward()
        optimizer.step()
    return acc, loss.float()

def testruns():
    #rand = np.random.randint(len(X_test)-400)
    #X,y = X_test[rand:rand+400], y_test[rand:rand+400]
    X,y = X_test, y_test
    with t.no_grad():
        val_acc, val_loss = fwd_pass(X, y)
    return val_acc, val_loss



#%%

model_name =  f"model_PaybackChallenge-{int(time.time())}"

def trainingOfModel():
    BATCH_SIZE = 10
    EPOCHS = 15
    with open("model_Payback.log", "a") as f:
        for epoch in range(EPOCHS):
            for i in range(0,len(X_train), BATCH_SIZE):
                batch_X = X_train[i:i+BATCH_SIZE]
                batch_y = y_train[i:i+BATCH_SIZE]
                if i<len(X_train)-10*BATCH_SIZE:
                    batch_X2 = X_train[i:i+10*BATCH_SIZE]
                    batch_y2 = y_train[i:i+10*BATCH_SIZE] 
                else:
                     batch_X2 = X_train[i-10*BATCH_SIZE:i]
                     batch_y2 = y_train[i-10*BATCH_SIZE:i]                    
                acc1, loss1 = fwd_pass(batch_X, batch_y, trainirenAllow = True)
                
                
                acc, loss = fwd_pass(batch_X2, batch_y2)
                
                if i % BATCH_SIZE*10 == 0:
                    val_acc, val_loss = testruns()
                    # save to the log file
                    f.write(f"{model_name},{round(time.time(), 3)}, \
                            {round(float(acc),3)},{round(float(loss),4)},\
                            {round(float(val_acc),3)},{round(float(val_loss),4)}\n")        


trainingOfModel()


#%%
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


'''
model_name = "model_fashionMnist-1619198864"
'''

style.use("ggplot")

def create_acc_loss_graph(model_name):
    contents = open("model_Payback.log","r").read().split("\n")
    
    times = []
    acces = []
    losses = []
    
    val_accs = []
    val_losses = []
    
    for c in contents:
        if model_name in c:
            name, timestamp, acc, loss, val_acc, val_loss = c.split(",")
            
            times.append(float(timestamp))
            acces.append(float(acc))
            losses.append(float(loss))
            
            val_accs.append(float(val_acc))
            val_losses.append(float(val_loss))
            
        
    ax1 = plt.subplot2grid((2,1), (0,0))
    ax2 = plt.subplot2grid((2,1), (1,0), sharex = ax1)

    ax1.plot(times, acces, label="accucacy")
    ax1.plot(times, val_accs, label = "val_acc")
    ax1.legend(loc=2)
    
    ax2.plot(times, losses, label="loss")
    ax2.plot(times, val_losses, label = "val_loss")
    ax2.legend(loc=2)
    
    plt.show()


create_acc_loss_graph(model_name)
  






