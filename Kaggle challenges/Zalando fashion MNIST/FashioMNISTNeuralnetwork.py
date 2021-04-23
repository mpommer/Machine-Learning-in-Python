# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 17:16:17 2021
Neural Network to classify clothes:
0 T-shirt/top
1 Trouser
2 Pullover
3 Dress
4 Coat
5 Sandal
6 Shirt
7 Sneaker
8 Bag
9 Ankle boot
@author: Marcel Pommer
"""

import torch as t
import torchvision as tv
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
from matplotlib import style
import torch.optim as optim
import time
import os
import cv2
import numpy as np
from tqdm import tqdm

#%% load the dataset
root = "C:/Users/marce/Documents/Dokumente/Python Scripts/machine learning projekte/IntroductionPytorch"

train = datasets.FashionMNIST(root = root, train=True, download = True,
                       transform = transforms.Compose([transforms.ToTensor()]))

test = datasets.FashionMNIST(root = root, train=False, download = True, transform =
                       transforms.Compose([transforms.ToTensor()]))
del root
#%%
# shuffle the dataset and use batchsize 10
trainset = t.utils.data.DataLoader(train, batch_size = 10, shuffle = True)
testset = t.utils.data.DataLoader(test, batch_size = 10, shuffle = True)

for data in trainset:
    break
  

# x is first pic and y is first label
x, y = data[0][0], data[1][0]
z = data[0][0].view(-1,28,28)
#caution, shape is 1,28,28
x.shape

# label of the first image ist the 6 ntry
print(y)
# and the picture looks like a shirt, perfect
plt.imshow(data[0][0].view(28,28))
plt.show()

del x,y,z
#%% lets check the frequencies
total = 0
counter_dict = {0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0}

for data in trainset:
    Xs, ys = data
    for y in ys:
        counter_dict[int(y)] += 1
        total += 1


for i in counter_dict:
    print(counter_dict[i]/total * 100)
    
# too good distributed, erro?
del data, i, y, Xs, ys, total
#%% building the model with fc
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        # 28*28 pixels as imput, can be specified, 64 as first layer
        # second layer takes 64 as input
        # The outputlayer needs 10 as output, since we have 10 
        # possible numbers
        self.fc1 = nn.Linear(28*28, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 10)
        
    def forward(self, x):
        # F.relu, activates the neuron.
        # relu runs on the output
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        # thts why we dont want to use it here but only call the fc4
        x = self.fc4(x)
        # we want for the output sth like [0,1] or at least [0.1,0.9]
        # thats why we use softmax, for a regression we use another one
        # dim one is to distribute over the output, dim = 0
        # would output the wrong dimension
        return F.log_softmax(x,dim=1)
        
# call the class
model = Model()
print(Model)


#%% training the mdoel

optimizer = optim.Adam(model.parameters(), lr = 0.005)
# EPOCHS goes so many times trough the data
EPOCHS = 3

for data in trainset:
    X, y = data
    break

for epoch in range(EPOCHS):
    for data in trainset:
        # data is a bathc of featuressets and labels
        X,y = data
        # dont add all gradients of each batch together, sets gradient to 0
        model.zero_grad()
        output = model(X.view(-1,28*28))
        # use nll_loss as loss function, negative log likelihood loss
        loss = F.nll_loss(output, y)
        #  appliey loss function through the networks backward
        loss.backward()
        optimizer.step()
    print(loss)
    
del X,y, data, EPOCHS, epoch, loss
    
#%% get accuracy
correct = 0
total = 0

# disable gradient calculation
with t.no_grad():
    for data in trainset:
        X, y = data
        output = model(X.view(-1, 28*28))
        for idx, i in enumerate(output):
            if t.argmax(i) == y[idx]:
                correct += 1
            total +=1
            
print("Acc: ", round(correct/total, 3))
# accuracy of 0.831, not bad for the beginning 

correct = 0
total = 0

# disable gradient calculation
with t.no_grad():
    for data in testset:
        X, y = data
        output = model(X.view(-1, 28*28))
        for idx, i in enumerate(output):
            if t.argmax(i) == y[idx]:
                correct += 1
            total +=1

print("Acc: ", round(correct/total, 3))
# same acc for tst data, no overitting



del correct, total, output, X, y, i , idx, data











###############################################################################
#%% convolution

trainset = t.utils.data.DataLoader(train, batch_size = 4, shuffle = True)
testset = t.utils.data.DataLoader(test, batch_size = 4, shuffle = True)
#%%

class ModelCon(nn.Module):
    def __init__(self):
        super().__init__()
# 1 input, 32 outputs (conv features ), 5 kernal size
        #self.conv1 = nn.Conv2d(1, 6, 5)
        #self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv1 = nn.Conv2d(1, 32, kernel_size = 5)
        #self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size = 5)


        self.fc1 = nn.Linear(64*4*4, 32)
        self.fc2 = nn.Linear(32, 10)
        
  
        
    def forward(self, x):
        # since we have already calculated to find dimension
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2,2))
        # in order to find the correct input size I
        # tested and printed the size of x
        #print("test")
        #print(x.size)
        # flatten
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x
        
# call the class
model = ModelCon()
print(model)

# optimizer adam with learning rate 0.01
optimizer = optim.Adam(model.parameters(), lr = 0.01)
# cross entropy as loss function
loss_fn = nn.CrossEntropyLoss()




#%% test the model

loss_arr = []
loss_epoch_arr = []
max_epochs = 1
for epoch in range(max_epochs):
    #iterate through all the batches in each epoch
    for data in tqdm(trainset):
    #keeping the network in training mode     
        model.train()     
        inputs, labels = data            
        #clear the gradients     
        optimizer.zero_grad()     
        #forward pass     
        outputs = model(inputs)      
        loss = loss_fn(outputs, labels)     
        #backward pass     
        loss.backward()     
        optimizer.step()     
        loss_arr.append(loss.item()) 
    loss_epoch_arr.append(loss.item()) 
    print(loss)





#%% evaluation function
def evaluation(trainset):
     total, correct = 0, 0
     #keeping the network in evaluation mode 
     model.eval()
     for data in trainset:
         inputs, labels = data
         
     
         outputs = model(inputs)
        # pred = t.argmax(outputs[1])
         _, pred = t.max(outputs.data, 1)
         total += labels.size(0)
         correct += (pred == labels).sum().item()
     return  correct/total * 100



#%% accuracy train set
print(evaluation(trainset))
#%% accuracy test set
print(evaluation(testset))


#%% check accuracy during the runs

def fwd_pass(X,y, trainirenAllow = False):
    correct = 0
    if trainirenAllow:
        # set gradient to 0
        model.zero_grad()
    output = model(X)
    _, pred = t.max(output.data, 1)
    correct = (pred == y).sum().item()
    acc = correct/len(X) * 100
    loss = loss_fn(output, y)
    
    if trainirenAllow:
        loss.backward()
        optimizer.step()
    return acc, loss
 

    
    
  

def testruns(size = 100):
    # chose ranodm slice
    
    testset = t.utils.data.DataLoader(test, batch_size = size, shuffle = True)
    for data in testset:
        X,y = data
        break
        
    with t.no_grad():
        val_acc, val_loss = fwd_pass(X.view(-1,1,28,28), y)
    return val_acc, val_loss



val_acc, val_loss = testruns(size = 500)
print("Accuracy of the model: ", val_acc)
print("Loss of the model: ", val_loss)


#%%
model_name = f"model_fashionMnist-{int(time.time())}"

optimizer = optim.Adam(model.parameters(), lr = 0.005)
# mse as loss function
loss_function = nn.CrossEntropyLoss()

trainset = t.utils.data.DataLoader(train, batch_size = 60000, shuffle = True)
for data in trainset:
    X_train,y_train = data
    
    
def trainingOfModel():
    BATCH_SIZE = 100
    EPOCHS = 1
    with open("model_FashionMnist.log", "a") as f:
        for epoch in range(EPOCHS):
            for i in tqdm(range(0,len(X_train), BATCH_SIZE)):
                batch_X = X_train[i:i+BATCH_SIZE].view(-1, 1, 28, 28)
                batch_y = y_train[i:i+BATCH_SIZE]
        
                acc, loss = fwd_pass(batch_X, batch_y, trainirenAllow = True)
                
                if i % 100 == 0:
                    val_acc, val_loss = testruns(size=1000)
                    # save to the log file
                    f.write(f"{model_name},{round(time.time(), 3)}, \
                            {round(float(acc),3)},{round(float(loss),4)},\
                            {round(float(val_acc),3)},{round(float(val_loss),4)}\n")
 
                    
with open("testset", "a") as f:
     f.write(f"test")
                        


trainingOfModel()
print("Accuracy of the model: ", val_acc)
print("Loss of the model: ", val_loss)


#%% 

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

model_name = "model_fashionMnist-1619197437"

style.use("ggplot")

def create_acc_loss_graph(model_name):
    contents = open("model_FashionMnist.log","r").read().split("\n")
    
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
  

