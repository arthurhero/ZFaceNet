import matplotlib.pyplot as plt 
import numpy as np
import time
from datetime import timedelta
import math
import socket
from multiprocessing import Pool
import sys 
import urllib2
import cv2 
import subprocess
import random
import os

import torch 
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

import dataloader as dl

model_path="models/model.ckpt"
triplet_model_path="models/triplet_model.ckpt"

#################################resnet structure

# Block numbers.
num_blocks= [3,4,6,3]

##################################Other params
num_epochs=10
vali_epoch_num= 5
test_epoch_num = 10

img_size = dl.img_size
num_channels = dl.num_channels 
num_classes = dl.num_classes

#stochastic gradient descent
mini_batch_size  = dl.mini_batch_size
momentum_coeff = 0.9

#regularization
weight_decay_coeff = 1e-5

learning_rate = 0.1
decrease_factor = 10.0    #when validation accuracy stop increasing

bias_init = 0.01

######################################

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Convolutional neural network
def init_weights(m):
    if type(m)==nn.Linear or type(m)==nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(bias_init)

# BottleNeck Class
class BottleNeck(nn.Module):
    expansion = 4
    def __init__(self,in_channels,channels,stride=1):
        super(BottleNeck,self).__init__()
        self.conv1=nn.Conv2d(in_channels,channels,kernel_size=1,stride=1)
        self.bn1=nn.BatchNorm2d(channels)
        self.conv2=nn.Conv2d(channels,channels,kernel_size=3,stride=stride,padding=1)
        self.bn2=nn.BatchNorm2d(channels)
        self.conv3=nn.Conv2d(channels,channels*self.expansion,kernel_size=1,stride=1)
        self.bn3=nn.BatchNorm2d(channels*self.expansion)
        self.dim_change=None
        if stride != 1 or in_channels != channels*self.expansion:
            self.dim_change=nn.Sequential(
                    nn.Conv2d(in_channels,channels*self.expansion,kernel_size=1,stride=stride),
                    nn.BatchNorm2d(channels*self.expansion))
        self.conv1.apply(init_weights)
        self.conv2.apply(init_weights)
        self.conv3.apply(init_weights)
        if self.dim_change is not None:
            self.dim_change.apply(init_weights)

    def forward(self,x):
        res=x
        output=F.relu(self.bn1(self.conv1(x)))
        output=F.relu(self.bn2(self.conv2(output)))
        output=self.bn3(self.conv3(output))
        if self.dim_change is not None:
            res=self.dim_change(res)
        output += res
        output = F.relu(output)
        return output
        

class ResNet(nn.Module):
    def __init__(self,block):
        super(ResNet, self).__init__()
        self.in_channels=64
        self.conv1=nn.Conv2d(3,self.in_channels,kernel_size=7,stride=2,padding=3)
        self.conv1.apply(init_weights)
        self.bn1=nn.BatchNorm2d(self.in_channels)
        self.pool1=nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        self.layer1=self._layer(block,64,num_blocks[0],stride=1)
        self.layer2=self._layer(block,128,num_blocks[1],stride=2)
        self.layer3=self._layer(block,256,num_blocks[2],stride=2)
        self.layer4=self._layer(block,512,num_blocks[3],stride=2)
        self.averagePool=nn.AvgPool2d(kernel_size=7,stride=1)
        self.fc=nn.Linear(512*block.expansion,num_classes)
        self.fc.apply(init_weights)
        # Load checkpoint
        if os.path.isfile(model_path):
            self.load_state_dict(torch.load(model_path))
            print "loaded checkpoint!"

    def _layer(self,block,channels,num_blocks,stride=1):
        blocks=[]
        blocks.append(block(self.in_channels,channels,stride=stride))
        self.in_channels=channels*block.expansion
        for i in range(1,num_blocks):
            blocks.append(block(self.in_channels,channels))
        return nn.Sequential(*blocks)

    def forward(self,x): 
        out = F.relu(self.pool1(self.bn1(self.conv1(x))))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.self.averagePool(out)
        out = out.reshape(out.size(0), -1) 
        out = self.fc(out)
        return out 

# Train the model
def train():
    model = ResNet(BottleNeck).to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,momentum=momentum_coeff,weight_decay=weight_decay_coeff)
    for epoch in range(num_epochs):
        imgs, labels = dl.get_mini_batch()
        imgs = map(torch.FloatTensor,imgs)
        imgs = torch.stack(imgs)
        labels = torch.LongTensor(labels)
        imgs = imgs.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        _, predicted = torch.max(outputs.data, 1)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print ('Epoch [{}/{}], Loss: {:.4f}'
                .format(epoch+1, num_epochs, loss.item()))
        total=labels.size(0)
        correct=(predicted == labels).sum().item()
        # Save the model checkpoint
        torch.save(model.state_dict(), model_path)
        print('Training Accuracy of the model on the {} training images: {} %'.format(total, 100 * correct / total))

# Validate the model
def validate():
    model = ResNet(BottleNeck).to(device)
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for i in range(vali_epoch_num):
            imgs, labels = dl.get_test_batch(vali=True)
            imgs = map(torch.FloatTensor,imgs)
            imgs = torch.stack(imgs)
            labels = torch.LongTensor(labels)
            imgs = imgs.to(device)
            labels = labels.to(device)
            outputs = model(imgs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print('Validation Accuracy of the model on the {} validation images: {} %'.format(total, 100 * correct / total))

# Get score distance
def get_score_distance(p1,p2):
    model = ResNet(BottleNeck).to(device)
    model.eval()
    with torch.no_grad():
        img1,img2 = dl.get_one_pair(p1,p2)
        img1=torch.FloatTensor(img1)
        img2=torch.FloatTensor(img2)
        imgs = torch.stack([img1,img2])
        imgs = imgs.to(device)
        outputs = model(imgs)
        d1=outputs[0]
        d2=outputs[1]
        sm = nn.Softmax(dim=0)
        d1=sm(d1)
        d2=sm(d2)
        dis=(d1-d2).pow(2).sum()
        print ('Distance between {} and {} is {:.2f}'.format(p1,p2,dis))

'''
get_score_distance('Aamir_Khan','Aamir_Khan')
get_score_distance('Aamir_Khan','Chris_Hemsworth')
get_score_distance('Aamir_Khan','Scarlett_Pomers')
get_score_distance('Aamir_Khan','Helen_McCrory')
'''
