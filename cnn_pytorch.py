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

model_path="models/model5.ckpt"
triplet_model_path="models/triplet_model5.ckpt"

#####################################CNN structure

# Convolutional Layer 1.
filter_size1 = 3
num_filters1 = 64

# Convolutional Layer 2.
filter_size2 = 3
num_filters2 = 64

# Convolutional Layer 3.
filter_size3 = 3
num_filters3 = 128

# Convolutional Layer 4.
filter_size4 = 3
num_filters4 = 128

# Convolutional Layer 5.
filter_size5 = 3
num_filters5 = 256

# Convolutional Layer 6.
filter_size6 = 3
num_filters6 = 256

# Convolutional Layer 7.
filter_size7 = 3
num_filters7 = 256

# Convolutional Layer 8.
filter_size8 = 3
num_filters8 = 512

# Convolutional Layer 9.
filter_size9 = 3
num_filters9 = 512

# Convolutional Layer 10.
filter_size10 = 3
num_filters10 = 512

# Convolutional Layer 11.
filter_size11 = 3
num_filters11 = 512

# Convolutional Layer 12.
filter_size12 = 3
num_filters12 = 512

# Convolutional Layer 13.
filter_size13 = 3
num_filters13 = 512

# Convolutional Layer 14.
filter_size14 = 7
num_filters14 = 4096 

# Fully-connected layer.
fc_size = 4096

#triplet_size = 1024

##################################Other params
num_epochs=80
vali_epoch_num= 5
test_epoch_num = 10

num_channels = dl.num_channels 
num_classes = dl.num_classes

img_size = dl.img_size

#stochastic gradient descent
mini_batch_size  = dl.mini_batch_size
momentum_coeff = 0.2
#momentum_coeff = 0.9

#regularization
weight_decay_coeff = 5e-4
#dropout_rate = 0.5     #applied after 2 FC layers
dropout_rate = 0     #applied after 2 FC layers

#learning_rate = 10e-3
learning_rate = 10e-4
decrease_factor = 10.0    #when validation accuracy stop increasing

#triplet_learning_rate = 0.25
triplet_learning_rate = 10e-3

#weights initialization
weights_init_mean= 0.0
weights_init_std = 10e-2

bias_init = 0.0
######################################

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Convolutional neural network
def init_weights(m):
    if type(m)==nn.Linear or type(m)==nn.Conv2d:
        torch.nn.init.normal_(m.weight,weights_init_mean,weights_init_std)
        m.bias.data.fill_(bias_init)

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.relupool=nn.Sequential(nn.ReLU(),nn.MaxPool2d(kernel_size=2,stride=2))
        self.layer1 = nn.Sequential(
            nn.Conv2d(num_channels, num_filters1, kernel_size=filter_size1, padding=1),
            nn.BatchNorm2d(num_filters1),
            nn.ReLU())
        self.layer1.apply(init_weights)
        self.layer2 = nn.Sequential(
            nn.Conv2d(num_filters1, num_filters2, kernel_size=filter_size2, padding=1),
            nn.BatchNorm2d(num_filters2))
        self.layer2.apply(init_weights)
        self.layer3 = nn.Sequential(
            nn.Conv2d(num_filters2, num_filters3, kernel_size=filter_size3, padding=1),
            nn.BatchNorm2d(num_filters3),
            nn.ReLU())
        self.layer3.apply(init_weights)
        self.layer4 = nn.Sequential(
            nn.Conv2d(num_filters3, num_filters4, kernel_size=filter_size4, padding=1),
            nn.BatchNorm2d(num_filters4))
        self.layer4.apply(init_weights)
        self.layer5 = nn.Sequential(
            nn.Conv2d(num_filters4, num_filters5, kernel_size=filter_size5, padding=1),
            nn.BatchNorm2d(num_filters5),
            nn.ReLU())
        self.layer5.apply(init_weights)
        self.layer6 = nn.Sequential(
            nn.Conv2d(num_filters5, num_filters6, kernel_size=filter_size6, padding=1),
            nn.BatchNorm2d(num_filters6),
            nn.ReLU())
        self.layer6.apply(init_weights)
        self.layer7 = nn.Sequential(
            nn.Conv2d(num_filters6, num_filters7, kernel_size=filter_size7, padding=1),
            nn.BatchNorm2d(num_filters7))
        self.layer7.apply(init_weights)
        #########################
        self.layer8 = nn.Sequential(
            nn.Conv2d(num_filters7, num_filters8, kernel_size=filter_size8, padding=1),
            nn.BatchNorm2d(num_filters8),
            nn.ReLU())
        self.layer8.apply(init_weights)
        self.layer9 = nn.Sequential(
            nn.Conv2d(num_filters8, num_filters9, kernel_size=filter_size9, padding=1),
            nn.BatchNorm2d(num_filters9),
            nn.ReLU())
        self.layer9.apply(init_weights)
        self.layer10 = nn.Sequential(
            nn.Conv2d(num_filters9, num_filters10, kernel_size=filter_size10, padding=1),
            nn.BatchNorm2d(num_filters10))
        self.layer10.apply(init_weights)
        self.layer11 = nn.Sequential(
            nn.Conv2d(num_filters10, num_filters11, kernel_size=filter_size11, padding=1),
            nn.BatchNorm2d(num_filters11),
            nn.ReLU())
        self.layer11.apply(init_weights)
        self.layer12 = nn.Sequential(
            nn.Conv2d(num_filters11, num_filters12, kernel_size=filter_size12, padding=1),
            nn.BatchNorm2d(num_filters12),
            nn.ReLU())
        self.layer12.apply(init_weights)
        self.layer13 = nn.Sequential(
            nn.Conv2d(num_filters12, num_filters13, kernel_size=filter_size13, padding=1),
            nn.BatchNorm2d(num_filters13))
        self.layer13.apply(init_weights)
        #########################
        self.layer14 = nn.Sequential(
            nn.Conv2d(num_filters13, num_filters14, kernel_size=filter_size14, padding=0),
            nn.ReLU())
        self.layer14.apply(init_weights)
        self.fc1 = nn.Sequential(
                nn.Linear(num_filters14, fc_size),
                nn.BatchNorm1d(fc_size),
                nn.Dropout(p=dropout_rate))
        self.fc1.apply(init_weights)
        self.fc2 = nn.Linear(fc_size, num_classes)
        self.fc2.apply(init_weights)
        # Load checkpoint
        if os.path.isfile(model_path):
            self.load_state_dict(torch.load(model_path))
            print "loaded checkpoint!"
    
    def forward(self, x): 
        batch_size=x.size()[0]
        res1=x
        out = self.layer1(x)
        out = self.layer2(out)
        pad1 = Variable(torch.zeros(batch_size,num_filters2-num_channels,img_size,img_size).to(device))
        respad1 = torch.cat((res1,pad1),1)
        out += respad1
        out = self.relupool(out)
        res2 = out
        out = self.layer3(out)
        out = self.layer4(out)
        pad2 = Variable(torch.zeros(batch_size,num_filters4-num_filters2,img_size/2,img_size/2).to(device))
        respad2 = torch.cat((res2,pad2),1)
        out += respad2
        out = self.relupool(out)
        res3 = out
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        pad3 = Variable(torch.zeros(batch_size,num_filters7-num_filters4,img_size/4,img_size/4).to(device))
        respad3 = torch.cat((res3,pad3),1)
        out += respad3
        out = self.relupool(out)
        res4 = out
        out = self.layer8(out)
        out = self.layer9(out)
        out = self.layer10(out)
        pad4 = Variable(torch.zeros(batch_size,num_filters10-num_filters7,img_size/8,img_size/8).to(device))
        respad4 = torch.cat((res4,pad4),1)
        out += respad4
        out = self.relupool(out)
        res5 = out
        out = self.layer11(out)
        out = self.layer12(out)
        out = self.layer13(out)
        # no need for padding here
        out += res5
        out = self.relupool(out)
        out = self.layer14(out)
        out = out.reshape(out.size(0), -1) 
        out = self.fc1(out)
        out = self.fc2(out)
        return out 

class TripNet(ConvNet):
    def __init__(self):
        super(TripNet, self).__init__()
        '''
        self.fc3 = nn.Sequential(
                nn.BatchNorm1d(num_classes),
                nn.Linear(num_classes, triplet_size))
        self.fc3.apply(init_weights)
        '''
        # Load checkpoint
        if os.path.isfile(triplet_model_path):
            self.load_state_dict(torch.load(triplet_model_path))
            print "loaded triplet checkpoint!"

    def forward(self,img1s,img2s,img3s): 
        out1s = super(TripNet,self).forward(img1s)
        out2s = super(TripNet,self).forward(img2s)
        out3s = super(TripNet,self).forward(img3s)
        return out1s,out2s,out3s

    def predict(self, x):
        out = super(TripNet,self).forward(x)
        return out

# Triplet Training
triplet_margin = 1 

def triplet_loss(out1s,out2s,out3s):
    sm = nn.Softmax(dim=1)
    anchs=sm(out1s)
    posis=sm(out2s)
    negas=sm(out3s)
    dis_ps=F.pairwise_distance(anchs,posis,2)
    dis_ns=F.pairwise_distance(anchs,negas,2)
    #print dis_ps,dis_ns
    target=torch.FloatTensor(dis_ps.size()).fill_(-1.0).to(device)
    target=Variable(target)
    criterion=torch.nn.MarginRankingLoss(margin=triplet_margin)
    loss=criterion(dis_ps,dis_ns,target)
    return loss

def triplet_train():
    model = TripNet().to(device)
    model.train()
    # Freeze the model except last fc layer
    for child in model.children():
        for param in child.parameters():
            param.requires_grad = False
    model.fc2.weight.requires_grad = True
    model.fc2.bias.requires_grad = True
    triplet_optimizer = torch.optim.Adam(model.parameters(), lr=triplet_learning_rate)
    for epoch in range(num_epochs):
        img1s,img2s,img3s= dl.get_triplet_batch()
        img1s = torch.stack(map(torch.FloatTensor,img1s)).to(device)
        img2s = torch.stack(map(torch.FloatTensor,img2s)).to(device)
        img3s = torch.stack(map(torch.FloatTensor,img3s)).to(device)
        # Forward pass
        out1s,out2s,out3s= model(img1s,img2s,img3s)
        loss = triplet_loss(out1s,out2s,out3s)
        # Backward and optimize
        triplet_optimizer.zero_grad()
        loss.backward()
        triplet_optimizer.step()
        print ('Epoch [{}/{}], Loss: {:.4f}'
                .format(epoch+1, num_epochs, loss.item()))
        # Save the model checkpoint
        torch.save(model.state_dict(), triplet_model_path)

# Train the model
def train():
    model = ConvNet().to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,momentum=momentum_coeff)
    #optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,momentum=momentum_coeff,weight_decay=weight_decay_coeff)
    #optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,weight_decay=weight_decay_coeff)
    for epoch in range(num_epochs):
        imgs, labels = dl.get_mini_batch()
        imgs = map(torch.FloatTensor,imgs)
        imgs = torch.stack(imgs)
        labels = torch.LongTensor(labels)
        imgs = imgs.to(device)
        labels = labels.to(device)
        #print imgs,labels
        
        # Forward pass
        outputs = model(imgs)
        #print outputs[0],outputs[1]
        loss = criterion(outputs, labels)
        _, predicted = torch.max(outputs.data, 1)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #print model.fc2.weight
        
        print ('Epoch [{}/{}], Loss: {:.4f}'
                .format(epoch+1, num_epochs, loss.item()))
        total=labels.size(0)
        correct=(predicted == labels).sum().item()
        # Save the model checkpoint
        torch.save(model.state_dict(), model_path)
        print('Training Accuracy of the model on the {} training images: {} %'.format(total, 100 * correct / total))

# Validate the model
def validate():
    model = ConvNet().to(device)
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

def validate_triplet():
    model = TripNet().to(device)
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
            outputs = model.predict(imgs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print('Validation Accuracy of the model on the {} validation images: {} %'.format(total, 100 * correct / total))

# Get score distance
def get_score_distance(p1,p2):
    model = ConvNet().to(device)
    #model = TripNet().to(device)
    model.eval()
    with torch.no_grad():
        img1,img2 = dl.get_one_pair(p1,p2)
        img1=torch.FloatTensor(img1)
        img2=torch.FloatTensor(img2)
        imgs = torch.stack([img1,img2])
        imgs = imgs.to(device)
        outputs = model(imgs)
        #outputs = model.predict(imgs)
        d1=outputs[0]
        d2=outputs[1]
        sm = nn.Softmax(dim=0)
        d1=sm(d1)*100
        d2=sm(d2)*100
        dis=(d1-d2).pow(2).sum()
        print ('Distance between {} and {} is {:.2f}'.format(p1,p2,dis))

get_score_distance('Aamir_Khan','Aamir_Khan')
get_score_distance('Aamir_Khan','Chris_Hemsworth')
get_score_distance('Aamir_Khan','Scarlett_Pomers')
get_score_distance('Aamir_Khan','Helen_McCrory')

get_score_distance('Chris_Hemsworth','Aamir_Khan')
get_score_distance('Chris_Hemsworth','Chris_Hemsworth')
get_score_distance('Chris_Hemsworth','Scarlett_Pomers')
get_score_distance('Chris_Hemsworth','Helen_McCrory')
