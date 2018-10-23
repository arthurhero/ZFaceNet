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
import torchvision
import torchvision.transforms as transforms

import dataloader as dl

avg_path="vgg_face_dataset/avg/"
folder_path="vgg_face_dataset/files_10/"
validation_path="vgg_face_dataset/validation_10/"
test_path="vgg_face_dataset/test_10/"
model_path="models/model_11.ckpt"
triplet_model_path="models/triplet_model_11.ckpt"

#####################################CNN structure

# Convolutional Layer 1.
filter_size1 = 3
num_filters1 = 16

# Convolutional Layer 2.
filter_size2 = 3
num_filters2 = 32

# Convolutional Layer 3.
filter_size3 = 3
num_filters3 = 64

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

'''
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
'''

# Convolutional Layer 14.
filter_size14 = 7
num_filters14 = 1024

# Fully-connected layer.
fc_size = 512

triplet_size = 10

##################################Other params
num_epochs=100
vali_epoch_num= 5
test_epoch_num = 10

num_channels = dl.num_channels 
num_classes = dl.num_classes

#stochastic gradient descent
mini_batch_size  = dl.mini_batch_size
momentum_coeff = 0.9

#regularization
weight_decay_coeff = 5e-4
dropout_rate = 0.5     #applied after 2 FC layers

learning_rate = 10e-2
decrease_factor = 10.0    #when validation accuracy stop increasing

triplet_learning_rate = 0.25

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
        self.layer1 = nn.Sequential(
            nn.Conv2d(num_channels, num_filters1, kernel_size=filter_size1, padding=1),
            nn.BatchNorm2d(num_filters1),
            nn.ReLU())
        self.layer1.apply(init_weights)
        self.layer2 = nn.Sequential(
            nn.Conv2d(num_filters1, num_filters2, kernel_size=filter_size2, padding=1),
            nn.BatchNorm2d(num_filters2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2.apply(init_weights)
        self.layer3 = nn.Sequential(
            nn.Conv2d(num_filters2, num_filters3, kernel_size=filter_size3, padding=1),
            nn.BatchNorm2d(num_filters3),
            nn.ReLU())
        self.layer3.apply(init_weights)
        self.layer4 = nn.Sequential(
            nn.Conv2d(num_filters3, num_filters4, kernel_size=filter_size4, padding=1),
            nn.BatchNorm2d(num_filters4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer4.apply(init_weights)
        self.layer5 = nn.Sequential(
            nn.Conv2d(num_filters4, num_filters5, kernel_size=filter_size5, padding=1),
            nn.BatchNorm2d(num_filters5),
            #nn.ReLU())
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer5.apply(init_weights)
        self.layer6 = nn.Sequential(
            nn.Conv2d(num_filters5, num_filters6, kernel_size=filter_size6, padding=1),
            nn.BatchNorm2d(num_filters6),
            #nn.ReLU())
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer6.apply(init_weights)
        self.layer7 = nn.Sequential(
            nn.Conv2d(num_filters6, num_filters7, kernel_size=filter_size7, padding=1),
            nn.BatchNorm2d(num_filters7),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer7.apply(init_weights)
        #########################
        self.layer14 = nn.Sequential(
            nn.Conv2d(num_filters7, num_filters14, kernel_size=filter_size14, padding=0),
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
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer14(out)
        out = out.reshape(out.size(0), -1) 
        out = self.fc1(out)
        out = self.fc2(out)
        return out 

class ConvNet2(ConvNet):
    def __init__(self):
        super(ConvNet2, self).__init__()
        self.fc3 = nn.Sequential(
                nn.BatchNorm1d(num_classes),
                nn.Linear(num_classes, triplet_size))
        self.fc3.apply(init_weights)
        # Load checkpoint
        if os.path.isfile(triplet_model_path):
            self.load_state_dict(torch.load(triplet_model_path))
            print "loaded triplet checkpoint!"

    def forward(self, ts): 
        outs = list()
        for t in ts:
            out = super(ConvNet2,self).forward(t)
            out = nn.functional.normalize(out,p=2)
            out = self.fc3(out)
            outs.append(out)
        outs = torch.stack(outs)
        return outs

    def predict(self, x):
        out = super(ConvNet2,self).forward(x)
        return out


# Triplet Training
triplet_margin = 0.5 

def triplet_loss_one(d1,d2,d3):
    #d1 d2 d3 are l2-normalized affine-transformed outputs
    sm = nn.Softmax(dim=0)
    d1=sm(d1)
    d2=sm(d2)
    d3=sm(d3)
    dis_p=(d1-d2).pow(2).sum()
    dis_n=(d1-d3).pow(2).sum()
    loss = triplet_margin - dis_n + dis_p
    loss = torch.max(torch.cuda.FloatTensor((0,)),loss)
    return loss

def triplet_loss(results):
    loss = torch.cuda.FloatTensor((0,))
    for result in results:
        d1 = result[0]
        d2 = result[1]
        d3 = result[2]
        loss += triplet_loss_one(d1,d2,d3)
    return loss

def triplet_train():
    model = ConvNet2().to(device)
    model.train()
    # Freeze the model except last fc layer
    for child in model.children():
        for param in child.parameters():
            param.requires_grad = False
    model.fc2.weight.requires_grad = True
    model.fc2.bias.requires_grad = True
    model.fc3[1].weight.requires_grad = True
    model.fc3[1].bias.requires_grad = True
    # Optimizer
    triplet_optimizer = torch.optim.SGD(model.parameters(), lr=triplet_learning_rate,momentum=momentum_coeff)
    for epoch in range(num_epochs):
        triplets = dl.get_triplet_batch()
        triplets = map(lambda t:map(torch.FloatTensor,t),triplets)
        triplets = map(torch.stack,triplets)
        triplets = torch.stack(triplets)
        triplets = triplets.to(device)
        # Forward pass
        outputs = model(triplets)
        #print model.fc3[1].weight
        #print outputs
        loss = triplet_loss(outputs)
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
        #print model.fc1[0].weight
        #print model.fc2.weight
        #print predicted 
        #print labels 
        
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
    model = ConvNet2().to(device)
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
