#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 14 17:29:06 2021

@author: wangyumeng
"""

import numpy as np
import random
import torch
import torch.optim as optim
from torch.utils.data import random_split, DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms

import utilis
import CNN
from CNN import ConvNet
from FedAvg import avg_parameters


torch.manual_seed(1234)
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(36)
device = torch.device("cpu")

# load data
dataset1, dataset2 = random_split(datasets.FashionMNIST('data', train=True, 
                                  transform=transforms.ToTensor()), [10000, 50000])
dataset3, dataset4 = random_split(datasets.FashionMNIST('data', train=False, 
                                  transform=transforms.ToTensor()), [5000, 5000])

# train, valid and test data
train_loader = DataLoader(dataset1, batch_size=500, shuffle=True)
valid_loader = DataLoader(dataset3, batch_size=500, shuffle=True)
test_loader  = DataLoader(dataset4, batch_size=500, shuffle=False)

# reboot data
sample = utilis.get_sample_from_dataset(dataset2) 
label  = utilis.get_label_from_dataset(dataset2)
sample_loader = DataLoader(sample, shuffle=False)  

# local data
m = 10 # the number of client
idx_train = utilis.partition(range(10000), m)
idx_valid = utilis.partition(range(5000), m)
local_train_loader = {}
local_valid_loader = {}
for i in range(m):
    local_train_loader['client'+str(i+1)] = DataLoader(Subset(train_loader.dataset, 
                                            idx_train[i]), batch_size=250, shuffle=True)
    local_valid_loader['client'+str(i+1)] = DataLoader(Subset(valid_loader.dataset, 
                                            idx_valid[i]), batch_size=250, shuffle=True)

print('\n------------optimal estimator------------\n')
cnn_opt = ConvNet().to(device)
optim_opt = optim.Adam(cnn_opt.parameters())
num_acc_opt = []
for epoch in range(1, 401):
    CNN.train(cnn_opt, device, train_loader, optim_opt, epoch)
    acc = CNN.test(cnn_opt, device, valid_loader)
    num_acc_opt.append(acc)
    if epoch > 15:
        gap = np.max(num_acc_opt[-15:]) - np.min(num_acc_opt[-15:])
        if gap <= 1:  break
acc_opt = CNN.test(cnn_opt, device, test_loader)
del cnn_opt


print('\n------------local estimator------------\n')
cnn_avg = CNN.ConvNet().to(device)
parameters1 = cnn_avg.state_dict()
clients_parameters = {}

all_acc_local = []
all_outputs = np.empty(shape=[0, 50000], dtype=int)
for i in range(m):
    print('----------------clients {}--------------\n'.format(i + 1))
    cnn_local = ConvNet().to(device)
    cnn_local.load_state_dict(parameters1, strict=True)
    optim_local = optim.Adam(cnn_local.parameters())
    num_acc_local = []
    for epoch in range(1, 501):
        CNN.train(cnn_local, device, local_train_loader['client'+str(i+1)], optim_local, epoch)
        acc = CNN.test(cnn_local, device, local_valid_loader['client'+str(i+1)])
        num_acc_local.append(acc)
        if epoch > 15:
            gap = np.max(num_acc_local[-15:]) - np.min(num_acc_local[-15:])
            if gap <= 1.6:  break
    acc_local = CNN.test(cnn_local, device, test_loader)  
    all_acc_local.append(acc_local)
    
    #save parameter
    clients_parameters['client'+str(i+1)] = cnn_local.state_dict()
    
    # generate reboot label
    outputs = CNN.predict(cnn_local, device, sample_loader)
    outputs = [int(x[0]) for x in outputs]
    all_outputs = np.append(all_outputs, [outputs], axis=0)
    del cnn_local
    
acc_local_mean = np.mean(all_acc_local)
acc_local_best = np.max(all_acc_local)


print('------------reboot estimator------------\n')
valid_sam = random.sample(range(50000), 2500)
train_sam = np.delete(range(50000), valid_sam)

reboot_sample_train = sample[train_sam]
reboot_sample_valid = sample[valid_sam]
reboot_label_train = all_outputs[0, train_sam]
reboot_label_valid = all_outputs[0, valid_sam]
for j in range(1, m):
    reboot_sample_train = torch.cat((reboot_sample_train, sample[train_sam]), 0)
    reboot_sample_valid = torch.cat((reboot_sample_valid, sample[valid_sam]), 0)
    reboot_label_train = np.hstack((reboot_label_train, all_outputs[j,train_sam]))
    reboot_label_valid = np.hstack((reboot_label_valid, all_outputs[j,valid_sam]))
reboot_train = TensorDataset(reboot_sample_train, torch.tensor(reboot_label_train))
reboot_valid = TensorDataset(reboot_sample_valid, torch.tensor(reboot_label_valid))
reboot_train_loader = DataLoader(reboot_train, batch_size=500, shuffle=True)
reboot_valid_loader = DataLoader(reboot_valid, batch_size=500, shuffle=True)


cnn_reboot = ConvNet().to(device)
optim_reboot = optim.Adam(cnn_reboot.parameters())
num_acc_reboot = []
for epoch in range(1, 201):
    CNN.train(cnn_reboot, device, reboot_train_loader, optim_reboot, epoch)
    acc = CNN.test(cnn_reboot, device, reboot_valid_loader)
    num_acc_reboot.append(acc)
    if epoch > 15:
        gap = np.max(num_acc_reboot[-15:]) - np.min(num_acc_reboot[-15:])
        #if gap <= .6:  break
        if gap <= .8:  break
acc_reboot = CNN.test(cnn_reboot, device, test_loader)
del cnn_reboot


print('------------average estimator------------\n')
parameters2 = avg_parameters(parameters1.keys(), clients_parameters)
cnn_avg.load_state_dict(parameters2, strict=True)
acc_avg = CNN.test(cnn_avg, device, test_loader)
del cnn_avg

print('optimal CNN accuracy = {:.2f}%'.format(acc_opt))
print('local CNN  mean accuracy = {:.2f}%'.format(acc_local_mean))
print('local CNN  best accuracy = {:.2f}%'.format(acc_local_best))
print('reboot CNN accuracy = {:.2f}%'.format(acc_reboot))
print('Average CNN accuracy = {:.2f}%'.format(acc_avg))
