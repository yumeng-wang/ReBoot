#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 26 22:15:29 2021

@author: wangyumeng
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import random_split, DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms

import utilis
import CNN
from FedAvg import local_update, avg_parameters
torch.manual_seed(1234)

# load data
dataset1, dataset2 = random_split(datasets.FashionMNIST('data', train=True, 
                                  transform=transforms.ToTensor()), [10000, 50000])
dataset3, dataset4 = random_split(datasets.FashionMNIST('data', train=False, 
                                  transform=transforms.ToTensor()), [5000, 5000])

# reboot data
sample = utilis.get_sample_from_dataset(dataset2) 
label  = utilis.get_label_from_dataset(dataset2)
sample_loader = DataLoader(sample, shuffle=False)  

# train, valid and test data
train_loader = DataLoader(dataset1, batch_size=25, shuffle=True)
valid_loader = DataLoader(dataset3, batch_size=25, shuffle=True)
test_loader  = DataLoader(dataset4, batch_size=25, shuffle=False)

   
m = 20  
num_comm = 50
#device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
torch.set_num_threads(18)
device = torch.device("cpu")

idx_train = utilis.partition(range(10000), m)
idx_valid = utilis.partition(range(5000), m)
local_train_loader = {}
local_valid_loader = {}
for i in range(m):
    local_train_loader['client'+str(i+1)] = DataLoader(Subset(train_loader.dataset, idx_train[i]), 
                                                             batch_size=25, shuffle=True)
    local_valid_loader['client'+str(i+1)] = DataLoader(Subset(valid_loader.dataset, idx_valid[i]), 
                                                             batch_size=25, shuffle=True)
    
num_acc = []
global_CNN = CNN.ConvNet().to(device)
global_parameters = global_CNN.state_dict()

for i in range(num_comm):
    print("\n----------------communicate round {}----------------\n".format(i+1))
    clients_parameters = {}
    for k in range(m):
        print("----------------clients {}--------------\n".format(k+1))
        client_train_loader = local_train_loader['client'+str(k+1)]
        client_valid_loader = local_valid_loader['client'+str(k+1)]
        clients_parameters['client'+str(k+1)] = local_update(client_train_loader, 
                                                client_valid_loader, global_parameters, device)
    global_parameters = avg_parameters(global_parameters.keys(), clients_parameters)

    global_CNN.eval()
    with torch.no_grad():
        global_CNN.load_state_dict(global_parameters, strict=True)
        sum_acc = 0
        for data, label in test_loader:
            data, label = data.to(device), label.to(device)
            outputs = torch.argmax(global_CNN(data), dim=1)
            sum_acc += (outputs == label).sum()
        acc_avg = 100. * sum_acc.float() / len(test_loader.dataset)
        num_acc.append(acc_avg)
        print('FedAvg accuracy: {}'.format(acc_avg))
print(num_acc)

