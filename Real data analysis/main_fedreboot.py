#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 26 22:15:29 2021

@author: wangyumeng
"""

import numpy as np
import random
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import random_split, DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms

import utilis
import CNN
from CNN import ConvNet
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
torch.set_num_threads(36)
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
reboot_CNN = CNN.ConvNet().to(device)
reboot_parameters = reboot_CNN.state_dict()

for i in range(num_comm):
    print("\n----------------communicate round {}----------------\n".format(i+1))
    clients_parameters = {}
    all_outputs = np.empty(shape=[0, 50000], dtype=int)
    
    for k in range(m):
        print("----------------clients {}--------------\n".format(k+1))
        client_train_loader = local_train_loader['client'+str(k+1)]
        client_valid_loader = local_valid_loader['client'+str(k+1)]
        clients_parameters['client'+str(k+1)] = local_update(client_train_loader, 
                                                client_valid_loader, reboot_parameters, device)

        local_CNN = CNN.ConvNet().to(device)
        local_CNN.load_state_dict(clients_parameters['client'+str(k+1)], strict=True)
        outputs = CNN.predict(local_CNN, device, sample_loader)
        outputs = [int(x[0]) for x in outputs]
        all_outputs = np.append(all_outputs, [outputs], axis=0)
        del local_CNN
        

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
        reboot_train_loader = DataLoader(reboot_train, batch_size=512, shuffle=True, num_workers=8)
        reboot_valid_loader = DataLoader(reboot_valid, batch_size=512, shuffle=True, num_workers=8)


    optim_reboot = optim.Adam(reboot_CNN.parameters())
    num_acc_reboot = []
    for epoch in range(1, 101):
        CNN.train(reboot_CNN, device, reboot_train_loader, optim_reboot, epoch)
        acc = CNN.test(reboot_CNN, device, reboot_valid_loader)
        num_acc_reboot.append(acc)
        if epoch > 10:
            gap = np.max(num_acc_reboot[-10:]) - np.min(num_acc_reboot[-10:])
            if gap <= .5:  break
    
    #acc_reboot = CNN.test(reboot_CNN, device, test_loader)
    reboot_CNN.eval()
    with torch.no_grad():
        sum_acc = 0
        for data, label in test_loader:
            data, label = data.to(device), label.to(device)
            outputs = torch.argmax(reboot_CNN(data), dim=1)
            sum_acc += (outputs == label).sum()
        acc_reboot = 100. * sum_acc.float() / len(test_loader.dataset)
        num_acc.append(acc_reboot)
        print('FedReboot accuracy: {}'.format(acc_reboot))
    
    reboot_parameters = reboot_CNN.state_dict()
print(num_acc)
