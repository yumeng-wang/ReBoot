#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 28 20:08:58 2021

@author: wangyumeng
"""

import numpy as np
from torch import optim

import CNN

def local_update(train_loader, valid_loader, global_para, device):
    client_CNN = CNN.ConvNet().to(device)
    client_CNN.load_state_dict(global_para, strict=True)
    optimizer = optim.Adam(client_CNN.parameters())
    for epoch in range(1, 11):
        CNN.train(client_CNN, device, train_loader, optimizer, epoch)
    return client_CNN.state_dict()

def avg_parameters(keys, clients_parameters):
    sum_dict = dict.fromkeys(keys, 0)   
    avg_dict = dict.fromkeys(keys, 0)  
    for val in clients_parameters.values():
        for key in keys:
            sum_dict[key] = sum_dict[key] + val[key]
    for key in keys:
        avg_dict[key] = sum_dict[key] / len(clients_parameters)
    return avg_dict
