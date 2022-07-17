#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 18 12:24:44 2021

@author: wangyumeng
"""

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader

def mode(nums):
    counts = np.bincount(nums)   
    return(np.argmax(counts))

def partition(lst, n):
    division = len(lst) / float(n)
    return [list(lst)[int(round(division * i)): int(round(division * (i + 1)))] for i in range(n)]

def concat_sample(sample, m):
    concat_sample = sample
    for i in range(m - 1):
        concat_sample = ConcatDataset([concat_sample, sample])
    sample_loader = DataLoader(concat_sample, batch_size=len(concat_sample))
    for x in sample_loader:
        sample_all = x
    return sample_all

def get_sample_from_dataset(dataset):
    dataloader = DataLoader(dataset, batch_size=len(dataset))
    for x, y in dataloader:
        sample = x
    return sample

def get_label_from_dataset(dataset):
    dataloader = DataLoader(dataset, batch_size=len(dataset))
    for x, y in dataloader:
        label = y
    return label

def get_specific_label_dataset(dataset, label):  
    dataset_idx = []
    for idx, (data, target) in enumerate(dataset):
        if target in label:
            dataset_idx.append(idx)
    return torch.utils.data.Subset(dataset, dataset_idx)

def get_stop_point(lst, length, tol=.8):
    for i in range(len(lst)):
        gap = np.max(lst[i:i+length]) - np.min(lst[i:i+length])
        if gap <= tol:  break
    return i



# load dataset
#train_dataset = datasets.MNIST('data', train=True, transform=transforms.Compose([
#                           transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))
#test_dataset = datasets.MNIST('data', train=False, transform=transforms.Compose([
#                           transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))

#label = [0, 1] 
#train_dataset = get_specific_label_dataset(train_dataset, label)
#test_dataset = get_specific_label_dataset(test_dataset, label)


