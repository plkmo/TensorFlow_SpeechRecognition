# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 16:34:27 2019

@author: WT
"""
import os
import pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

def load_pickle(filename):
    completeName = os.path.join("./data/",\
                                filename)
    with open(completeName, 'rb') as pkl_file:
        data = pickle.load(pkl_file)
    return data

def save_as_pickle(filename, data):
    completeName = os.path.join("./data/",\
                                filename)
    with open(completeName, 'wb') as output:
        pickle.dump(data, output)
        
class dataset(Dataset):
    def __init__(self, data):
        self.data = np.array(data)
    
    def __len__(self):
        return (len(self.data))
    
    def __getitem__(self, idx):
        X = self.data[idx, 0]
        y = self.data[idx, 1]
        return X, y
        
if __name__ == "__main__":
    data = load_pickle("data.pkl")