# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 17:20:31 2019

@author: WT
"""
import os
import pickle
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd
import numpy as np
from model_train import DenseNetV2, dataset
from tqdm import tqdm

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
        
if __name__=="__main__":
    ### Loads data and best model
    data = load_pickle("data.pkl")
    batch_size = 32
    df = pd.DataFrame(data=np.array(data), columns=["mfcc", "label"])
    cuda = torch.cuda.is_available()
    net = DenseNetV2(c_in=1, c_out=32, batch_size=batch_size)
    if cuda:
        net.cuda()
    checkpoint = torch.load(os.path.join("./data/","test_model_best_%d.pth.tar" % 0))
    net.load_state_dict(checkpoint['state_dict'])
    net.eval()
    data_set = dataset(df["mfcc"], df["label"])
    train_loader = DataLoader(data_set, batch_size=batch_size, shuffle=False, \
                              num_workers=0, pin_memory=False)
    y_pred = []; y_true = []
    for X, y in tqdm(train_loader):
        y_true.extend(list(y.numpy()))
        if cuda:
            X = X.cuda().float()
        outputs = net(X)
        _, predicted = torch.max(outputs.data, 1)
        if cuda:
            predicted = predicted.cpu().numpy()
        else:
            predicted = predicted.numpy()
        y_pred.extend(list(predicted))
    
    c_m = confusion_matrix(y_true, y_pred)
    fig = plt.figure(figsize=(25,25))
    ax = fig.add_subplot(111)
    sb.heatmap(c_m, annot=True, annot_kws={"fontsize":15})
    ax.set_title("Confusion Matrix", fontsize=20)
    ax.set_xlabel("Actual class", fontsize=17)
    ax.set_ylabel("Predicted", fontsize=17)
    plt.savefig(os.path.join("./data/", "confusion_matrix.png"))