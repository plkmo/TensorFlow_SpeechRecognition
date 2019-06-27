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
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from models import DenseNetV2
from preprocess_speech import extract_MFCC
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import logging
from argparse import ArgumentParser
from tqdm import tqdm

logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', \
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger(__file__)

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
    def __init__(self, X, y):
        self.X = X
        self.y = y
        
    def __len__(self):
        return (len(self.y))
    
    def __getitem__(self, idx):
        X = self.X.iloc[idx]
        y = self.y.iloc[idx]
        return X, y
    
def load_dataloaders(args):
    basepath = "./data/"
    data_path = os.path.join(basepath, "data.pkl")
    if os.path.isfile(data_path):
        data = load_pickle(data_path)
        logger.info("Loaded preprocessed data.")
    else:
        logger.info("Preprocessing...")
        extract_MFCC(args)
        data = load_pickle(data_path)
    df = pd.DataFrame(data=np.array(data), columns=["mfcc", "label"])
    ## train-test split
    X_train, X_test, y_train, y_test = train_test_split(df["mfcc"], df["label"],\
                                                      test_size = 0.2,\
                                                      random_state = 7,\
                                                      shuffle=False,\
                                                      stratify=df["label"])
    trainset = dataset(X_train, y_train)
    testset = dataset(X_test, y_test)
    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, \
                              num_workers=0, pin_memory=False)
    test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, \
                              num_workers=0, pin_memory=False)
    train_length = len(trainset)
    return train_loader, test_loader, train_length
    
def load_state(net, optimizer, scheduler, args, load_best=False):
    """ Loads saved model and optimizer states if exists """
    base_path = "./data/"
    checkpoint_path = os.path.join(base_path,"test_checkpoint_%d.pth.tar" % args.model_no)
    best_path = os.path.join(base_path,"test_model_best_%d.pth.tar" % args.model_no)
    start_epoch, best_pred, checkpoint = 0, 0, None
    if (load_best == True) and os.path.isfile(best_path):
        checkpoint = torch.load(best_path)
        logger.info("Loaded best model.")
    elif os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        logger.info("Loaded checkpoint model.")
    if checkpoint != None:
        start_epoch = checkpoint['epoch']
        best_pred = checkpoint['best_acc']
        net.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        logger.info("Loaded model and optimizer.")    
    return start_epoch, best_pred

def load_results(args):
    """ Loads saved results if exists """
    losses_path = "./data/test_losses_per_epoch_%d.pkl" % args.model_no
    accuracy_path = "./data/test_accuracy_per_epoch_%d.pkl" % args.model_no
    if os.path.isfile(losses_path) and os.path.isfile(accuracy_path):
        losses_per_epoch = load_pickle("test_losses_per_epoch_%d.pkl" % args.model_no)
        accuracy_per_epoch = load_pickle("test_accuracy_per_epoch_%d.pkl" % args.model_no)
        logger.info("Loaded results buffer")
    else:
        losses_per_epoch, accuracy_per_epoch = [], []
    return losses_per_epoch, accuracy_per_epoch

def model_eval(net, test_loader, cuda=None):
    correct = 0
    total = 0
    print("Evaluating...")
    with torch.no_grad():
        net.eval()
        for data in tqdm(test_loader):
            images, labels = data
            if cuda:
                images, labels = images.cuda(), labels.cuda()
            images = images.float(); labels = labels.long()
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print("Accuracy of the network on the %d test images: %d %%" % (total,\
                                                                    100*correct/total))
    return 100*correct/total

        
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--n_mfcc", type=int, help="number of MFCC coefficients")
    parser.add_argument("--n_fft", type=int, help="Length of FFT window")
    parser.add_argument("--hop_length", type=int, help="number of samples between successive frames")
    parser.add_argument("--mfcc_bin_len", type=int, help="MFCC binning length")
    parser.add_argument("--batch_size", type=int, default=50, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=300, help="No of epochs")
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
    parser.add_argument("--gradient_acc_steps", type=int, default=1, help="Number of steps of gradient accumulation")
    parser.add_argument("--max_norm", type=float, default=1.0, help="Clipped gradient norm")
    parser.add_argument("--model_no", type=int, default=0, help="Model ID")
    args = parser.parse_args()
    
    logger.info("Loading data and model...")
    train_loader, test_loader, train_length = load_dataloaders(args)
    
    cuda = torch.cuda.is_available()
    net = DenseNetV2(c_in=1, c_out=32, batch_size=args.batch_size)
    if cuda:
        net.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20,40,60,80,100,200,300], gamma=0.77)
    
    start_epoch, best_pred = load_state(net, optimizer, scheduler, args, load_best=False)    
    losses_per_epoch, accuracy_per_epoch = load_results(args)
    
    logger.info("Starting training process...")
    for e in range(start_epoch, args.num_epochs):
        net.train()
        total_loss = 0.0; losses_per_batch = []
        for i, (X, y) in enumerate(train_loader):
            if cuda:
                X = X.cuda().float(); y = y.cuda()
            outputs = net(X)
            loss = criterion(outputs, y)
            loss = loss/args.gradient_acc_steps
            loss.backward()
            clip_grad_norm_(net.parameters(), args.max_norm)
            if (e % args.gradient_acc_steps) == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
            total_loss += loss.item()
            if i % 50 == 49: # print every 50 mini-batches of size = batch_size
                losses_per_batch.append(total_loss/50)
                print('[Epoch: %d, %5d/ %d points] total loss per batch: %.7f' %
                      (e, (i + 1)*args.batch_size, train_length, total_loss/50))
                total_loss = 0.0
        losses_per_epoch.append(sum(losses_per_batch)/len(losses_per_batch))
        score = model_eval(net, test_loader, cuda=cuda)
        accuracy_per_epoch.append(score)
        if score > best_pred:
            best_pred = score
            torch.save({
                    'epoch': e + 1,\
                    'state_dict': net.state_dict(),\
                    'best_acc': accuracy_per_epoch[-1],\
                    'optimizer' : optimizer.state_dict(),\
                    'scheduler' : scheduler.state_dict(),\
                }, os.path.join("./data/" , "test_model_best_%d.pth.tar" % args.model_no))
        if (e % 5) == 0:
            save_as_pickle("test_losses_per_epoch_%d.pkl" % args.model_no, losses_per_epoch)
            save_as_pickle("test_accuracy_per_epoch_%d.pkl" % args.model_no, accuracy_per_epoch)
            torch.save({
                    'epoch': e + 1,\
                    'state_dict': net.state_dict(),\
                    'best_acc': accuracy_per_epoch[-1],\
                    'optimizer' : optimizer.state_dict(),\
                    'scheduler' : scheduler.state_dict(),\
                }, os.path.join("./data/" , "test_checkpoint_%d.pth.tar" % args.model_no))
    
    logger.info("Finished training!")
    fig = plt.figure(figsize=(20,20))
    ax = fig.add_subplot(111)
    ax.scatter([e for e in range(len(losses_per_epoch))], losses_per_epoch)
    ax.tick_params(axis="both", length=2, width=1, labelsize=14)
    ax.set_xlabel("Epoch", fontsize=22)
    ax.set_ylabel("Loss per batch", fontsize=22)
    ax.set_title("Loss vs Epoch", fontsize=32)
    plt.savefig(os.path.join("./data/" ,"loss_vs_epoch.png"))
    
    fig2 = plt.figure(figsize=(20,20))
    ax2 = fig2.add_subplot(111)
    ax2.scatter([e for e in range(len(accuracy_per_epoch))], accuracy_per_epoch)
    ax2.tick_params(axis="both", length=2, width=1, labelsize=14)
    ax2.set_xlabel("Epoch", fontsize=22)
    ax2.set_ylabel("Test Accuracy", fontsize=22)
    ax2.set_title("Test Accuracy vs Epoch", fontsize=32)
    plt.savefig(os.path.join("./data/" ,"accuracy_vs_epoch.png"))