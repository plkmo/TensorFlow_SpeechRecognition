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
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

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
    
### Loads model and optimizer states
def load(net, optimizer, load_best=False):
    base_path = "./data/"
    if load_best == False:
        checkpoint = torch.load(os.path.join(base_path,"checkpoint.pth.tar"))
    else:
        checkpoint = torch.load(os.path.join(base_path,"model_best.pth.tar"))
    start_epoch = checkpoint['epoch']
    best_pred = checkpoint['best_acc']
    net.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return start_epoch, best_pred

def model_eval(net, test_loader, cuda=None):
    correct = 0
    total = 0
    with torch.no_grad():
        net.eval()
        for data in test_loader:
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

##### DenseNet #####
class DenseLayer(nn.Module):
    def __init__(self, c_in, c_out, kernel_size=3, droprate=0.1):
        super(DenseLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels=c_in, out_channels=c_out,\
                               kernel_size=kernel_size, stride=1, padding=1)
        self.batch_norm = nn.BatchNorm2d(c_out)
        self.droprate = droprate
        if self.droprate > 0:
            self.drop = nn.Dropout(p=self.droprate)

    def forward(self, s):
        out = torch.relu(self.batch_norm(self.conv(s)))
        if self.droprate > 0:
            out = self.drop(out)
        out = torch.cat((s, out), 1)
        return out
    
class BottleneckLayer(nn.Module):
    def __init__(self, c_in, c_out, kernel_size=3, droprate=0.1):
        super(BottleneckLayer, self).__init__()
        self.conv1 = nn.Conv2d(c_in, 4*c_out, kernel_size=1, padding=0, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(4*c_out)
        self.droprate = droprate
        if self.droprate > 0:
            self.drop1 = nn.Dropout(p=self.droprate)
        self.conv2 = nn.Conv2d(4*c_out, c_out, kernel_size=kernel_size, padding=1, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(c_out)

    def forward(self, x):
        out = torch.relu(self.batch_norm1(self.conv1(x)))
        if self.droprate > 0:
            out = self.drop1(out)
        out = torch.relu(self.batch_norm2(self.conv2(out)))
        out = torch.cat((x, out), 1)
        return out

class DownsampleLayer(nn.Module):
    def __init__(self, c_in, c_out):
        super(DownsampleLayer, self).__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, kernel_size=1, stride=1,\
                               bias=False)
        self.batch_norm = nn.BatchNorm2d(c_out)
        self.avgpool = nn.AvgPool2d(kernel_size=2)

    def forward(self, x):
        out = torch.relu(self.batch_norm(self.conv1(x)))
        out = self.avgpool(out)
        return out

class DenseBlock(nn.Module):
    def __init__(self, c_in, k, num_layers, layertype=DenseLayer, kernel_size=3, droprate=0.1):
        super(DenseBlock, self).__init__()
        self.c_in = c_in
        self.k = k
        self.num_layers = num_layers
        for block in range(self.num_layers):
            setattr(self, "dense_%i" % block, layertype(self.c_in + block*self.k , self.k, \
                                                        droprate=droprate, kernel_size=kernel_size))
        self.last = self.c_in + (block + 1)*self.k
    
    def forward(self, seq):
        for block in range(self.num_layers):
            seq = getattr(self, "dense_%i" % block)(seq)
        return seq

class DenseNet_Block(nn.Module):
    def __init__(self, c_in, c_out, batch_size):
        super(DenseNet_Block, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.batch_size = batch_size
        self.k = 16
        # Initial convolution layer
        self.conv1 = nn.Conv2d(in_channels=self.c_in, out_channels=self.c_out,\
                               kernel_size=5, stride=1, padding=1)
        self.batch_norm1 = nn.BatchNorm2d(self.c_out)
        self.drop1 = nn.Dropout(p=0.05)
        # 1st dense + downsample block
        self.dense1 = DenseBlock(self.c_out, k=self.k, num_layers=7, kernel_size=3,\
                                 layertype=DenseLayer, droprate=0.05)
        self.ds1 = DownsampleLayer(self.dense1.last, int(self.dense1.last/2))
        # 2nd dense + downsample block
        self.dense2 = DenseBlock(int(self.dense1.last/2), k=self.k, num_layers=6, kernel_size=3,\
                                 layertype=BottleneckLayer, droprate=0.02)
        self.ds2 = DownsampleLayer(self.dense2.last, int(self.dense2.last/2))
        # 3rd dense + downsample block
        self.dense3 = DenseBlock(int(self.dense2.last/2), k=self.k, num_layers=5, layertype=BottleneckLayer,\
                                 droprate=0.02)
        self.ds3 = DownsampleLayer(self.dense3.last, int(self.dense3.last/2))
        # 4th dense + downsample block
        self.dense4 = DenseBlock(int(self.dense3.last/2), k=self.k, num_layers=5, layertype=BottleneckLayer,\
                                 droprate=0.05)
        self.ds4 = DownsampleLayer(self.dense4.last, int(self.dense4.last/2))
        '''
        # 5th dense + downsample block
        self.dense5 = DenseBlock(int(self.dense4.last/2), k=self.k, num_layers=2, layertype=BottleneckLayer,\
                                 droprate=0.01)
        self.ds5 = DownsampleLayer(self.dense5.last, int(self.dense5.last/2))
        # 6th dense + downsample block
        self.dense6 = DenseBlock(int(self.dense5.last/2), k=self.k, num_layers=2, layertype=BottleneckLayer,\
                                 droprate=0.01)
        self.ds6 = DownsampleLayer(self.dense6.last, int(self.dense6.last/2))
        '''
        
    def forward(self, seq):
        # seq input = batch_size X height X width
        seq = seq.unsqueeze(1)
        seq = torch.relu(self.batch_norm1(self.conv1(seq))); #print(seq.shape)
        seq = self.drop1(seq); #print(seq.shape)
        seq = self.ds1(self.dense1(seq)); #print(seq.shape)
        seq = self.ds2(self.dense2(seq)); #print(seq.shape)
        seq = self.ds3(self.dense3(seq)); #print(seq.shape)
        seq = self.ds4(self.dense4(seq)); #print(seq.shape)
        #seq = self.ds5(self.dense5(seq)); print(seq.shape)
        #seq = self.ds6(self.dense6(seq)); print(seq.shape)
        return seq

class DenseNetV2(nn.Module):
    def __init__(self, c_in, c_out, batch_size):
        super(DenseNetV2, self).__init__()
        self.batch_size = batch_size
        self.denseblock = DenseNet_Block(c_in=c_in, c_out=c_out,\
                                         batch_size=self.batch_size)
        self.fc1 = nn.Linear(81, 11)
    
    def forward(self, seq):
        seq = self.denseblock(seq); #print(seq.shape)
        seq = seq.reshape(len(seq[:,0,0]), -1)
        seq = self.fc1(seq)
        return seq
        
if __name__ == "__main__":
    data = load_pickle("data.pkl")
    batch_size = 32
    df = pd.DataFrame(data=np.array(data), columns=["mfcc", "label"])
    ## train-test split
    X_train, X_test, y_train, y_test = train_test_split(df["mfcc"], df["label"],\
                                                      test_size = 0.2,\
                                                      random_state = 7,\
                                                      shuffle=True,\
                                                      stratify=df["label"])
    trainset = dataset(X_train, y_train)
    testset = dataset(X_test, y_test)
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=False, \
                              num_workers=0, pin_memory=False)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, \
                              num_workers=0, pin_memory=False)
    cuda = torch.cuda.is_available()
    net = DenseNetV2(c_in=1, c_out=32, batch_size=batch_size)
    if cuda:
        net.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.01)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20,40,60,80,100,200,300], gamma=0.7)

    try:
        start_epoch, best_pred = load(net, optimizer, load_best=False)
    except:
        start_epoch = 0; best_pred = 0
    end_epoch = 330

    try:
        losses_per_epoch = load_pickle("losses_per_epoch.pkl")
        accuracy_per_epoch = load_pickle("test_accuracy_per_epoch.pkl")
    except:
        losses_per_epoch = []; accuracy_per_epoch = [];

    for e in range(start_epoch, end_epoch):
        net.train()
        total_loss = 0.0; losses_per_batch = []
        for i, (X, y) in enumerate(train_loader):
            if cuda:
                X = X.cuda().float(); y = y.cuda()
            optimizer.zero_grad()
            outputs = net(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if i % 50 == 49: # print every 50 mini-batches of size = batch_size
                losses_per_batch.append(total_loss/50)
                print('[Epoch: %d, %5d/ %d points] total loss per batch: %.7f' %
                      (e, (i + 1)*batch_size, len(trainset), total_loss/50))
                total_loss = 0.0
        losses_per_epoch.append(sum(losses_per_batch)/len(losses_per_batch))
        score = model_eval(net, test_loader, cuda=cuda)
        accuracy_per_epoch.append(score)
        if score > best_pred:
            best_pred = score
            torch.save({
                    'epoch': e + 1,\
                    'state_dict': net.state_dict(),\
                    'best_acc': score,\
                    'optimizer' : optimizer.state_dict(),\
                }, os.path.join("./data/" ,"model_best.pth.tar"))
        if (e % 10) == 0:
            save_as_pickle("losses_per_epoch.pkl", losses_per_epoch)
            save_as_pickle("test_accuracy_per_epoch.pkl", accuracy_per_epoch)
            torch.save({
                    'epoch': e + 1,\
                    'state_dict': net.state_dict(),\
                    'best_acc': score,\
                    'optimizer' : optimizer.state_dict(),\
                }, os.path.join("./data/" ,"checkpoint.pth.tar"))
    
    fig = plt.figure(figsize=(20,20))
    ax = fig.add_subplot(111)
    ax.scatter([e for e in range(len(losses_per_epoch))], losses_per_epoch)
    ax.tick_params(axis="both", length=2, width=1, labelsize=14)
    ax.set_xlabel("Epoch", fontsize=22)
    ax.set_ylabel("Loss per batch", fontsize=22)
    ax.set_title("Loss vs Epoch", fontsize=32)
    print('Finished Training')
    plt.savefig(os.path.join("./data/" ,"loss_vs_epoch.png"))
    
    fig2 = plt.figure(figsize=(20,20))
    ax2 = fig2.add_subplot(111)
    ax2.scatter([e for e in range(len(accuracy_per_epoch))], accuracy_per_epoch)
    ax2.tick_params(axis="both", length=2, width=1, labelsize=14)
    ax2.set_xlabel("Epoch", fontsize=22)
    ax2.set_ylabel("Test Accuracy", fontsize=22)
    ax2.set_title("Test Accuracy vs Epoch", fontsize=32)
    print('Finished Training')
    plt.savefig(os.path.join("./data/" ,"accuracy_vs_epoch.png"))