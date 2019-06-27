# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 10:21:55 2019

@author: WT
"""
import torch
import torch.nn as nn

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