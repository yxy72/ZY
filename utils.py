import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset,DataLoader
import os
import pickle


def maxPooling(arr,targetSize):
    size = len(arr)/targetSize
    if(len(arr)%targetSize!=0):
        
        # print(type(size))
        return "不能整除！"
    res = []
    group = []
    for i in arr:
        group.append(i)
        if(len(group)==size):
            res.append(np.max(np.array(group)))
            group = []
    return np.array(res)

# EDA数据的CNN模型，input_channel为1
# EDA+PPG融合的CNN模型，input_channel为2
class MODEL_CNN_EDA(nn.Module):
    def __init__(self,input_channel):
        super().__init__()
        self.input_channel = input_channel
        self.cov_unit1 = nn.Sequential(
            
            nn.Conv1d(in_channels=input_channel, out_channels=16, kernel_size=3,stride = 1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3,stride = 1),
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3,stride = 1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3,stride = 1),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3,stride = 1),
            nn.MaxPool1d(2),
            nn.ReLU(),
            
            nn.BatchNorm1d(64),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3,stride = 1),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3,stride = 1),
            nn.MaxPool1d(2),
            nn.ReLU(),
            
            nn.BatchNorm1d(128),
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3,stride = 1),
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3,stride = 1),
            nn.MaxPool1d(2),
            nn.ReLU(),         
            nn.AdaptiveMaxPool1d(8),
            
            nn.Flatten(),
            
            nn.Linear(2048,1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(1024,512),
            nn.ReLU(),
            nn.Dropout(0.86),
            
            nn.Linear(512,100),
            nn.ReLU(),
            nn.Dropout(0.86),
            
            nn.Linear(100,3),
            
        )

    def forward(self, x):
        x = self.cov_unit1(x)
        return x

# PPG数据的CNN模型，PPG数据较EDA更为良好，模型可以更小
class Model_CNN_PPG(nn.Module):
    def __init__(self,input_channel=1):
        super().__init__()
        
        self.cov_unit1 = nn.Sequential(
            
            nn.Conv1d(in_channels=input_channel, out_channels=16, kernel_size=3,stride = 1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3,stride = 1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3,stride = 1),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3,stride = 1),
            nn.MaxPool1d(2),
            nn.ReLU(),
            
            nn.BatchNorm1d(64),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3,stride = 1),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3,stride = 1),
            nn.MaxPool1d(2),
            nn.ReLU(),
            
            nn.AdaptiveMaxPool1d(8),
            
            
            
            
            nn.Flatten(),
            
            nn.Linear(1024,512),
            nn.ReLU(),
            nn.Dropout(0.85),
            
            nn.Linear(512,100),
            nn.ReLU(),
            nn.Dropout(0.85),
            
            nn.Linear(100,3),
        )

    def forward(self, x):
        
#         print(x.shape)
        x = self.cov_unit1(x)
#         print(x.shape)
#         x = x.view(x.size(0), -1) 
#         print(x.shape)
#         x = self.fc1(x)
#         print(x.size)
        return x


# EDA数据的LSTM模型，性能较差
class MODEL_LSTM_EDA(nn.Module):
    def __init__(self,input_channel=1200):
        super().__init__()
        self.LSTM = nn.LSTM(input_size=1200, hidden_size=4096, num_layers=2,dropout=0.02,batch_first=True)

        
        self.Linear = nn.Sequential(
            
            nn.Linear(4096,4096),
            nn.ReLU(),
            nn.BatchNorm1d(4096),
            nn.Dropout(0.2),
            
            nn.Linear(4096,3),
        )
        

    def forward(self, x):
        x,_ = self.LSTM(x)
        x = x[:, -1, :]
        x = self.Linear(x)
        return x