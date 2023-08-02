import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset,DataLoader
import os

from sklearn.model_selection import KFold
from torch import nn
from torchsummary import summary
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


from torch.optim.lr_scheduler import StepLR
import pickle


# 该类用于即时中止模型过拟合
class EarlyStopping:
    
    def __init__(self,filename,save_path="./OUTPUT/model", patience=100, delta=0):
        self.save_path = save_path
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.filename = filename

    def __call__(self, val_acc, model):
        
        score = val_acc
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint( model)
            self.counter = 0

    def save_checkpoint(self, model):

        path = os.path.join(self.save_path, self.filename)

        torch.save( model.state_dict(), path)	# 这里会存储迄今最优模型的参数


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


def avePooling(arr,targetSize):
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
    

    
def TRAIN_CNN(model,modelName,target,KFSORT,dataLoader_train,dataLoader_valid,dataCount,epochs=500,lr=0.0001,scheduler_step_size=100,scheduler_gamma=0.97):
    
    
 
    
    #model = MODEL_CNN_EDA(input_channel=1)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr) #0.0008

    criterion = torch.nn.CrossEntropyLoss()

    scheduler = StepLR(optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)

    lossHistory = []
    accHistory = []
    trainAccHis = []



    early_stopping = EarlyStopping(patience=200,filename = "model_"+modelName+"_"+target+"_5("+str(KFSORT+1)+").pth")

    for epoch in range(0,epochs):
        model.train()
        LOSS = 0
        ACC = 0
        for batch_data, batch_targets in dataLoader_train:
            optimizer.zero_grad()
            outputs = model(batch_data.to(device))
            loss = criterion(outputs, batch_targets.to(device))
            loss.backward()

            LOSS += loss.detach().cpu().numpy()
            ACC += (torch.max(outputs,dim=1)[1] == batch_targets.to(device)).sum()

            optimizer.step()

        scheduler.step()

        lossHistory.append(LOSS)

        print(f"epoch{epoch+1}: {str(LOSS)[0:7]}",end="\t ")
        cal = (ACC/dataCount).cpu()
        print(f"acc_t[{str(np.array(cal))[0:5]}]",end="\t ")
        trainAccHis.append(cal)
        print("lr["+str(optimizer.param_groups[0]['lr'])+"]",end="\t ")

    #     for p in model.parameters():
    #         print(p.grad.data)

        model.eval()
        with torch.no_grad():
            all_outputs = []
            all_targets = []
            for batch_data, batch_targets in dataLoader_valid:
                outputs = model(batch_data.to(device))
                _, predicted = torch.max(outputs.data, 1)
                all_outputs.append(predicted.cpu().numpy())
                all_targets.append(batch_targets.cpu().numpy())

            all_outputs = np.concatenate(all_outputs)
            all_targets = np.concatenate(all_targets)
            accuracy = accuracy_score(all_targets, all_outputs)
            accHistory.append(accuracy)
    #         accAverHis.append(np.mean(np.array(accHistory)))
            print(f"acc_val: {str(accuracy)[0:7]}，",end="\t")
            print('OK '+'='*20 if(np.max(np.array(accHistory)) == accuracy) else str(100*(accuracy - np.max(np.array(accHistory))))[0:5],end="\t")


        if(epoch>50):
            print(f"[{ str(np.abs(np.mean(lossHistory)-np.mean(lossHistory[0:epoch-50])))[0:4] }")
    #         if(np.abs(np.mean(lossHistory)-np.mean(lossHistory[0:epoch-50]))<0.05):
    #             if(np.abs(100*(accuracy - np.max(np.array(accHistory))))<5):
    #                 print("已提前结束。")
    #                 break


        else:
            print("")

        early_stopping(accuracy, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break   
    return (lossHistory,accHistory,trainAccHis)
