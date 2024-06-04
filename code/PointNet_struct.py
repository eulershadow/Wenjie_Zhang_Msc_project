import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch import optim

class Tnet(nn.Module):
    def __init__(self, k=3):
        super().__init__()
        self.k=k
        self.conv1 = nn.Conv1d(k,64,1)
        self.conv2 = nn.Conv1d(64,128,1)
        self.conv3 = nn.Conv1d(128,1024,1)
        self.fc1 = nn.Linear(1024,512)
        self.fc2 = nn.Linear(512,256)
        self.fc3 = nn.Linear(256,k*k)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)


    def forward(self, input):
        # input.shape == (bs,n,3)
        bs = input.size(0)
        xb = F.relu(self.bn1(self.conv1(input)))
        xb = F.relu(self.bn2(self.conv2(xb)))
        xb = F.relu(self.bn3(self.conv3(xb)))
        pool = nn.MaxPool1d(xb.size(-1))(xb)
        flat = nn.Flatten(1)(pool)
        xb = F.relu(self.bn4(self.fc1(flat)))
        xb = F.relu(self.bn5(self.fc2(xb)))

        #initialize as identity
        init = torch.eye(self.k, requires_grad=True).repeat(bs,1,1)
        if xb.is_cuda:
            init=init.cuda()
        matrix = self.fc3(xb).view(-1,self.k,self.k) + init
        return matrix


class Transform(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_transform = Tnet(k=3)
        self.feature_transform = Tnet(k=64)
        self.conv1 = nn.Conv1d(3,64,1)
        self.conv2 = nn.Conv1d(64,128,1)
        self.conv3 = nn.Conv1d(128,1024,1)
        
        self.fc1 = nn.Linear(3,64)
        self.fc2 = nn.Linear(64,64)
        
        self.fc3 = nn.Linear(64,64)
        self.fc4 = nn.Linear(64,128)
        self.fc5 = nn.Linear(128,1024)


        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        
        self.bn4 = nn.BatchNorm1d(64)
        self.bn5 = nn.BatchNorm1d(64)

    def forward(self, input):
        
        matrix3x3 = self.input_transform(input)
        # batch matrix multiplication
        #print(matrix3x3.size())
        xb = torch.bmm(torch.transpose(input,1,2), matrix3x3).transpose(1,2)
        #print(xb.size())
        #print(xb)

        xb = F.relu(self.bn1(self.conv1(xb)))
        # xb = F.relu(self.bn4(self.fc1(xb)))
        # xb = F.relu(self.bn5(self.fc2(xb)))

        matrix64x64 = self.feature_transform(xb)
        xb = torch.bmm(torch.transpose(xb,1,2), matrix64x64).transpose(1,2)
        # print(xb.size())

        xb = F.relu(self.bn2(self.conv2(xb)))
        xb = self.bn3(self.conv3(xb))
        
        # xb = F.relu(self.bn1(self.fc3(xb)))
        # xb = F.relu(self.bn2(self.fc4(xb)))
        # xb = self.bn3(self.fc5(xb))
        
        xb = nn.MaxPool1d(xb.size(-1))(xb)
        output = nn.Flatten(1)(xb)
        return output, matrix3x3, matrix64x64

class PointNet(nn.Module):
    def __init__(self, classes = 10):
        super().__init__()
        self.transform = Transform()
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, classes)
        

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(p=0.3)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        
        self.optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.9)
        self.loss_fn = nn.CrossEntropyLoss()
        self.running_loss = 0
        self.loss = None
        self.losses = []

    def forward(self, input):
        xb, matrix3x3, matrix64x64 = self.transform(input)
        xb = F.relu(self.bn1(self.fc1(xb)))
        xb = F.relu(self.bn2(self.dropout(self.fc2(xb))))
        output = self.fc3(xb)
        return self.logsoftmax(output)
    
    def fit(self, x, targets):
        #train/optimize/fit
        #print(x.size)
        # x = x.to(torch.float32)
        x = x.squeeze(1).permute(0, 2, 1)
        targets = targets.to(torch.long)
        #print(targets)
        preds = self.forward(x)
        #print(targets.dtype)
        self.loss = self.loss_fn(preds, targets)
        self.loss.backward()
        self.optimizer.step()
        
        loss_item = self.loss.item()
        return loss_item


    def reset_loss(self, value):
        self.running_loss = value
        self.losses = []
        return



def dim4_cm (real, pred,pred_result):

    cm = [[0,0,0,0],[0,0,0,0]]
    for i in range(len(pred)):
        pos = 0
        if float(pred[i]) < 0.5 and pred_result == 0:
            pos = 1
        elif float(pred[i]) >= 0.5 and pred_result == 0:
            pos = 0
        if float(pred[i]) < 0.5 and pred_result == 1:
            pos = 2
        elif float(pred[i]) >= 0.5 and pred_result == 1:
            pos = 3
            
            
        if real[i] == 0:

            cm[0][pos] += 1
        else:
            cm[1][pos] += 1
    return cm