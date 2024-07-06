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
    def __init__(self, classes = 10, optimizer = "SGD"):
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
        if optimizer == "SGD":
            self.optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.9)
        else:
            self.optimizer = optim.Adam(self.parameters(), lr=0.00001)
            
        self.loss_fn = nn.CrossEntropyLoss(weight = torch.tensor([0.4,0.6]))
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


class PointNet_2Multihead(nn.Module):
    def __init__(self, classes = 10, optimizer = "SGD"):
        super().__init__()
        self.transform = Transform()
        self.transform2 = Transform()
        
        self.fc1 = nn.Linear(2048, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, classes)
        

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(p=0.3)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        
        self.optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.9)
        if optimizer == "SGD":
            self.optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.9)
        else:
            self.optimizer = optim.Adam(self.parameters(), lr=0.00001)
            
        self.loss_fn = nn.CrossEntropyLoss(weight = torch.tensor([0.4,0.6]))
        self.running_loss = 0
        self.loss = None
        self.losses = []

    def forward(self, input1, input2):
        #xb, matrix3x3, matrix64x64 = self.transform(input)
        xa, matrix3x3, matrix64x64 = self.transform(input1)
        xb, matrix3x3, matrix64x64 = self.transform2(input2)
        xc = torch.cat((xa, xb), dim=1)
        xc = F.relu(self.bn1(self.fc1(xc)))
        xc = F.relu(self.bn2(self.dropout(self.fc2(xc))))
        output = self.fc3(xc)
        return self.logsoftmax(output)
    
    def fit(self, input1, input2, targets):
        #train/optimize/fit
        #print(x.size)
        # x = x.to(torch.float32)
        input1 = input1.squeeze(1).permute(0, 2, 1)
        input2 = input2.squeeze(1).permute(0, 2, 1)
        targets = targets.to(torch.long)
        #print(targets)
        preds = self.forward(input1,input2)
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


class Transform_2Multi(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_transform = Tnet(k=3)
        self.feature_transform = Tnet(k=64)
        self.conv1 = nn.Conv1d(3,64,1)
        self.conv2 = nn.Conv1d(64,128,1)
        self.conv3 = nn.Conv1d(128,512,1)
        
        self.fc1 = nn.Linear(3,64)
        self.fc2 = nn.Linear(64,64)
        
        self.fc3 = nn.Linear(64,64)
        self.fc4 = nn.Linear(64,128)
        self.fc5 = nn.Linear(128,1024)


        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(512)
        
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
    
class PointNet_3Multihead(nn.Module):
    def __init__(self, classes = 10, optimizer = "SGD"):
        super().__init__()
        self.transform = Transform()
        self.transform2 = Transform()
        self.transform3 = Transform()
        
        self.fc1 = nn.Linear(3072, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, classes)
        

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(p=0.3)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        
        self.optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.9)
        if optimizer == "SGD":
            self.optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.9)
        else:
            self.optimizer = optim.Adam(self.parameters(), lr=0.00001)
        self.loss_fn = nn.CrossEntropyLoss(weight = torch.tensor([0.4,0.6]))
        self.running_loss = 0
        self.loss = None
        self.losses = []

    def forward(self, input1, input2, input3):
        #xb, matrix3x3, matrix64x64 = self.transform(input)
        xa, matrix3x3, matrix64x64 = self.transform(input1)
        xb, matrix3x3, matrix64x64 = self.transform2(input2)
        xc, matrix3x3, matrix64x64 = self.transform3(input3)
        
        
        xd = torch.cat((xa, xb), dim=1)
        xd = torch.cat((xd, xc), dim=1)
        xd = F.relu(self.bn1(self.fc1(xd)))
        xd = F.relu(self.bn2(self.dropout(self.fc2(xd))))
        output = self.fc3(xd)
        return self.logsoftmax(output)
    
    def fit(self, input1, input2, input3, targets):
        #train/optimize/fit
        #print(x.size)
        # x = x.to(torch.float32)
        input1 = input1.squeeze(1).permute(0, 2, 1)
        input2 = input2.squeeze(1).permute(0, 2, 1)
        input3 = input3.squeeze(1).permute(0, 2, 1)
        targets = targets.to(torch.long)
        #print(targets)
        preds = self.forward(input1,input2,input3)
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

class DNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, hidden_size2, output_size, optimizer = "SGD"):
        super(DNNModel, self).__init__()

        # Define layers
        self.Flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, output_size)
        self.loss_fn = nn.CrossEntropyLoss()
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.dropout = nn.Dropout(p=0.3)
        
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size2)
        
        self.optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.9)
        if optimizer == "SGD":
            self.optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.9)
        else:
            self.optimizer = optim.Adam(self.parameters(), lr=0.00001)
        self.running_loss = 0
        self.loss = None
        self.losses = []

    def forward(self, x, target =None):
        # Forward pass
        #x = self.Flatten(x)
        x = self.bn1(self.fc1(x))
        x = self.relu(x)
        x = self.bn2(self.fc2(x))
        x = self.relu(x)
        x = self.fc3(x)
        
        return x
    
class PointNet_3Multihead_withDNN(nn.Module):
    def __init__(self, classes = 10, optimizer = "SGD"):
        super().__init__()
        self.transform = Transform()
        self.transform2 = Transform()
        self.transform4 = Transform_2Multi()
        self.transform5 = Transform_2Multi()
        
        self.transform3 = DNNModel(172,512,256,128)
        
        self.fc1 = nn.Linear(1152, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, classes)
        

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(p=0.3)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        
        self.optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.9)
        if optimizer == "SGD":
            self.optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.9)
        else:
            self.optimizer = optim.Adam(self.parameters(), lr=0.00001)
            
        self.loss_fn = nn.CrossEntropyLoss(weight = torch.tensor([0.4,0.6]))
        self.running_loss = 0
        self.loss = None
        self.losses = []

    def forward(self, input1, input2, input3):
        #xb, matrix3x3, matrix64x64 = self.transform(input)
        xa, matrix3x3, matrix64x64 = self.transform4(input1)
        xb, matrix3x3, matrix64x64 = self.transform5(input2)
        xc = self.transform3(input3)
        
        
        xd = torch.cat((xa, xb), dim=1)
        xd = torch.cat((xd, xc), dim=1)
        xd = F.relu(self.bn1(self.fc1(xd)))
        xd = F.relu(self.bn2(self.dropout(self.fc2(xd))))
        output = self.fc3(xd)
        return self.logsoftmax(output)
    
    def fit(self, input1, input2, input3, targets):
        #train/optimize/fit
        #print(x.size)
        # x = x.to(torch.float32)
        input1 = input1.squeeze(1).permute(0, 2, 1)
        input2 = input2.squeeze(1).permute(0, 2, 1)
        targets = targets.to(torch.long)
        #print(targets)
        preds = self.forward(input1,input2,input3)
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

class PointNet_2Multimodal_withDNN(nn.Module):
    def __init__(self, classes = 10, optimizer = "SGD"):
        super().__init__()
        self.transform = Transform()
        self.transform2 = Transform_2Multi()
        self.transform3 = DNNModel(172,512,256,128)
        
        self.fc1 = nn.Linear(640, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, classes)
        

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(p=0.3)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        
        self.optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.9)
        if optimizer == "SGD":
            self.optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.9)
        else:
            self.optimizer = optim.Adam(self.parameters(), lr=0.00001)
        self.loss_fn = nn.CrossEntropyLoss(weight = torch.tensor([0.4,0.6]))
        self.running_loss = 0
        self.loss = None
        self.losses = []

    def forward(self, input1, input2):
        #xb, matrix3x3, matrix64x64 = self.transform(input)
        xa, matrix3x3, matrix64x64 = self.transform2(input1)
        xb = self.transform3(input2)
        
        
        xd = torch.cat((xa, xb), dim=1)
        xd = F.relu(self.bn1(self.fc1(xd)))
        xd = F.relu(self.bn2(self.dropout(self.fc2(xd))))
        output = self.fc3(xd)
        return self.logsoftmax(output)
    
    def fit(self, input1, input2, targets):
        #train/optimize/fit
        #print(x.size)
        # x = x.to(torch.float32)
        input1 = input1.squeeze(1).permute(0, 2, 1)
        targets = targets.to(torch.long)
        #print(targets)
        preds = self.forward(input1,input2)
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
    
class PointNet_2Branch(nn.Module):
    def __init__(self, classes = 10, optimizer = "SGD"):
        super().__init__()
        self.transform = Transform()
        self.transform2 = Transform_2Multi()
        self.transform3 = DNNModel(172,512,256,128)
        
        self.fc1 = nn.Linear(640, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, classes)
        

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(p=0.3)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        
        self.optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.9)
        if optimizer == "SGD":
            self.optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.9)
        else:
            self.optimizer = optim.Adam(self.parameters(), lr=0.00001)
            
        self.loss_fn = nn.CrossEntropyLoss(weight = torch.tensor([0.4,0.6]))
        self.running_loss = 0
        self.loss = None
        self.losses = []

    def forward(self, input1, input2):
        #xb, matrix3x3, matrix64x64 = self.transform(input)
        xa, matrix3x3, matrix64x64 = self.transform2(input1)
        xb = self.transform3(input2)
        
        
        xd = torch.cat((xa, xb), dim=1)
        xd = F.relu(self.bn1(self.fc1(xd)))
        xd = F.relu(self.bn2(self.dropout(self.fc2(xd))))
        output = self.fc3(xd)
        return self.logsoftmax(output)
    
    def fit(self, input1, input2, targets):
        #train/optimize/fit
        #print(x.size)
        # x = x.to(torch.float32)
        input1 = input1.squeeze(1).permute(0, 2, 1)
        targets = targets.to(torch.long)
        #print(targets)
        preds = self.forward(input1,input2)
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
    
class PointNet_3Branch(nn.Module):
    def __init__(self, classes = 10, optimizer = "SGD"):
        super().__init__()
        self.transform = Transform()
        self.transform2 = Transform()
        self.transform4 = Transform_2Multi()
        self.transform5 = Transform_2Multi()
        
        self.transform3 = DNNModel(172,512,256,128)
        
        self.fc1 = nn.Linear(1152, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, classes)
        

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(p=0.3)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        
        self.optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.9)
        if optimizer == "SGD":
            self.optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.9)
        else:
            self.optimizer = optim.Adam(self.parameters(), lr=0.00001)
            
        self.loss_fn = nn.CrossEntropyLoss(weight = torch.tensor([0.4,0.6]))
        self.running_loss = 0
        self.loss = None
        self.losses = []

    def forward(self, input1, input2, input3):
        #xb, matrix3x3, matrix64x64 = self.transform(input)
        xa, matrix3x3, matrix64x64 = self.transform4(input1)
        xb, matrix3x3, matrix64x64 = self.transform5(input2)
        xc = self.transform3(input3)
        
        
        xd = torch.cat((xa, xb), dim=1)
        xd = torch.cat((xd, xc), dim=1)
        xd = F.relu(self.bn1(self.fc1(xd)))
        xd = F.relu(self.bn2(self.dropout(self.fc2(xd))))
        output = self.fc3(xd)
        return self.logsoftmax(output)
    
    def fit(self, input1, input2, input3, targets):
        #train/optimize/fit
        #print(x.size)
        # x = x.to(torch.float32)
        input1 = input1.squeeze(1).permute(0, 2, 1)
        input2 = input2.squeeze(1).permute(0, 2, 1)
        targets = targets.to(torch.long)
        #print(targets)
        preds = self.forward(input1,input2,input3)
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
    
    