from . import data_process_ml
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.hub import load_state_dict_from_url
import sklearn
import os
import pandas as pd
import copy

import matplotlib.pyplot as plt
from torch.utils.data import Dataset, random_split
from torch import optim

device = "cuda"

morpho_path = ".\AneuX\data-v1.0\data\morpho-per-cut.csv"
patient_path = ".\AneuX\data-v1.0\data\clinical.csv"

morpho_data_patient = data_process_ml.read_and_combine_data(morpho_path,patient_path)
merged_dataset = data_process_ml.encode_column(morpho_data_patient)
merged_dataset= data_process_ml.drop_columns(merged_dataset)
merged_dataset= data_process_ml.output_cut1anddome(merged_dataset)

class AneuxDataset_Morph(Dataset):

    def __init__(self, df, transform = None):

        self.transform = transform
        self.df = df.astype(float)
        self.label = df["status_ruptured"].copy()
        self.label = torch.from_numpy(np.array(self.label, dtype= int)) 
        self.raw_data = copy.deepcopy(self.df)
        self.raw_data.loc[self.raw_data['sex_male'] == 0, 'sex_male'] = 2
        #print(self.raw_data.iloc[0])
        self.raw_data.drop(("status_ruptured"),axis=1,inplace=True)
        self.raw_data.drop(("sex_female"),axis=1,inplace=True)
        self.raw_data['age'].fillna(self.raw_data['age'].mean(), inplace=True)
        #self.raw_data.drop(("age"),axis=1,inplace=True)
        self.raw_data = self.raw_data
        self.my_device = "cuda:0"
    
    def __getitem__(self, index):
        
        """ Returns one data pair (image and target caption). """
        
        data = torch.from_numpy(np.array(self.raw_data.iloc[index],dtype = np.float64))
        label = self.label[index]
        #print(type(image))
        if self.transform is not None:      
            data= self.transform(data)
            
        return data,label

    def __len__(self):
                                
        return len(self.label)
    
def stats(loader, net):
    correct = 0
    total = 0
    running_loss = 0
    n = 0    # counter for number of minibatches
    with torch.no_grad():
        for data in loader:
            inputs_v, labels = data
            
            loss_fn = nn.CrossEntropyLoss()

            #to work with gpu you will need to load data and labels to gpu
            inputs_v = inputs_v.to(device)
            labels = labels.to(device)
            
            inputs_v = inputs_v.to(torch.float32)
            labels = labels.to(torch.long)

            outputs = net.forward(inputs_v)

            # accumulate loss
            running_loss += loss_fn(outputs, labels)
            n += 1

            # accumulate data for accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)    # add in the number of labels in this minibatch
            correct += (predicted == labels).sum().item()  # add in the number of correct labels

    return running_loss/n, correct/total


# run model
def run_model_get(train_loader_input,vaild_loader_input,nepochs, modelnet,results_path, filename, transform = None,dataset =None):
    os.makedirs(results_path, exist_ok = True)
    saveCkpt = results_path + filename
    statsrec = np.zeros((4,nepochs))
    modelnet = modelnet.to(device)
    max_atst = 0
    for epoch in range(nepochs):  # loop over the dataset multiple times
        correct = 0          # number of examples predicted correctly (for accuracy)
        total = 0            # number of examples
        epoch_loss = 0.0   # accumulated loss (for mean loss)
        n = 0
        
        if transform != None:
            dataset.transform_image(transform)  
                        
        for data in train_loader_input:
            inputs, labels = data
            #print(inputs, labels)
            
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            inputs = inputs.to(torch.float32)
            labels = labels.to(torch.float32)
            #print(labels)
            modelnet.optimizer.zero_grad()
            # Forward, backward, and update parameters
            loss = modelnet.fit(inputs, labels) # note: .to(device) helps to load data to your gpu
            # accumulate loss
            epoch_loss += loss
            n += 1
            outputs = modelnet.forward(inputs)
            # accumulate data for accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)    # add in the number of labels in this minibatch
            correct += (predicted == labels).sum().item()  # add in the number of correct labels

        # collect together statistics for this epoch
        ltrn = epoch_loss/n
        atrn = correct/total
        ltst, atst = stats(vaild_loader_input, modelnet)

        ltst = ltst.item() #item() moves the tensor data with 1 element to CPU

        statsrec[:,epoch] = (ltrn, atrn, ltst, atst)
        print(f"epoch: {epoch} training loss: {ltrn: .3f} training accuracy: {atrn: .1%}  test loss: {ltst: .3f} test accuracy: {atst: .1%}")
        
        if atst > max_atst:
            model_path = "DNN_dome_lr0.0001_epoch_test.pth"
            torch.save(modelnet.state_dict(), model_path)
            max_atst = atst

    # save network parameters, losses and accuracy
    torch.save({"state_dict": modelnet.state_dict(), "stats": statsrec}, saveCkpt)
    
    
class DNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, hidden_size2, output_size):
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
        
        #Adam
        #self.optimizer = optim.Adam(self.parameters(), lr=0.001) #weight decay
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.running_loss = 0
        self.loss = None
        self.losses = []

    def forward(self, x, target =None):
        # Forward pass
        #x = self.Flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        
        return x
    def fit(self, x, targets):
        #train/optimize/fit
        preds = self.forward(x)
        #print(preds, targets)
        self.loss = self.loss_fn(preds, targets.long())
        self.loss.backward()
        self.optimizer.step()
        
        loss_item = self.loss.item()
        return loss_item


    def reset_loss(self, value):
        self.running_loss = value
        self.losses = []
        return