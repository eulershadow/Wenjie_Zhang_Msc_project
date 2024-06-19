import vtk
import os
import numpy as np
import math, random
from . import data_process_ml
random.seed = 42
import copy

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
from torchvision import transforms, utils

import scipy.spatial.distance
# import plotly.graph_objects as go
# import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch import optim


device = "cuda"
def stats(loader, net):
    correct = 0
    total = 0
    running_loss = 0
    n = 0    # counter for number of minibatches
    with torch.no_grad():
        for data in loader:
            inputs_v, _, _, labels = data
            
            loss_fn = nn.CrossEntropyLoss()

            #to work with gpu you will need to load data and labels to gpu
            inputs_v = inputs_v.to(device)
            labels = labels.to(device)
            
            inputs_v = inputs_v.to(torch.float32)
            inputs_v = inputs_v.squeeze(1).permute(0, 2, 1)
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
def run_model_get(train_loader_input,vaild_loader_input,nepochs, modelnet,results_path, filename, transform = None,dataset =None, model_name = "None"):
    os.makedirs(results_path, exist_ok = True)
    saveCkpt = results_path + filename
    statsrec = np.zeros((4,nepochs))
    modelnet = modelnet.to(device)
    max_atst = 0
    max_loss = 100
    for epoch in range(nepochs):  # loop over the dataset multiple times
        correct = 0          # number of examples predicted correctly (for accuracy)
        total = 0            # number of examples
        epoch_loss = 0.0   # accumulated loss (for mean loss)
        n = 0
        
        if transform != None:
            dataset.transform_image(transform)  
                        
        for data in train_loader_input:
            inputs, _, _, labels = data
            #noised_inputs=torch.randn_like(inputs)+inputs
            #to work with gpu you will need to load data and labels to gpu
            
            
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            inputs = inputs.to(torch.float32)
            inputs = inputs.squeeze(1).permute(0, 1, 2)
            labels = labels.to(torch.float32)
            #print(labels)
            modelnet.optimizer.zero_grad()
            # Forward, backward, and update parameters
            loss = modelnet.fit(inputs, labels) # note: .to(device) helps to load data to your gpu
            # accumulate loss
            epoch_loss += loss
            n += 1
            inputs = inputs.squeeze(1).permute(0, 2, 1)
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
        
        # if atst > max_atst:
        #     model_path = model_name
        #     torch.save(modelnet.state_dict(), model_path)
        #     max_atst = atst
        #Save the model with lowest loss
        if ltst < max_loss:
            model_path = model_name
            torch.save(modelnet.state_dict(), model_path)
            max_loss = ltst

    # save network parameters, losses and accuracy
    torch.save({"state_dict": modelnet.state_dict(), "stats": statsrec}, saveCkpt)

def stats_2mh(loader, net):
    correct = 0
    total = 0
    running_loss = 0
    n = 0    # counter for number of minibatches
    with torch.no_grad():
        for data in loader:
            inputs_v, inputs_v2,_, labels = data
            
            loss_fn = nn.CrossEntropyLoss()

            #to work with gpu you will need to load data and labels to gpu
            inputs_v = inputs_v.to(device)
            inputs_v2 = inputs_v2.to(device)
            labels = labels.to(device)
            
            inputs_v = inputs_v.to(torch.float32)
            inputs_v = inputs_v.squeeze(1).permute(0, 2, 1)
            inputs_v2 = inputs_v2.to(torch.float32)
            inputs_v2 = inputs_v2.squeeze(1).permute(0, 2, 1)
            labels = labels.to(torch.long)

            outputs = net.forward(inputs_v,inputs_v2)

            # accumulate loss
            running_loss += loss_fn(outputs, labels)
            n += 1

            # accumulate data for accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)    # add in the number of labels in this minibatch
            correct += (predicted == labels).sum().item()  # add in the number of correct labels

    return running_loss/n, correct/total

def run_model_2multi_head(train_loader_input,vaild_loader_input,nepochs, modelnet,results_path, filename, transform = None,dataset =None, model_name = "None"):
    os.makedirs(results_path, exist_ok = True)
    saveCkpt = results_path + filename
    statsrec = np.zeros((4,nepochs))
    modelnet = modelnet.to(device)
    max_atst = 0
    max_loss = 100
    for epoch in range(nepochs):  # loop over the dataset multiple times
        correct = 0          # number of examples predicted correctly (for accuracy)
        total = 0            # number of examples
        epoch_loss = 0.0   # accumulated loss (for mean loss)
        n = 0
        
        if transform != None:
            dataset.transform_image(transform)  
                        
        for data in train_loader_input:
            inputs, inputs2, _, labels = data
            #noised_inputs=torch.randn_like(inputs)+inputs
            #to work with gpu you will need to load data and labels to gpu
            
            
            inputs = inputs.to(device)
            inputs2 = inputs2.to(device)
            labels = labels.to(device)
            
            inputs = inputs.to(torch.float32)
            inputs = inputs.squeeze(1).permute(0, 1, 2)
            
            inputs2 = inputs2.to(torch.float32)
            inputs2 = inputs2.squeeze(1).permute(0, 1, 2)
            
            labels = labels.to(torch.float32)
            #print(labels)
            modelnet.optimizer.zero_grad()
            # Forward, backward, and update parameters
            loss = modelnet.fit(inputs, inputs2, labels) # note: .to(device) helps to load data to your gpu
            # accumulate loss
            epoch_loss += loss
            n += 1
            inputs = inputs.squeeze(1).permute(0, 2, 1)
            inputs2 = inputs2.squeeze(1).permute(0, 2, 1)
            outputs = modelnet.forward(inputs, inputs2)
            # accumulate data for accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)    # add in the number of labels in this minibatch
            correct += (predicted == labels).sum().item()  # add in the number of correct labels

        # collect together statistics for this epoch
        ltrn = epoch_loss/n
        atrn = correct/total
        ltst, atst = stats_2mh(vaild_loader_input, modelnet)

        ltst = ltst.item() #item() moves the tensor data with 1 element to CPU

        statsrec[:,epoch] = (ltrn, atrn, ltst, atst)
        print(f"epoch: {epoch} training loss: {ltrn: .3f} training accuracy: {atrn: .1%}  test loss: {ltst: .3f} test accuracy: {atst: .1%}")
        
        #Save the model with highest accuracy
        # if atst > max_atst:
        #     model_path = model_name
        #     torch.save(modelnet.state_dict(), model_path)
        #     max_atst = atst
        #Save the model with lowest loss
        if ltst < max_loss:
            model_path = model_name
            torch.save(modelnet.state_dict(), model_path)
            max_loss = ltst

    # save network parameters, losses and accuracy
    torch.save({"state_dict": modelnet.state_dict(), "stats": statsrec}, saveCkpt)
    
def stats_3mh(loader, net):
    correct = 0
    total = 0
    running_loss = 0
    n = 0    # counter for number of minibatches
    with torch.no_grad():
        for data in loader:
            inputs_v, inputs_v2, inputs_v3,labels = data
            
            loss_fn = nn.CrossEntropyLoss()

            #to work with gpu you will need to load data and labels to gpu
            inputs_v = inputs_v.to(device)
            inputs_v2 = inputs_v2.to(device)
            inputs_v3 = inputs_v3.to(device)
            labels = labels.to(device)
            
            inputs_v = inputs_v.to(torch.float32)
            inputs_v = inputs_v.squeeze(1).permute(0, 2, 1)
            inputs_v2 = inputs_v2.to(torch.float32)
            inputs_v2 = inputs_v2.squeeze(1).permute(0, 2, 1)
            inputs_v3 = inputs_v3.to(torch.float32)
            inputs_v3 = inputs_v3.squeeze(1).permute(0, 2, 1)
            
            labels = labels.to(torch.long)

            outputs = net.forward(inputs_v,inputs_v2,inputs_v3)

            # accumulate loss
            running_loss += loss_fn(outputs, labels)
            n += 1

            # accumulate data for accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)    # add in the number of labels in this minibatch
            correct += (predicted == labels).sum().item()  # add in the number of correct labels

    return running_loss/n, correct/total

def run_model_3multi_head(train_loader_input,vaild_loader_input,nepochs, modelnet,results_path, filename, transform = None,dataset =None, model_name = "None"):
    os.makedirs(results_path, exist_ok = True)
    saveCkpt = results_path + filename
    statsrec = np.zeros((4,nepochs))
    modelnet = modelnet.to(device)
    max_atst = 0
    max_loss = 100
    for epoch in range(nepochs):  # loop over the dataset multiple times
        correct = 0          # number of examples predicted correctly (for accuracy)
        total = 0            # number of examples
        epoch_loss = 0.0   # accumulated loss (for mean loss)
        n = 0
        
        if transform != None:
            dataset.transform_image(transform)  
                        
        for data in train_loader_input:
            inputs, inputs2, inputs3, labels = data
            #noised_inputs=torch.randn_like(inputs)+inputs
            #to work with gpu you will need to load data and labels to gpu
            
            
            inputs = inputs.to(device)
            inputs2 = inputs2.to(device)
            inputs3 = inputs3.to(device)
            labels = labels.to(device)
            
            inputs = inputs.to(torch.float32)
            inputs = inputs.squeeze(1).permute(0, 1, 2)
            
            inputs2 = inputs2.to(torch.float32)
            inputs2 = inputs2.squeeze(1).permute(0, 1, 2)
            
            inputs3 = inputs3.to(torch.float32)
            inputs3 = inputs3.squeeze(1).permute(0, 1, 2)
            
            labels = labels.to(torch.float32)
            #print(labels)
            modelnet.optimizer.zero_grad()
            # Forward, backward, and update parameters
            loss = modelnet.fit(inputs, inputs2, inputs3, labels) # note: .to(device) helps to load data to your gpu
            # accumulate loss
            epoch_loss += loss
            n += 1
            inputs = inputs.squeeze(1).permute(0, 2, 1)
            inputs2 = inputs2.squeeze(1).permute(0, 2, 1)
            inputs3 = inputs3.squeeze(1).permute(0, 2, 1)
            outputs = modelnet.forward(inputs, inputs2,inputs3)
            # accumulate data for accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)    # add in the number of labels in this minibatch
            correct += (predicted == labels).sum().item()  # add in the number of correct labels

        # collect together statistics for this epoch
        ltrn = epoch_loss/n
        atrn = correct/total
        ltst, atst = stats_3mh(vaild_loader_input, modelnet)

        ltst = ltst.item() #item() moves the tensor data with 1 element to CPU

        statsrec[:,epoch] = (ltrn, atrn, ltst, atst)
        print(f"epoch: {epoch} training loss: {ltrn: .3f} training accuracy: {atrn: .1%}  test loss: {ltst: .3f} test accuracy: {atst: .1%}")
        
        #Save the model with highest accuracy
        # if atst > max_atst:
        #     model_path = model_name
        #     torch.save(modelnet.state_dict(), model_path)
        #     max_atst = atst
        #Save the model with lowest loss
        if ltst < max_loss:
            model_path = model_name
            torch.save(modelnet.state_dict(), model_path)
            max_loss = ltst

    # save network parameters, losses and accuracy
    torch.save({"state_dict": modelnet.state_dict(), "stats": statsrec}, saveCkpt)
device = "cuda"

def stats_3mh_dnn(loader, net):
    correct = 0
    total = 0
    running_loss = 0
    n = 0    # counter for number of minibatches
    with torch.no_grad():
        for data in loader:
            inputs_v, inputs_v2, inputs_v3,labels = data
            
            loss_fn = nn.CrossEntropyLoss()

            #to work with gpu you will need to load data and labels to gpu
            inputs_v = inputs_v.to(device)
            inputs_v2 = inputs_v2.to(device)
            inputs_v3 = inputs_v3.to(device)
            labels = labels.to(device)
            
            inputs_v = inputs_v.to(torch.float32)
            inputs_v = inputs_v.squeeze(1).permute(0, 2, 1)
            inputs_v2 = inputs_v2.to(torch.float32)
            inputs_v2 = inputs_v2.squeeze(1).permute(0, 2, 1)
            inputs_v3 = inputs_v3.to(torch.float32)
            
            labels = labels.to(torch.long)

            outputs = net.forward(inputs_v,inputs_v2,inputs_v3)

            # accumulate loss
            running_loss += loss_fn(outputs, labels)
            n += 1

            # accumulate data for accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)    # add in the number of labels in this minibatch
            correct += (predicted == labels).sum().item()  # add in the number of correct labels

    return running_loss/n, correct/total

def run_model_3multi_head_dnn(train_loader_input,vaild_loader_input,nepochs, modelnet,results_path, filename, transform = None,dataset =None, model_name = "None"):
    os.makedirs(results_path, exist_ok = True)
    saveCkpt = results_path + filename
    statsrec = np.zeros((4,nepochs))
    modelnet = modelnet.to(device)
    max_atst = 0
    max_loss = 100
    for epoch in range(nepochs):  # loop over the dataset multiple times
        correct = 0          # number of examples predicted correctly (for accuracy)
        total = 0            # number of examples
        epoch_loss = 0.0   # accumulated loss (for mean loss)
        n = 0
        
        if transform != None:
            dataset.transform_image(transform)  
                        
        for data in train_loader_input:
            inputs, inputs2, inputs3, labels = data
            #noised_inputs=torch.randn_like(inputs)+inputs
            #to work with gpu you will need to load data and labels to gpu
            
            
            inputs = inputs.to(device)
            inputs2 = inputs2.to(device)
            inputs3 = inputs3.to(device)
            labels = labels.to(device)
            
            inputs = inputs.to(torch.float32)
            inputs = inputs.squeeze(1).permute(0, 1, 2)
            
            inputs2 = inputs2.to(torch.float32)
            inputs2 = inputs2.squeeze(1).permute(0, 1, 2)
            
            inputs3 = inputs3.to(torch.float32)
            
            
            labels = labels.to(torch.float32)
            #print(labels)
            modelnet.optimizer.zero_grad()
            # Forward, backward, and update parameters
            loss = modelnet.fit(inputs, inputs2, inputs3, labels) # note: .to(device) helps to load data to your gpu
            # accumulate loss
            epoch_loss += loss
            n += 1
            inputs = inputs.squeeze(1).permute(0, 2, 1)
            inputs2 = inputs2.squeeze(1).permute(0, 2, 1)
            outputs = modelnet.forward(inputs, inputs2,inputs3)
            # accumulate data for accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)    # add in the number of labels in this minibatch
            correct += (predicted == labels).sum().item()  # add in the number of correct labels

        # collect together statistics for this epoch
        ltrn = epoch_loss/n
        atrn = correct/total
        ltst, atst = stats_3mh_dnn(vaild_loader_input, modelnet)

        ltst = ltst.item() #item() moves the tensor data with 1 element to CPU

        statsrec[:,epoch] = (ltrn, atrn, ltst, atst)
        print(f"epoch: {epoch} training loss: {ltrn: .3f} training accuracy: {atrn: .1%}  test loss: {ltst: .3f} test accuracy: {atst: .1%}")
        
        #Save the model with highest accuracy
        # if atst > max_atst:
        #     model_path = model_name
        #     torch.save(modelnet.state_dict(), model_path)
        #     max_atst = atst
        #Save the model with lowest loss
        if ltst < max_loss:
            model_path = model_name
            torch.save(modelnet.state_dict(), model_path)
            max_loss = ltst

    # save network parameters, losses and accuracy
    torch.save({"state_dict": modelnet.state_dict(), "stats": statsrec}, saveCkpt)


def stats_2mh_dnn(loader, net):
    correct = 0
    total = 0
    running_loss = 0
    n = 0    # counter for number of minibatches
    with torch.no_grad():
        for data in loader:
            inputs_v, inputs_v2, inputs_v3,labels = data
            
            loss_fn = nn.CrossEntropyLoss()

            #to work with gpu you will need to load data and labels to gpu
            inputs_v = inputs_v.to(device)
            inputs_v2 = inputs_v2.to(device)
            inputs_v3 = inputs_v3.to(device)
            labels = labels.to(device)
            
            inputs_v = inputs_v.to(torch.float32)
            inputs_v = inputs_v.squeeze(1).permute(0, 2, 1)
            inputs_v2 = inputs_v2.to(torch.float32)
            inputs_v2 = inputs_v2.squeeze(1).permute(0, 2, 1)
            inputs_v3 = inputs_v3.to(torch.float32)
            
            labels = labels.to(torch.long)

            outputs = net.forward(inputs_v,inputs_v3)

            # accumulate loss
            running_loss += loss_fn(outputs, labels)
            n += 1

            # accumulate data for accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)    # add in the number of labels in this minibatch
            correct += (predicted == labels).sum().item()  # add in the number of correct labels

    return running_loss/n, correct/total

def run_model_2multi_head_dnn(train_loader_input,vaild_loader_input,nepochs, modelnet,results_path, filename, transform = None,dataset =None, model_name = "None"):
    os.makedirs(results_path, exist_ok = True)
    saveCkpt = results_path + filename
    statsrec = np.zeros((4,nepochs))
    modelnet = modelnet.to(device)
    max_atst = 0
    max_loss = 100
    for epoch in range(nepochs):  # loop over the dataset multiple times
        correct = 0          # number of examples predicted correctly (for accuracy)
        total = 0            # number of examples
        epoch_loss = 0.0   # accumulated loss (for mean loss)
        n = 0
        
        if transform != None:
            dataset.transform_image(transform)  
                        
        for data in train_loader_input:
            inputs, inputs2, inputs3, labels = data
            #noised_inputs=torch.randn_like(inputs)+inputs
            #to work with gpu you will need to load data and labels to gpu
            
            
            inputs = inputs.to(device)
            inputs2 = inputs2.to(device)
            inputs3 = inputs3.to(device)
            labels = labels.to(device)
            
            inputs = inputs.to(torch.float32)
            inputs = inputs.squeeze(1).permute(0, 1, 2)
            
            inputs2 = inputs2.to(torch.float32)
            inputs2 = inputs2.squeeze(1).permute(0, 1, 2)
            
            inputs3 = inputs3.to(torch.float32)
            
            
            labels = labels.to(torch.float32)
            #print(labels)
            modelnet.optimizer.zero_grad()
            # Forward, backward, and update parameters
            loss = modelnet.fit(inputs, inputs3, labels) # note: .to(device) helps to load data to your gpu
            # accumulate loss
            epoch_loss += loss
            n += 1
            inputs = inputs.squeeze(1).permute(0, 2, 1)
            inputs2 = inputs2.squeeze(1).permute(0, 2, 1)
            outputs = modelnet.forward(inputs,inputs3)
            # accumulate data for accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)    # add in the number of labels in this minibatch
            correct += (predicted == labels).sum().item()  # add in the number of correct labels

        # collect together statistics for this epoch
        ltrn = epoch_loss/n
        atrn = correct/total
        ltst, atst = stats_2mh_dnn(vaild_loader_input, modelnet)

        ltst = ltst.item() #item() moves the tensor data with 1 element to CPU

        statsrec[:,epoch] = (ltrn, atrn, ltst, atst)
        print(f"epoch: {epoch} training loss: {ltrn: .3f} training accuracy: {atrn: .1%}  test loss: {ltst: .3f} test accuracy: {atst: .1%}")
        
        #Save the model with highest accuracy
        # if atst > max_atst:
        #     model_path = model_name
        #     torch.save(modelnet.state_dict(), model_path)
        #     max_atst = atst
        #Save the model with lowest loss
        if ltst < max_loss:
            model_path = model_name
            torch.save(modelnet.state_dict(), model_path)
            max_loss = ltst

    # save network parameters, losses and accuracy
    torch.save({"state_dict": modelnet.state_dict(), "stats": statsrec}, saveCkpt)
    
    
def save_model(net,model_path):
    model_path = model_path
    torch.save(net.state_dict(), model_path)
    return True

def load_model(net,model_path):
    #net_copy = copy.deepcopy(net)
    model_path = model_path
    net.load_state_dict(torch.load(model_path))
    return net

def show_graph(path,device):
    data = torch.load(path, map_location=device) # by doing map_location=device, you can use trained model on GPU --> to test on CPU
    statsrec = data["stats"]
    fig, ax1 = plt.subplots()
    plt.plot(statsrec[0], 'r', label = 'training loss', )
    plt.plot(statsrec[2], 'g', label = 'test loss' )
    plt.legend(loc='lower right')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Training and test loss, and test accuracy')
    ax2=ax1.twinx()
    ax2.plot(statsrec[1], 'm', label = 'training accuracy')
    ax2.plot(statsrec[3], 'b', label = 'test accuracy')
    ax2.set_ylabel('accuracy')
    plt.legend(loc='upper right')
    fig.savefig("roc.svg")
    plt.show()
    
def dim4_cm_dl (real, pred,pred_result, precent = 0.6):

    cm = [[0,0,0,0],[0,0,0,0]]
    for i in range(len(pred)):
        pos = 0
        if float(pred[i]) < 0.6 and pred_result[i] == 0:
            pos = 1
        elif float(pred[i]) >= 0.6 and pred_result[i] == 0:
            pos = 0
        if float(pred[i]) < 0.6 and pred_result[i] == 1:
            pos = 2
        elif float(pred[i]) >= 0.6 and pred_result[i] == 1:
            pos = 3
            
            
        if real[i] == 0:

            cm[0][pos] += 1
        else:
            cm[1][pos] += 1
    return cm

    
    
    