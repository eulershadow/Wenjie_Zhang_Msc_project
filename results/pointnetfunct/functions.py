
import torch
import torch.nn as nn
import matplotlib as plt
import os
import numpy as np
import copy
import seaborn as sns

device = "cuda"



def save_model(net,model_path):
    model_path = model_path
    torch.save(net.state_dict(), model_path)
    return True

def load_model(net,model_path):
    net_copy = copy.deepcopy(net)
    model_path = model_path
    net_copy.load_state_dict(torch.load(model_path))
    return net_copy

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
    
def test():
    print("test4")
    
def dim4_cm (real, pred):

    cm = [[0,0,0,0],[0,0,0,0]]
    for i in range(len(pred)):
        pos = 0
        if float(pred[i]) < 0.25:
            pos = 0
        elif 0.25 <= float(pred[i]) < 0.5:
            pos = 1
        elif 0.8 > float(pred[i]) >= 0.5:
            pos = 2
        elif float(pred[i]) >= 0.8:
            pos = 3
            
        if real[i] == 0:

            cm[0][pos] += 1
        else:
            cm[1][pos] += 1
    return cm

def show_pred_cm(data_predictions, test_set_target):
    #cm = confusion_matrix(list(test_set_target), data_predictions)
    cm = dim4_cm(list(test_set_target), data_predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Pred Unrupture","Pred uncertain Unrupture","Pred uncertain Rupture","Pred Rupture"], yticklabels=["Real Unrupture"," Real Rupture"])