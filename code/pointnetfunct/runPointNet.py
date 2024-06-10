import vtk
import os
import numpy as np
import itertools
import math, random
import copy

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
from torchvision import transforms, utils
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch import optim
import open3d as o3d
import vtk

import scipy.spatial.distance
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from .PointNet_struct import PointNet,PointNet_2Multihead,PointNet_3Multihead
import seaborn as sns
from . import functions as fun
from .PointNet_dataset import Aneuxmodel_Dataset,Aneux_Dataset_save,Aneux_Dataset_load
import data_process_ml
from .PointNet_trainingfunct import run_model_get,run_model_2multi_head,run_model_3multi_head,show_graph,save_model,load_model,dim4_cm_dl



#parameter data load
root = "..\..\msc_data\models-v1.0\models"
IA = "aneurysms\\remeshed\\area-001"
Vessel = "vessels\\remeshed\\area-001"
IA_root = os.path.join(root,IA)
Vessel_root = os.path.join(root,Vessel)
list1 = os.listdir(Vessel_root)
list2 = os.listdir(IA_root)

morpho_path = ".\AneuX\data-v1.0\data\morpho-per-cut.csv"
patient_path = ".\AneuX\data-v1.0\data\clinical.csv"
morpho_data_patient = data_process_ml.read_and_combine_data(morpho_path,patient_path)
merged_dataset = data_process_ml.encode_column(morpho_data_patient)
merged_dataset = data_process_ml.drop_columns(merged_dataset)
morpho_data_cut1,morpho_data_dome = data_process_ml.output_cut1anddome(merged_dataset)

morpho_data_patient[morpho_data_patient["cuttype"] == "cut1"]

#3D data load
import pandas as pd
df = pd.DataFrame()
# Aneux_Dataset = Aneuxmodel_Dataset(root = root,
#                                    df=morpho_data_patient[morpho_data_patient["cuttype"] == "dome"],
#                                    transform = transforms.ToTensor(),
#                                    mesh = "area-001",
#                                    cuttype = "dome",
#                                    crop = False,
#                                    points = 1000,
#                                    limit = 700)
Aneux_Dataset = Aneux_Dataset_load("./Aneux_Dataset_1000pt_sample_clip")

print(len(Aneux_Dataset.model_table),len(Aneux_Dataset.label))

# load dataset, transfer the open3d pointnet to the tensotflow object
train_size = int(len(Aneux_Dataset) * 0.8) # 80% training data
valid_size = len(Aneux_Dataset) - train_size
train_data, valid_data = random_split(Aneux_Dataset, [train_size, valid_size])

#training loader
train_loader = torch.utils.data.DataLoader(
    train_data,
    batch_size=50,
    shuffle=True,
    #num_workers=2, 
    pin_memory=True
)

valid_loader = torch.utils.data.DataLoader(
    valid_data,
    batch_size=50, # Forward pass only so batch size can be larger
    shuffle=False,
    #num_workers=2, 
    pin_memory=True
)

#training pointNet
pointnet = PointNet(classes=2)
run_model_get(train_loader_input = train_loader,
                        vaild_loader_input = valid_loader,
                        nepochs = 50, 
                        modelnet = pointnet,
                        results_path = "./result", 
                        filename = "/dome_test.pt")


# pointnet_mh = PointNet_2Multihead(classes=2)
# run_model_2multi_head(train_loader_input = train_loader,
#                         vaild_loader_input = valid_loader,
#                         nepochs = 50, 
#                         modelnet = pointnet_mh,
#                         results_path = "./result", 
#                         filename = "/2multibranch_test.pt")

# pointnet_mh = PointNet_3Multihead(classes=3)
# run_model_3multi_head(train_loader_input = train_loader,
#                         vaild_loader_input = valid_loader,
#                         nepochs = 50, 
#                         modelnet = pointnet_mh,
#                         results_path = "./result", 
#                         filename = "/3multibranch_test.pt")




pointnet_copy = PointNet(classes=2)
model_path = "./Ckp_point/pointnet_dome_lr0.0001_39epoch_test.pth"

#show graph
show_graph(path ="./result/dome_test.pt",device = "cuda")
#save model
model_path = "pointnet_dome_test.pth"
torch.save(pointnet.state_dict(), model_path)
#load model
load_model(pointnet_copy,model_path)


# y_true = []
# y_pred = []
# y_pred_result = []
# pointnet_copy = pointnet_copy.to("cpu")

# #basic pointnet
# with torch.no_grad():
#     for inputs_v,_,labels in valid_loader:
#         inputs_v = inputs_v.to(torch.float32)
#         inputs_v = inputs_v.squeeze(1).permute(0, 2, 1)
#         labels = labels.to(torch.long)
#         outputs = pointnet_copy.forward(inputs_v)
        
#         #print(torch.exp(outputs))

#         outputs_value = (torch.max(torch.exp(outputs), 1)[0]).data.cpu().numpy()
#         outputs_result = (torch.max(torch.exp(outputs), 1)[1]).data.cpu().numpy()
        
        
#         #print(outputs)
#         y_pred.extend(outputs_value)
#         y_pred_result.extend(outputs_result)
            
#         labels = labels.data.cpu().numpy()
#         y_true.extend(labels) 


# cm = dim4_cm_dl(y_true, y_pred,y_pred_result)  
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Pred Unrupture","Pred uncertain Unrupture","Pred uncertain Rupture","Pred Rupture"], yticklabels=["Real Unrupture"," Real Rupture"])
# #fun.show_pred_cm(cm,y_true)  
# # classes = ["Unrupture","Rupture"]    
# # cf_matrix = confusion_matrix(y_true, y_pred)
# # plt.figure(figsize=(10, 8)) 
# # sns.heatmap(cf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes,annot_kws={"size": 9})
# # plt.xlabel("Predicted Labels")
# # plt.ylabel("True Labels")
# plt.title("PointNet Training set Confusion Matrix")
# plt.show()