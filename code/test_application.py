
import vtk
import os
import numpy as np
import itertools
import math, random

random.seed = 42
import copy

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
from torchvision import transforms, utils
from torch.utils.data import Subset
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

import scipy.spatial.distance
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
device = "cuda"
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pointnetfunct.functions as fun
import pointnetfunct.data_process_ml as data_process_ml

from pointnetfunct.PointNet_dataset import Aneux_Dataset_load
from pointnetfunct.PointNet_trainingfunct import run_model_3multi_head_dnn
from pointnetfunct.PointNet_trainingfunct import run_model_get
from pointnetfunct.PointNet_trainingfunct import load_model
from pointnetfunct.PointNet_struct import Tnet,Transform,PointNet,PointNet_2Multihead,PointNet_3Multihead,PointNet_3Multihead_withDNN,PointNet_2Multimodal_withDNN
from pointnetfunct.evaluation import show_cm_dl,show_graph,print_values,draw_rocgraph
from pointnetfunct.test_functions import pointnet_2branch_testcase,pointnet_3_branch_testcase,pointnet_testcase

#load datasets

morpho_path = ".\AneuX\data-v1.0\data\morpho-per-cut.csv"
patient_path = ".\AneuX\data-v1.0\data\clinical.csv"
morpho_data_patient = data_process_ml.read_and_combine_data(morpho_path,patient_path)
merged_dataset = data_process_ml.encode_column(morpho_data_patient)
merged_dataset = data_process_ml.drop_columns(merged_dataset)
morpho_data_cut1,morpho_data_dome = data_process_ml.output_cut1anddome(merged_dataset)
#load Aneux Dataset
Aneux_1000pt_Dataset_uniform = Aneux_Dataset_load('./Datasets/Aneux_Dataset_1000pt_sample_600train.pt')
Aneux_1000pt_Dataset_uniform_test = Aneux_Dataset_load('./Datasets/Aneux_Dataset_1000pt_sample_100test.pt')

Aneux_2000pt_Dataset_uniform = Aneux_Dataset_load('./Datasets/Aneux_Dataset_2000pt_sample_600train.pt')
Aneux_2000pt_Dataset_uniform_test = Aneux_Dataset_load('./Datasets/Aneux_Dataset_2000pt_sample_100test.pt')

Aneux_1000pt_Dataset_ppd = Aneux_Dataset_load('./Datasets/Aneux_Dataset_1000pt_ppd_600train.pt')
Aneux_1000pt_Dataset_ppd_test = Aneux_Dataset_load('./Datasets/Aneux_Dataset_1000pt_ppd_100test.pt')

Aneux_2000pt_Dataset_ppd = Aneux_Dataset_load('./Datasets/Aneux_Dataset_2000pt_ppd_600train.pt')
Aneux_2000pt_Dataset_ppd_test = Aneux_Dataset_load('./Datasets/Aneux_Dataset_2000pt_ppd_100test.pt')


#seed use for training models



def output_result(points = 1000, sample = "uniform", model_type = "pointnet", cuttype = "dome",model_file = ""):

    

    Dataset = Aneux_1000pt_Dataset_uniform
    Dataset_test = Aneux_1000pt_Dataset_uniform_test
    
    if model_file == "":
        print("No such model")
        return
    elif not os.path.exists(model_file):
        print("No such model path")
        return
    else:
        pass
        
    if points == 1000 and sample == "uniform":
        Dataset = Aneux_1000pt_Dataset_uniform
        Dataset_test = Aneux_1000pt_Dataset_uniform_test
    elif points == 2000 and sample == "uniform":
        Dataset = Aneux_2000pt_Dataset_uniform
        Dataset_test = Aneux_2000pt_Dataset_uniform_test
    elif points == 1000 and sample == "ppd":
        Dataset = Aneux_1000pt_Dataset_ppd
        Dataset_test = Aneux_1000pt_Dataset_ppd_test
    elif points == 2000 and sample == "ppd":
        Dataset = Aneux_2000pt_Dataset_ppd
        Dataset_test = Aneux_2000pt_Dataset_ppd_test
    else:
        print("only 1000/2000 point and pdd/uniform sample available")
        return
    
    if cuttype == "dome":
        Dataset.cuttype = "dome"
        Dataset_test.cuttype = "dome"
    else:
        Dataset.cuttype = "cut1"
        Dataset_test.cuttype = "cut1"
        
    
        
    torch.manual_seed(32381912834800)
    #load training test and validation set
    train_size = int(len(Dataset) * 0.8) # 80% training data
    valid_size = len(Dataset) - train_size
    train_data, valid_data = random_split(Dataset, [train_size, valid_size])

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

    test_loader = torch.utils.data.DataLoader(
        Dataset_test,
        batch_size=50, # Forward pass only so batch size can be larger
        shuffle=False,
        #num_workers=2, 
        pin_memory=True
    )
    if model_type == "pointnet":
        AOCname = str(points) + "pt pointnet AOC" + "("+ sample +" sampling)"
        CMname = str(points) + "pt pointnet CM" + "("+ sample +" sampling)"
        print("\n Internal test result")
        y_true,y_pred,y_pred_result,y_probs = pointnet_testcase(model_file,valid_loader)
        draw_rocgraph(y_pred_result,y_true, y_probs, name=AOCname)
        show_cm_dl(y_true, y_pred,y_pred_result,name=CMname)
        print_values (y_true, y_pred_result)
        print("\n External test result")
        y_true,y_pred,y_pred_result,y_probs = pointnet_testcase(model_file,test_loader)
        draw_rocgraph(y_pred_result,y_true, y_probs, name=AOCname)
        show_cm_dl(y_true, y_pred,y_pred_result,name=CMname)
        print_values (y_true, y_pred_result)
        
    elif model_type == "pointnet_2branch":
        AOCname = str(points) + "pt pointnet AOC" + "("+ sample +" sampling)"
        CMname = str(points) + "pt pointnet CM" + "("+ sample +" sampling)"
        print("\n Internal test result")
        y_true,y_pred,y_pred_result,y_probs = pointnet_2branch_testcase(model_file,valid_loader)
        draw_rocgraph(y_pred_result,y_true, y_probs, name=AOCname)
        show_cm_dl(y_true, y_pred,y_pred_result,name=CMname)
        print_values (y_true, y_pred_result)
        print("\n External test result")
        y_true,y_pred,y_pred_result,y_probs = pointnet_2branch_testcase(model_file,test_loader)
        draw_rocgraph(y_pred_result,y_true, y_probs, name=AOCname)
        show_cm_dl(y_true, y_pred,y_pred_result,name=CMname)
        print_values (y_true, y_pred_result)
        
    elif model_type == "pointnet_3branch": 
        AOCname = str(points) + "pt pointnet 3branch AOC" + "("+ sample +" sampling)"
        CMname = str(points) + "pt pointnet 3branch CM" + "("+ sample +" sampling)"
        print("\n Internal test result")
        y_true,y_pred,y_pred_result,y_probs = pointnet_3_branch_testcase(model_file,valid_loader)
        draw_rocgraph(y_pred_result,y_true, y_probs, name=AOCname)
        show_cm_dl(y_true, y_pred,y_pred_result,name=CMname)
        print_values (y_true, y_pred_result)
        print("\n External test result")
        y_true,y_pred,y_pred_result,y_probs = pointnet_3_branch_testcase(model_file,test_loader)
        draw_rocgraph(y_pred_result,y_true, y_probs, name=AOCname)
        show_cm_dl(y_true, y_pred,y_pred_result,name=CMname)
        print_values (y_true, y_pred_result)
    else:
        print("No such model type")
        
    return