
import vtk
import os
import numpy as np
import itertools
import math, random
import data_process_ml
random.seed = 42
import copy
import sys
sys.path.append("..")

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
from torchvision import transforms, utils

import scipy.spatial.distance
# import plotly.graph_objects as go
# import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from .model_loader import reduce_mesh,mesh_to_point_cloud,vtp_to_mesh,vtp_to_point_cloud,vtp_to_point_cloud_cutvessel
import open3d as o3d
import pandas as pd

class Aneuxmodel_Dataset(Dataset):

    def __init__(self, df = pd.DataFrame(), root = "", transform = None,mesh = "area-001",cuttype = "dome",crop = False,points= 1000, limit = 700,load = True,paraout = False):

        self.root = root
        self.points = points
        self.limit = limit
        self.transform = transform
        self.mesh = mesh
        self.cuttype = cuttype
        self.crop = crop
        self.paraout = paraout
        if self.cuttype!= "dome" and self.cuttype!= "cut1":
            return "type error"
        
        self.df = df[df["cuttype"] == cuttype]
        self.label = []
        self.vessel_model_file = []
        self.cut1_model_file = []
        self.dome_model_file = []
        self.model_table = []
        
        self.cropdome_vessel_file = []
        self.cropcut1_vessel_file = []
        self.my_device = "cuda:0"
        
        merged_dataset = data_process_ml.encode_column(df)
        merged_dataset = data_process_ml.drop_columns(merged_dataset)
        morpho_data_cut1,morpho_data_dome = data_process_ml.output_cut1anddome(merged_dataset)
        
        self.raw_data = copy.deepcopy(morpho_data_dome)
        self.raw_data = self.raw_data.astype(float)
        
        self.raw_data.loc[self.raw_data['sex_male'] == 0, 'sex_male'] = 2
        #print(self.raw_data.iloc[0])
        self.raw_data.drop(("status_ruptured"),axis=1,inplace=True)
        self.raw_data.drop(("sex_female"),axis=1,inplace=True)
        self.raw_data.drop(("age"),axis=1,inplace=True)
        
        
        if load:
            self.training_data_load()
            self.label_loader()
            
        
            
    def label_loader(self):
        self.label = []
        for model in self.model_table:
            if  model in list(self.df["dataset"]):
                label_num = self.df[self.df["dataset"] == model]["status"]
                label_num = list(label_num)[0]
                if label_num == "ruptured":
                    label_num = 1
                elif label_num == "unruptured":
                    label_num = 0
                else:
                    print(label_num)
                self.label.append(label_num)
        self.label = torch.from_numpy(np.array(self.label))
        return True
    
    def find_dome_cut1(self,all_IA_model,model_name):
        IA_name_cut1 = ""
        IA_name_dome = ""
        for IA_model in all_IA_model:
                
            if model_name in IA_model and "cut1" in IA_model and model_name not in IA_name_cut1:
                #print("find:" + IA_model)
                IA_name_cut1 = IA_model
            elif model_name.split("_")[0] in IA_model and "cut1" in IA_model and IA_name_cut1 == "":
                IA_name_cut1 = IA_model
                
            if model_name in IA_model and "dome" in IA_model and model_name not in IA_name_dome:
                IA_name_dome = IA_model
            elif model_name.split("_")[0] in IA_model and "dome" in IA_model and IA_name_dome == "":
                IA_name_dome = IA_model
                
            if IA_name_cut1 != "" and IA_name_dome != "" and model_name in IA_name_cut1 and model_name in IA_name_dome:
                break
        return IA_name_cut1,IA_name_dome
    
    def find_vessel(self,all_vessel_model,model_name):
        vessel_name = ""
        for vessel_model in all_vessel_model:
            vessel_model_process =  vessel_model[:-4]  
            if (model_name in vessel_model or vessel_model_process in model_name ) and ( model_name not in vessel_name):
                #print("find:" + IA_model)
                vessel_name = vessel_model
            elif (model_name.split("_")[0] in vessel_model or vessel_model_process in model_name.split("_")[0] ) and vessel_name == "":
                vessel_name = vessel_model
                
                
            if vessel_name != ""  and (model_name in vessel_model or vessel_model_process in model_name):
                break
        return vessel_name
                
    def training_data_load(self):
        self.label = []
        self.vessel_model_file = []
        self.cut1_model_file = []
        self.dome_model_file = []
        self.model_table = []
        self.cropdome_vessel_file = []
        self.cropcut1_vessel_file = []
        
        #path of the files
        IA = "aneurysms\\remeshed\\area-001"
        Vessel = "vessels\\remeshed\\area-001"
        if self.mesh == "area-001":
            IA = "aneurysms\\remeshed\\area-001"
            Vessel = "vessels\\remeshed\\area-001"
        elif self.mesh == "area-005":
            IA = "aneurysms\r\emeshed\\area-005"
            Vessel = "vessels\\remeshed\\area-005"
        elif self.mesh == "orginal":
            IA = "aneurysms\\orginal"
            Vessel = "vessels\\orginal"
            
        IA_root = os.path.join(self.root,IA)
        Vessel_root = os.path.join(self.root,Vessel)
        #list of the model files
        all_vessel_model = os.listdir(Vessel_root)
        all_IA_model = os.listdir(IA_root)
        all_model_name = list(self.df["dataset"])

        for model in all_model_name[self.limit:]:
            #file name for IA cut1 and dome
            model_name = model
            # get cut1 file name and dome file name from ALL_IA_model list
            IA_name_cut1 = ""
            IA_name_dome = ""
            IA_name_cut1,IA_name_dome = self.find_dome_cut1(all_IA_model,model_name)

            #read the file path and add the model to the list
            if IA_name_cut1 in all_IA_model:
                IA_root_cut1 = os.path.join(IA_root,IA_name_cut1)
                cut1 = vtp_to_point_cloud(IA_root_cut1,points = self.points)
                self.cut1_model_file.append(cut1)
            else:
                print("missing a cut1 model: " + model_name)
                
            if IA_name_dome in all_IA_model:
                IA_root_dome = os.path.join(IA_root,IA_name_dome)
                dome = vtp_to_point_cloud(IA_root_dome,points = self.points)
                self.dome_model_file.append(dome)
            else:
                print("missing a dome model: " + model_name)
            
            
            # append vessel list  
            vessel_name = ""
            vessel_name = self.find_vessel(all_vessel_model,model_name)         
            if vessel_name in all_vessel_model:
                Vessel_model_root = os.path.join(Vessel_root,vessel_name)
                vessel = vtp_to_point_cloud(Vessel_model_root,points = self.points)
                self.vessel_model_file.append(vessel)
            else:
                print("missing a vessel model: " + model_name)
                
            
            if self.crop:
                IA_root_cut1 = os.path.join(IA_root,IA_name_cut1)
                IA_root_dome = os.path.join(IA_root,IA_name_dome)
                Vessel_model_root = os.path.join(Vessel_root,vessel_name)
                Vessel_crop_cut1 = vtp_to_point_cloud_cutvessel(Vessel_model_root,IA_root_cut1,points = self.points)
                Vessel_crop_dome = vtp_to_point_cloud_cutvessel(Vessel_model_root,IA_root_dome,points = self.points)
                if Vessel_crop_dome == False:
                    print(Vessel_model_root)
                self.cropcut1_vessel_file.append(Vessel_crop_cut1)
                self.cropdome_vessel_file.append(Vessel_crop_dome)
                        
            self.model_table.append(model_name)
        
        return True
        # self.org_imgs = np.array(self.org_imgs)
        # self.total_imgs = torch.from_numpy(np.array(self.total_imgs)) 
        # self.label = torch.from_numpy(np.array(self.label))  
        # return True

    
    def __getitem__(self, index):
        
        """ Returns one data pair (image and target caption). """
        cut_model = self.cut1_model_file[index]
        vessel_model = self.vessel_model_file[index]
        vessel_model_crop = vessel_model
        if self.cuttype == "dome":
            cut_model = self.dome_model_file[index]
        if self.crop and self.cuttype == "cut1":
            vessel_model_crop = self.cropcut1_vessel_file[index]
        if self.crop and self.cuttype == "dome":
            vessel_model_crop = self.cropdome_vessel_file[index]
            
        vessel_model = np.asarray(vessel_model.points)
        cut_model = np.asarray(cut_model.points)
        vessel_model_crop = np.asarray(vessel_model_crop.points)
        
        if self.transform is not None:      
            cut_model= self.transform(cut_model)
            vessel_model = self.transform(vessel_model)
            if self.crop:
                vessel_model_crop = self.transform(vessel_model_crop)
            
        label_return = self.label[index]
        
        if self.paraout:
            parameter = torch.from_numpy(np.array(self.raw_data.iloc[index],dtype = np.float64))
            return vessel_model,cut_model,parameter, label_return
        if self.crop: 
            return vessel_model,cut_model,vessel_model_crop, label_return
        return vessel_model,cut_model,label_return

    def __len__(self):
                                
        return len(self.model_table)
    
    
def Aneux_Dataset_save(dataset, filepath):
        data = {
            'df': dataset.df,
            #'raw_data': dataset.raw_data,
            'root': dataset.root,
            'transform': dataset.transform,
            'mesh': dataset.mesh,
            'cuttype': dataset.cuttype,
            'crop': dataset.crop,
            'points': dataset.points,
            'label': dataset.label.numpy(),
            'limit': dataset.limit,
            'vessel_model_file': [np.asarray(pcd.points) for pcd in dataset.vessel_model_file],
            'cut1_model_file': [np.asarray(pcd.points) for pcd in dataset.cut1_model_file],
            'dome_model_file': [np.asarray(pcd.points) for pcd in dataset.dome_model_file],
            'model_table': dataset.model_table,
            'cropdome_vessel_file': [np.asarray(pcd.points) for pcd in dataset.cropdome_vessel_file],
            'cropcut1_vessel_file': [np.asarray(pcd.points) for pcd in dataset.cropcut1_vessel_file],
        }
        torch.save(data, filepath)
    
def Aneux_Dataset_load(filepath):
    data = torch.load(filepath)
    dataset = Aneuxmodel_Dataset(
        df=data['df'],
        
        root=data['root'],
        transform=data['transform'],
        mesh=data['mesh'],
        cuttype=data['cuttype'],
        crop=data['crop'],
        points=data['points'],
        limit = data['limit'],
        load = False
    )
    dataset.label = torch.from_numpy(data['label'])
    #dataset.raw_data = data['raw_data']
    dataset.vessel_model_file = [o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts)) for pts in data['vessel_model_file']]
    dataset.cut1_model_file = [o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts)) for pts in data['cut1_model_file']]
    dataset.dome_model_file = [o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts)) for pts in data['dome_model_file']]
    dataset.model_table = data['model_table']
    dataset.cropdome_vessel_file = [o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts)) for pts in data['cropdome_vessel_file']]
    dataset.cropcut1_vessel_file = [o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts)) for pts in data['cropcut1_vessel_file']]
        
    return dataset