from .PointNet_struct import PointNet,PointNet_2Branch,PointNet_3Branch
from .PointNet_trainingfunct import load_model
import torch
   
def pointnet_testcase(model_path,loader):
    pointnet_copy = PointNet(classes=2)

    load_model(pointnet_copy,model_path)    
    y_true = []
    y_pred = []
    y_pred_result = []
    y_probs = []
    pointnet_copy = pointnet_copy.to("cpu")
    with torch.no_grad():
        for inputs_v,_,_,labels in loader:
            inputs_v = inputs_v.to(torch.float32)
            inputs_v = inputs_v.squeeze(1).permute(0, 2, 1)
            labels = labels.to(torch.long)
            outputs = pointnet_copy.forward(inputs_v)
            
            #print(torch.exp(outputs))

            outputs_value = (torch.max(torch.exp(outputs), 1)[0]).data.cpu().numpy()
            outputs_result = (torch.max(torch.exp(outputs), 1)[1]).data.cpu().numpy()
            
            
            #print(outputs)
            y_pred.extend(outputs_value)
            y_pred_result.extend(outputs_result)
                
            labels = labels.data.cpu().numpy()
            y_true.extend(labels)
            y_probs.extend(outputs)
            
    return y_true,y_pred,y_pred_result,y_probs

def pointnet_2branch_testcase(model_path,loader):
    pointnet_copy = PointNet_2Branch(classes=2)
    load_model(pointnet_copy,model_path)
    y_true = []
    y_pred = []
    y_probs = []
    y_pred_result = []
    pointnet_copy = pointnet_copy.to("cpu")
    with torch.no_grad():
        for inputs_v,inputs_v2,inputs_v3,labels in loader:
            inputs_v = inputs_v.to(torch.float32)
            inputs_v = inputs_v.squeeze(1).permute(0, 2, 1)
            inputs_v3 = inputs_v3.to(torch.float32)

            labels = labels.to(torch.long)
            outputs = pointnet_copy.forward(inputs_v,inputs_v3)
            
            #print(torch.exp(outputs))

            outputs_value = (torch.max(torch.exp(outputs), 1)[0]).data.cpu().numpy()
            outputs_result = (torch.max(torch.exp(outputs), 1)[1]).data.cpu().numpy()
            
            
            #print(outputs)
            y_pred.extend(outputs_value)
            y_pred_result.extend(outputs_result)
                
            labels = labels.data.cpu().numpy()
            y_true.extend(labels) 
            y_probs.extend(outputs)
            
    return y_true,y_pred,y_pred_result,y_probs

def pointnet_3_branch_testcase(model_path,loader):
    pointnet_copy = PointNet_3Branch(classes=2)
    load_model(pointnet_copy,model_path)
    y_true = []
    y_pred = []
    y_probs = []
    y_pred_result = []
    pointnet_copy = pointnet_copy.to("cpu")
    with torch.no_grad():
        for inputs_v,inputs_v2,inputs_v3,labels in loader:
            inputs_v = inputs_v.to(torch.float32)
            inputs_v = inputs_v.squeeze(1).permute(0, 2, 1)
            inputs_v2 = inputs_v2.to(torch.float32)
            inputs_v2 = inputs_v2.squeeze(1).permute(0, 2, 1)
            inputs_v3 = inputs_v3.to(torch.float32)

            labels = labels.to(torch.long)
            outputs = pointnet_copy.forward(inputs_v,inputs_v2,inputs_v3)
            
            #print(torch.exp(outputs))

            outputs_value = (torch.max(torch.exp(outputs), 1)[0]).data.cpu().numpy()
            outputs_result = (torch.max(torch.exp(outputs), 1)[1]).data.cpu().numpy()
            
            
            #print(outputs)
            y_pred.extend(outputs_value)
            y_pred_result.extend(outputs_result)
                
            labels = labels.data.cpu().numpy()
            y_true.extend(labels) 
            y_probs.extend(outputs)

            
    return y_true,y_pred,y_pred_result,y_probs

def DNN_testcase(model,loader):
    y_true = []
    y_pred = []
    y_pred_result = []
    y_probs = []
    pointnet_copy = model.to("cpu")
    with torch.no_grad():
        for inputs_v,labels in loader:
            inputs_v = inputs_v.to(torch.float32)
            labels = labels.to(torch.long)
            outputs = pointnet_copy.forward(inputs_v)
            
            #print(torch.exp(outputs))

            outputs_value = (torch.max(torch.exp(outputs), 1)[0]).data.cpu().numpy()
            outputs_result = (torch.max(torch.exp(outputs), 1)[1]).data.cpu().numpy()
            
            
            #print(outputs)
            y_pred.extend(outputs_value)
            y_pred_result.extend(outputs_result)
                
            labels = labels.data.cpu().numpy()
            y_true.extend(labels) 
            y_probs.extend(outputs)
    return y_true,y_pred,y_pred_result,y_probs