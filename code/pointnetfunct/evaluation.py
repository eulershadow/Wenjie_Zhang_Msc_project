
from sklearn.metrics import roc_curve, auc,classification_report
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import seaborn as sns
from sklearn.metrics import f1_score,accuracy_score, recall_score, precision_score,mean_squared_error
from . import functions as fun
from sklearn.metrics import roc_curve, roc_auc_score


def get_pred_3branch(model, valid_loader):
    y_true = []
    y_pred = []
    y_probs = []
    y_pred_result = []
    model = model.to("cpu")
    with torch.no_grad():
        for inputs_v,inputs_v2,inputs_v3,labels in valid_loader:
            inputs_v = inputs_v.to(torch.float32)
            inputs_v = inputs_v.squeeze(1).permute(0, 2, 1)
            inputs_v2 = inputs_v2.to(torch.float32)
            inputs_v2 = inputs_v2.squeeze(1).permute(0, 2, 1)
            inputs_v3 = inputs_v3.to(torch.float32)

            labels = labels.to(torch.long)
            outputs = model.forward(inputs_v,inputs_v2,inputs_v3)
            
            #print(torch.exp(outputs))

            outputs_value = (torch.max(torch.exp(outputs), 1)[0]).data.cpu().numpy()
            outputs_result = (torch.max(torch.exp(outputs), 1)[1]).data.cpu().numpy()
            
            
            #print(outputs)
            y_pred.extend(outputs_value)
            y_pred_result.extend(outputs_result)
                
            labels = labels.data.cpu().numpy()
            y_true.extend(labels) 
            y_probs.extend(outputs)
    return y_true, y_pred, y_pred_result, y_probs

def get_pred_2branch(model, valid_loader):
    y_true = []
    y_pred = []
    y_probs = []
    y_pred_result = []
    model = model.to("cpu")
    with torch.no_grad():
        for inputs_v,inputs_v2,inputs_v3,labels in valid_loader:
            inputs_v = inputs_v.to(torch.float32)
            inputs_v = inputs_v.squeeze(1).permute(0, 2, 1)
            inputs_v2 = inputs_v2.to(torch.float32)
            inputs_v2 = inputs_v2.squeeze(1).permute(0, 2, 1)

            labels = labels.to(torch.long)
            outputs = model.forward(inputs_v,inputs_v2,inputs_v3)
            
            #print(torch.exp(outputs))

            outputs_value = (torch.max(torch.exp(outputs), 1)[0]).data.cpu().numpy()
            outputs_result = (torch.max(torch.exp(outputs), 1)[1]).data.cpu().numpy()
            
            
            #print(outputs)
            y_pred.extend(outputs_value)
            y_pred_result.extend(outputs_result)
                
            labels = labels.data.cpu().numpy()
            y_true.extend(labels) 
            y_probs.extend(outputs)
    return y_true, y_pred, y_pred_result, y_probs

def get_pred_pointnet(model, valid_loader):
    y_true = []
    y_pred = []
    y_probs = []
    y_pred_result = []
    model = model.to("cpu")
    with torch.no_grad():
        for inputs_v,inputs_v2,inputs_v3,labels in valid_loader:
            inputs_v = inputs_v.to(torch.float32)
            inputs_v = inputs_v.squeeze(1).permute(0, 2, 1)
            inputs_v2 = inputs_v2.to(torch.float32)
            inputs_v2 = inputs_v2.squeeze(1).permute(0, 2, 1)
            inputs_v3 = inputs_v3.to(torch.float32)

            labels = labels.to(torch.long)
            outputs = model.forward(inputs_v)
            
            #print(torch.exp(outputs))

            outputs_value = (torch.max(torch.exp(outputs), 1)[0]).data.cpu().numpy()
            outputs_result = (torch.max(torch.exp(outputs), 1)[1]).data.cpu().numpy()
            
            
            #print(outputs)
            y_pred.extend(outputs_value)
            y_pred_result.extend(outputs_result)
                
            labels = labels.data.cpu().numpy()
            y_true.extend(labels) 
            y_probs.extend(outputs)
    return y_true, y_pred, y_pred_result, y_probs


def draw_rocgraph( y_pred,y_true, probs_class, name,interval = 0.02):
    classes = ["Unrupture","Rupture"]
    all_probs = probs_class

    top_classes = []
    top_classes_index = []

    #get top 2 and bottom 2
    class_df = pd.DataFrame(classification_report(y_true, y_pred, target_names=classes, digits=4, output_dict=True))
    class_df = class_df.drop(["accuracy","macro avg","weighted avg"], axis=1).transpose()
    top_classes = list(class_df.sort_values('precision',ascending = False).head(2).index)
    for class_idx in top_classes:
        top_classes_index.append(classes.index(class_idx))
        
    fpr, tpr, roc_auc = {}, {}, {}

    for class_idx in top_classes_index:
        y_true_class = (np.array(y_true) == class_idx).astype(int)
        probs_class = np.array(all_probs)[:, class_idx]
        fpr[class_idx], tpr[class_idx], _ = roc_curve(y_true_class, probs_class)

        roc_auc[class_idx] = auc(fpr[class_idx], tpr[class_idx])

    # Plot ROC curve
    plt.figure(figsize=(10, 6))
    full_list = top_classes
    full_list_index = top_classes_index
    for class_idx in range(len(full_list)):
        class_index = full_list_index[class_idx]
        plt.plot(fpr[class_index], tpr[class_index], label='Class ' +full_list[class_idx] + '(AUC = ' + f"{roc_auc[class_index]:.2f}" + '±'+str(interval) +')' )

    plt.title(name)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.show()
    
def show_graph(path,device,name = None):
    data = torch.load(path, map_location=device) # by doing map_location=device, you can use trained model on GPU --> to test on CPU
    statsrec = data["stats"]
    fig, ax1 = plt.subplots()
    plt.plot(statsrec[0], 'r', label = 'training loss', )
    plt.plot(statsrec[2], 'g', label = 'test loss' )
    plt.legend(loc='lower right')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    if name == None:
        plt.title('Training and test loss, and test accuracy')
    else:
        plt.title(str(name))
    ax2=ax1.twinx()
    ax2.plot(statsrec[1], 'm', label = 'training accuracy')
    ax2.plot(statsrec[3], 'b', label = 'test accuracy')
    ax2.set_ylabel('accuracy')
    plt.legend(loc='upper right')
    fig.savefig("roc.svg")

    plt.show()
    
def dim4_cm (real, pred,pred_result):

    cm = [[0,0,0,0],[0,0,0,0]]
    for i in range(len(pred)):
        pos = 0
        if float(pred[i]) < 0.75 and pred_result[i] == 0:
            pos = 1
        elif float(pred[i]) >= 0.75 and pred_result[i] == 0:
            pos = 0
        if float(pred[i]) < 0.75 and pred_result[i] == 1:
            pos = 2
        elif float(pred[i]) >= 0.75 and pred_result[i] == 1:
            pos = 3
            
            
        if real[i] == 0:

            cm[0][pos] += 1
        else:
            cm[1][pos] += 1
    return cm

def show_cm_dl (y_true, y_pred,y_pred_result,name):
    cm = dim4_cm(y_true, y_pred,y_pred_result)  
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Pred Unrupture","Pred uncertain Unrupture","Pred uncertain Rupture","Pred Rupture"], yticklabels=["Real Unrupture"," Real Rupture"])
    #fun.show_pred_cm(cm,y_true)  
    # classes = ["Unrupture","Rupture"]    
    # cf_matrix = confusion_matrix(y_true, y_pred)
    # plt.figure(figsize=(10, 8)) 
    # sns.heatmap(cf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes,annot_kws={"size": 9})
    # plt.xlabel("Predicted Labels")
    # plt.ylabel("True Labels")
    plt.title(str(name))
    plt.show()
    
def show_cm_ml(data_predictions, test_set_target):
    #cm = confusion_matrix(list(test_set_target), data_predictions)
    cm = fun.dim4_cm(list(test_set_target), data_predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Pred Unrupture","Pred uncertain Unrupture","Pred uncertain Rupture","Pred Rupture"], yticklabels=["Real Unrupture"," Real Rupture"])
    
def print_values (y_true, y_pred):
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, pos_label=1)
    recall = recall_score(y_true, y_pred, pos_label=1)
    precision0 = precision_score(y_true, y_pred, pos_label=0)
    recall0 = recall_score(y_true, y_pred, pos_label=0)
    
    accuracy = accuracy_score(y_true, y_pred)
    print(f'Accuracy: {accuracy}')
    print(f'F1 score: {f1}')
    
    print(f'Rupture Recall: {recall}')
    print(f'Rupture Precision: {precision}')
    print(f'UnRupture Recall: {recall0}')   
    print(f'UnRupture Precision: {precision0}')
    
def draw_ROC_ml (y_true_binary, y_pred_binary,name):
    fpr, tpr, thresholds = roc_curve(y_true_binary, y_pred_binary)
    auc = roc_auc_score(y_true_binary, y_pred_binary)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC for ' + name )
    plt.legend(loc="lower right")
    plt.show()

def print_values_advance (y_true, y_pred, name):
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, pos_label=1)
    recall = recall_score(y_true, y_pred, pos_label=1)
    precision0 = precision_score(y_true, y_pred, pos_label=0)
    recall0 = recall_score(y_true, y_pred, pos_label=0)
    data_mse=mean_squared_error(y_true = y_true, y_pred = y_pred)
    data_rmse = np.sqrt(data_mse)
    
    accuracy = accuracy_score(y_true, y_pred)
    print(name + f' Accuracy: {accuracy}')
    print(name + f' F1 score: {f1}')
    print( name + f' RMSE: {data_rmse}')
    
    print( name + f' Rupture Recall: {recall}')
    print( name + f' Rupture Precision: {precision}')
    print( name + f' UnRupture Recall: {recall0}')   
    print( name + f' UnRupture Precision: {precision0}')