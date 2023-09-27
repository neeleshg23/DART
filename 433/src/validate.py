import csv
import json
import os
import pprint
import yaml
import numpy as np
from numpy import nanargmax, sqrt
import pandas as pd
import torch
from torch import nn
from tqdm import tqdm
from sklearn.metrics import auc, f1_score, precision_score, recall_score, precision_recall_curve, roc_curve

from utils import select_model, replace_directory 
from threshold import find_optimal_threshold

model = None
device = None
BITMAP_SIZE = None
sigmoid = nn.Sigmoid()

def to_bitmap(n,bitmap_size):
    #l0=np.ones((bitmap_size),dtype = int)
    l0=np.zeros((bitmap_size),dtype = int)
    if(len(n)>0):
        for x in n:
            if x>0:
                l0[int(x)-1]=1
            elif x<0:
                l0[int(x)]=1
        l1=list(l0)
        return l1
    else:
        return list(l0)

def threshold_throttleing(test_df,throttle_type="f1",optimal_type="micro",topk=2,threshold=0.5):
    y_score=np.stack(test_df["y_score"])
    y_real=np.stack(test_df["future"])
    best_threshold=0
    if throttle_type=="roc":
        print("throttleing by roc curve")
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        threshold=dict()
        best_threshold_list=[]
        gmeans=dict()
        ix=dict()
        #pdb.set_trace()
        for i in range(BITMAP_SIZE):
            fpr[i], tpr[i], threshold[i] =roc_curve(y_real[:,i],y_score[:,i])
            roc_auc[i] = auc(fpr[i],tpr[i])
            #best:
            gmeans[i] = sqrt(tpr[i]*(1-fpr[i]))
            ix[i]=nanargmax(gmeans[i])
            best_threshold_list.append(threshold[i][ix[i]])
            #print('Dimension: i=%d, Best threshold=%f, G-Mean=%.3f' %(i, threshold[i][ix[i]], gmeans[i][ix[i]]))
        if optimal_type=="indiv":
            best_threshold=best_threshold_list
            y_pred_bin = (y_score-np.array(best_threshold) >0)*1
            test_df["predicted"]= list(y_pred_bin)#(all,[length])
        elif optimal_type=="micro":
            fpr["micro"], tpr["micro"], threshold["micro"] = roc_curve(y_real.ravel(), y_score.ravel())
            roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
            #best:
            gmeans["micro"] = sqrt(tpr["micro"]*(1-fpr["micro"]))
            ix["micro"]=nanargmax(gmeans["micro"])
            best_threshold=threshold["micro"][ix["micro"]]
            print('Best micro threshold=%f, G-Mean=%.3f' %(best_threshold, gmeans["micro"][ix["micro"]]))
            
            y_pred_bin = (y_score-best_threshold >0)*1
            test_df["predicted"]= list(y_pred_bin)#(all,[length])
            
    if throttle_type=="f1":
        # print("throttleing by precision-recall curve")
        p = dict()
        r = dict()
        threshold=dict()
        best_threshold_list=[]
        fscore=dict()
        ix=dict()
        
        p["micro"], r["micro"], threshold["micro"]=precision_recall_curve(y_real.ravel(),y_score.ravel())
        fscore["micro"] = (2 * p["micro"] * r["micro"]) / (p["micro"] + r["micro"])
        ix["micro"]=nanargmax(fscore["micro"])
        best_threshold=threshold["micro"][ix["micro"]]
        print('Best micro threshold=%f, fscore=%.3f' %(best_threshold, fscore["micro"][ix["micro"]]))
        y_pred_bin = (y_score-best_threshold >0)*1
        test_df["predicted"]= list(y_pred_bin)
        
    elif throttle_type=="topk":
        print("throttleing by topk:",topk)
        pred_index = torch.tensor(y_score).topk(topk)[1].cpu().detach().numpy()
        y_pred_bin=[to_bitmap(a,BITMAP_SIZE) for a in pred_index]
        test_df["predicted"]= list(y_pred_bin)
        
    elif throttle_type =="fixed_threshold":
        print("throttleing by fixed threshold:",threshold)
        best_threshold=threshold
        y_pred_bin = (y_score-np.array(best_threshold) >0)*1
        test_df["predicted"]= list(y_pred_bin)#(all,[length])
    
    return test_df, best_threshold

def model_prediction(test_loader, test_df, model_save_path):#"top_k";"degree";"optimal"
    # print("predicting")
    prediction=[]
    model.load_state_dict(torch.load(model_save_path))
    model.to(device)
    model.eval()
    y_score=np.array([])
    for data, _ in tqdm(test_loader):
        output = sigmoid(model(data))
        #prediction.extend(output.cpu())
        prediction.extend(output.cpu().detach().numpy())
    test_df["y_score"]= prediction

    return test_df[['id', 'cycle', 'addr', 'ip', 'block_address', 'future', 'y_score']]

def evaluate(y_test,y_pred_bin):
    f1_score_res=f1_score(y_test, y_pred_bin, average='micro')
    #recall: tp / (tp + fn)
    recall_score_res=recall_score(y_test, y_pred_bin, average='micro')
    #precision: tp / (tp + fp)
    precision_score_res=precision_score(y_test, y_pred_bin, average='micro',zero_division=0)
    print("p,r,f1:",precision_score_res,recall_score_res,f1_score_res)
    return precision_score_res,recall_score_res,f1_score_res

def run_val(test_loader, df_test, app_name, model_save_path, res_path, option, gpu_id):
    global model
    global device
    
    with open("params.yaml", "r") as p:
        params = yaml.safe_load(p)
    
    n_samples = params["threshold"]["samples"]
    
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    model = select_model(option)

    res = {}

    print("Validation start")
    
    #sampled_threshold = find_optimal_threshold(model, device, train_loader, model_save_path, n_samples)
    test_df = model_prediction(test_loader, df_test, model_save_path)

    # df_res, threshold = threshold_throttleing(test_df, throttle_type="fixed_threshold", threshold=sampled_threshold)
    # sample_p, sample_r, sample_f1 = evaluate(np.stack(df_res["future"]), np.stack(df_res["predicted"]))
    
    df_res, threshold = threshold_throttleing(test_df, throttle_type="f1", optimal_type="micro")
    microf1_p, microf1_r, microf1_f1 = evaluate(np.stack(df_res["future"]), np.stack(df_res["predicted"]))
    
    df_res, _ = threshold_throttleing(test_df, throttle_type="fixed_threshold", threshold=0.5)
    half_p, half_r, half_f1 = evaluate(np.stack(df_res["future"]), np.stack(df_res["predicted"]))
    
    report = {
        "model": option,
        "app": app_name,
        "validation": [ 
            # {
            #     "method":"sampled train data",
            #     "threshold": float(sampled_threshold),
            #     "p": float(sample_p),
            #     "r": float(sample_r),
            #     "f1": float(sample_f1)
            # },
            {
                "method":"micro precision f1",
                "threshold": float(threshold),
                "p": float(microf1_p),
                "r": float(microf1_r),
                "f1": float(microf1_f1)
            },
            {
                "method":"fixed threshold 0.5",
                "threshold": 0.5,
                "p": float(half_p),
                "r": float(half_r),
                "f1": float(half_f1) 
            }
        ]
    }

    pprint.pprint(report,sort_dicts=False)
    with open(res_path+'.val_res.json', 'w') as json_file:
        json.dump(report, json_file,indent=2)
