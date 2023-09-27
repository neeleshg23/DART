#%% y_train;y_score
import os
import pickle
import warnings
warnings.filterwarnings('ignore')
import sys
import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import numpy as np
import pandas as pd
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
from prep_single import read_load_trace_data, preprocessing, to_bitmap,preprocessing_gen
from torch.autograd import Variable
from sklearn.metrics import roc_curve,f1_score,recall_score,precision_score,accuracy_score
import lzma
from tqdm import tqdm
import os
import pdb
from torch.utils.data import Dataset,DataLoader
from sklearn.metrics import roc_curve, auc
from numpy import sqrt, argmax
import json
import yaml

from validate import threshold_throttleing
from data_loader import init_dataloader, MAPDataset_gen
from utils import select_model, replace_directory

with open("params.yaml", "r") as p:
    params = yaml.safe_load(p) 

trace_dir = params["system"]["traces"]
model_dir = params["system"]["model"]
res_dir = params["system"]["res"]
processed_dir = params["system"]["processed"]

TRAIN = params["trace-data"]["train"]
TOTAL = params["trace-data"]["total"]
SKIP = params["trace-data"]["skip"]
batch_size = params["trace-data"]["batch-size"]

epochs = params["train"]["epochs"]
lr = params["train"]["lr"]
gamma = params["train"]["gamma"]

hardware = params["hardware"]
BLOCK_BITS = params["hardware"]["block-bits"]
DELTA_BOUND = params["hardware"]["delta-bound"]
FILTER_SIZE = params["hardware"]["filter-size"]
BITMAP_SIZE = params["hardware"]["bitmap-size"]

Degree = params["simulator"]["degree"]

model = None 

def data_generator_gen(file_path,TRAIN_NUM,TOTAL_NUM,SKIP_NUM):
    _, eval_data = read_load_trace_data(file_path, TRAIN_NUM,TOTAL_NUM,SKIP_NUM)
    df_test = preprocessing_gen(eval_data, hardware)
    
    test_dataset = MAPDataset_gen(df_test)
    
    dev_dataloader=DataLoader(test_dataset,batch_size=batch_size,shuffle=False,collate_fn=test_dataset.collate_fn)
    
    return dev_dataloader,df_test

# def last_layer_prediction_gen(test_loader, test_df, model_save_path):#"top_k";"degree";"optimal"
#     print("predicting")
#     prediction=[]
#     model.load_state_dict(torch.load(model_save_path))
#     model.to(device)
#     model.eval()
#     y_score=np.array([])
#     for data in tqdm(test_loader):
#         output = nn.Sigmoid(model(data))
#         #prediction.extend(output.cpu())
#         prediction.extend(output.cpu().detach().numpy())
#     test_df["y_score"]= prediction

#     return test_df[['id', 'cycle', 'addr', 'ip','block_address', 'y_score']]

def evaluate(y_test,y_pred_bin):
    f1_score_res=f1_score(y_test, y_pred_bin, average='samples')
    #recall: tp / (tp + fn)
    recall_score_res=recall_score(y_test, y_pred_bin, average='samples')
    #precision: tp / (tp + fp)
    precision_score_res=precision_score(y_test, y_pred_bin, average='samples',zero_division=0)
    print("p,r,f1:",precision_score_res,recall_score_res,f1_score_res)
    return precision_score_res,recall_score_res,f1_score_res

##########################################################################################################
#%% New post_processing_delta_bitmap

def convert_hex(pred_block_addr):
    hex_val = int(pred_block_addr[0]) << BLOCK_BITS 
    hex_str = hex_val.to_bytes(((hex_val.bit_length() + 7) // 8), "big").hex().lstrip('0') 
    return hex_str 

def add_delta(block_address, pred_index):
    pred_delta = np.where(pred_index < DELTA_BOUND, pred_index+1, pred_index-BITMAP_SIZE) 
    return block_address + pred_delta

def bitmap_to_index_list(y_score, threshold):
    sorted_pred_index = np.argsort(y_score)[-np.count_nonzero(y_score >= threshold):]
    #return sorted index
    return sorted_pred_index

def post_processing_delta_filter(df,opt_threshold,filtering=True):
    print("post_processing, opt_threshold<0.9")
    if opt_threshold>0.9:
        opt_threshold=0.5
    df["pred_index"]=df.apply(lambda x: bitmap_to_index_list(x['y_score'], opt_threshold), axis=1)
    df=df.dropna()[['id', 'cycle', 'block_address', 'pred_index']]
    #add delta to block address
    df['pred_block_addr'] = df.apply(lambda x: add_delta(x['block_address'], x['pred_index']), axis=1)
    if filtering==True:
        #filter
        print("filtering")
        que = []
        pref_flag=[]
        dg_counter=0
        df["id_diff"]=df["id"].diff()
        for index, row in df.iterrows():
            if row["id_diff"]!=0:
                que.append(row["block_address"])
                dg_counter=0
            pred=row["pred_block_addr"]
            if dg_counter<Degree:
                if pred in que:
                    pref_flag.append(0)
                else:
                    que.append(pred)
                    pref_flag.append(1)
                    dg_counter+=1
            else:
                pref_flag.append(0)
            que=que[-FILTER_SIZE:]
        
        df["pref_flag"]=pref_flag
        df=df[df["pref_flag"]==1]
    
    df['pred_hex'] = df['pred_block_addr'].apply(convert_hex)
    df=df[["id","pred_hex"]]
    return df

#%%
def degree_stats(df,app_name,degree_stats_path):
    dic_dgsts={}
    dfc=df.groupby(["id"]).size().reset_index(name='counts')
    dfc=dfc.agg(["mean","max","min","median"])
    dic_dgsts["app"],dic_dgsts["mean"],dic_dgsts["max"],dic_dgsts["min"],dic_dgsts["median"]=\
        [app_name],[dfc['counts']["mean"]],[dfc['counts']["max"]],[dfc['counts']["min"]],[dfc['counts']["median"]]
    df_sts=pd.DataFrame(dic_dgsts)
    pd.DataFrame(df_sts).to_csv(degree_stats_path,header=1, index=False, sep=",") #pd_read=pd.read_csv(val_res_path,header=0,sep=" ")
    print(df_sts)
    print("Done: results saved at:", degree_stats_path)

#%%
if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("Usage: python src/generate_amm.py tracefile model_option K,...,K N,...,N gpu_id")
        exit(1)
    
    app = sys.argv[1]
    model_option = sys.argv[2]

    K_CLUSTER = [int(c) for c in sys.argv[3].split(",")]
    N_SUBSPACE = [int(d) for d in sys.argv[4].split(",")]

    gpu_id = sys.argv[5]
    
    file_path = os.path.join(trace_dir, app) 
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

    init_dataloader(gpu_id)

    model_save_path = os.path.join(model_dir, f"{app[:-7]}.{model_option}.pkl") 
    amm_df_path = model_save_path[:-4] + ".k."+".".join(map(str, K_CLUSTER))+".n."+".".join(map(str, N_SUBSPACE))+".amm_df.pkl"

    res_path = replace_directory(model_save_path, res_dir)

    with open(amm_df_path, 'rb') as f:
        test_df = pickle.load(f)

    with open(res_path+".val_res.json") as json_file:
        data = json.load(json_file)
    validation_list = data.get("validation")
    opt_threshold = validation_list[0].get("threshold")
    
    res_path += ".k."+".".join(map(str, K_CLUSTER))+".n."+".".join(map(str, N_SUBSPACE))+".amm_report.json"
    
    test_df = post_processing_delta_filter(test_df, opt_threshold, filtering=False)
    
    path_to_prefetch_file = res_path+".prefetch_file.txt"
    test_df[["id", "pred_hex"]].to_csv(path_to_prefetch_file, header=False, index=False, sep=" ")

    degree_stats_path = res_path+".degree_stats.csv"
    degree_stats(test_df[["id"]], app[:-7], degree_stats_path)
