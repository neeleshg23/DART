import gc
import itertools
import lzma
import os
import pickle
import sys
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import torch
import warnings

import yaml

from utils import select_clu
warnings.filterwarnings("ignore")
from torch.utils.data import DataLoader

from data_loader import MAPDataset, init_dataloader

def read_load_trace_data(load_trace, num_prefetch_warmup_instructions, num_total_instructions, skipping=0):
    def process_line(line):
        split = line.strip().split(', ')
        return int(split[0]), int(split[1]), int(split[2], 16), int(split[3], 16), split[4] == '1'

    train_data = []
    eval_data = []
    if load_trace[-2:] == 'xz':
        with lzma.open(load_trace, 'rt') as f:
            for line in f:
                pline = process_line(line)
                if pline[0]>skipping*1000000:
                    if pline[0] < num_prefetch_warmup_instructions * 1000000:
                        train_data.append(pline)
                    else:
                        if pline[0] < num_total_instructions * 1000000:
                            eval_data.append(pline)
                        else:
                            break
    else:
        with open(load_trace, 'r') as f:
            for line in f:
                pline = process_line(line)
                if pline[0]>skipping*1000000:
                    if pline[0] < num_prefetch_warmup_instructions * 1000000:
                        train_data.append(pline)
                    else:
                        if pline[0] < num_total_instructions * 1000000:
                            eval_data.append(pline)
                        else:
                            break

    return train_data, eval_data

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

def split_to_words(value,BN_bits=58,split_bits=6,norm=True):
    #res=[SPLITER_ID]
    res=[]
    for i in range(BN_bits//split_bits+1):
        divider=2**split_bits
        #res.append(value%(divider)+OFFSET)#add 1, range(1-64),0 as padding
        new_val=value%(divider)
        if norm==True:
            new_val=new_val/divider
        res.append(new_val)#
        value=value//divider
    return res

def delta_acc_list(delta,DELTA_BOUND=128):#delta accumulative list
    res=list(itertools.accumulate(delta))
    res=[i for i in res if abs(i)<=DELTA_BOUND]
    if len(res)==0:
        res="nan"
    return res


def addr_hash(x,HASH_BITS):
    t = int(x)^(int(x)>>32); 
    result = (t^(t>>HASH_BITS)) & (2**HASH_BITS-1); 
    return result/(2**HASH_BITS)

def ip_list_norm(ip_list,HASH_BITS):
    return [addr_hash(ip,HASH_BITS) for ip in ip_list]

def page_list_norm(page_list,current_page):
    return list(1/(abs(np.array(page_list)-current_page)+1))
     
    
def preprocessing(data, hardware):
    print("preprocessing with context")

    BLOCK_BITS, PAGE_BITS, BLOCK_NUM_BITS, SPLIT_BITS, LOOK_BACK, PRED_FORWARD, DELTA_BOUND, BITMAP_SIZE = hardware["block-bits"], hardware["page-bits"], hardware["block-num-bits"], hardware["split-bits"], hardware["look-back"], hardware["pred-forward"], hardware["delta-bound"], hardware["bitmap-size"]

    df = pd.DataFrame(data)
    df.columns=["id", "cycle", "addr", "ip", "hit"]
    df['raw']=df['addr']
    df['block_address'] = [x >> BLOCK_BITS for x in df['raw']]
    df['page_address'] = [x >> PAGE_BITS for x in df['raw']]
    df['page_offset'] = [x - (x >> PAGE_BITS << PAGE_BITS) for x in df['raw']]
    df['block_index'] = [int(x >> BLOCK_BITS) for x in df['page_offset']]  
    df['block_addr_delta'] = df['block_address'].diff()
    
    df['patch'] = df.apply(lambda x: split_to_words(x['block_address'],BLOCK_NUM_BITS,SPLIT_BITS),axis=1)
    
    # past
    for i in range(LOOK_BACK):
        df['delta_past_%d'%(i+1)]=df['block_addr_delta'].shift(periods=(i+1))
        df['block_addr_past_%d'%(i+1)]=df['block_address'].shift(periods=(i+1))
        df['patch_past_%d'%(i+1)]=df['patch'].shift(periods=(i+1))
        df['ip_past_%d'%(i+1)]=df['ip'].shift(periods=(i+1))
        df['page_past_%d'%(i+1)]=df['page_address'].shift(periods=(i+1))
    
    #Pem, update, debug 2019/09/18
    past_addr_name=['block_addr_past_%d'%(i) for i in range(LOOK_BACK,0,-1)]
    past_name=['patch_past_%d'%(i) for i in range(LOOK_BACK,0,-1)]
    past_ip_name=['ip_past_%d'%(i) for i in range(LOOK_BACK,0,-1)]
    past_page_name=['page_past_%d'%(i) for i in range(LOOK_BACK,0,-1)]
    past_delta_name=['delta_past_%d'%(i) for i in range(LOOK_BACK,0,-1)]
    past_name.append('patch')
    # past_ip_name.append('ip')
    past_page_name.append('page_address')
    #Pem, update done
    
    df["past"]=df[past_name].values.tolist()
    df["past_block_addr_abs"]=df[past_addr_name].values.tolist()
    df["past_ip_abs"]=df[past_ip_name].values.tolist()
    df["past_page_abs"]=df[past_page_name].values.tolist()
    df['past_delta_abs']=df[past_delta_name].values.tolist()

    df=df.dropna()

    df['past_block_addr']=df.apply(lambda x: [item for item in x['past_block_addr_abs'] if pd.notnull(item)], axis=1)
    df['past_delta']=df.apply(lambda x: [item for item in x['past_delta_abs'] if pd.notnull(item)], axis=1)
    df['past_ip']=df.apply(lambda x: ip_list_norm(x['past_ip_abs'],16),axis=1)
    df['past_page']=df.apply(lambda x: page_list_norm(x['past_page_abs'],x['page_address']),axis=1)
    
    #labels
    '''
    future_idx: delta to the prior addr
    future_delta: accumulative delta to current addr
    '''
    for i in range(PRED_FORWARD):
        df['delta_future_%d'%(i+1)]=df['block_addr_delta'].shift(periods=-(i+1))
    
    for i in range(PRED_FORWARD):
            if i==0:
                df["future_idx"]=df[['delta_future_%d'%(i+1)]].values.astype(int).tolist()
            else:   
                df["future_idx"]=np.hstack((df["future_idx"].values.tolist(), df[['delta_future_%d'%(i+1)]].values.astype(int))).tolist()
                
                #df[['delta_future_%d'%(i+1)]].values.tolist()
    
    #delta bitmap
    df["future_delta"]=df.apply(lambda x: delta_acc_list(x['future_idx'],DELTA_BOUND),axis=1)
    
    df=df[df["future_delta"]!="nan"]
    
    df["future"]=df.apply(lambda x: to_bitmap(x['future_delta'],BITMAP_SIZE),axis=1)
    df=df.dropna()
    return df[['id', 'cycle', 'addr', 'ip', 'hit', 'raw', 'block_address',
       'page_address', 'page_offset', 'block_index', 'block_addr_delta',
       'patch','past','past_block_addr','past_ip','past_page','past_delta','future']]


def main():
    with open("params.yaml", "r") as p:
        params = yaml.safe_load(p)
    trace_dir = params["system"]["traces"]
    processed_dir = params["system"]["processed"]

    app = sys.argv[1]
    app_name = app[:-7]

    gpu_id = sys.argv[2]
    init_dataloader(gpu_id)

    num_tch =params["cluster"]["number"] 
    cluster_option = params["cluster"]["option"]
    
    train   = params["trace-data"]["train"]
    total   = params["trace-data"]["total"]
    skip    = params["trace-data"]["skip"]
    n_batch = params["trace-data"]["batch-size"]
    hw      = params["hardware"]
    
    os.makedirs(os.path.join(processed_dir), exist_ok=True)
    os.makedirs(os.path.join(processed_dir, cluster_option), exist_ok=True)

    train_data, eval_data = read_load_trace_data(os.path.join(trace_dir, app), train, total, skip)

    df_train= preprocessing(train_data, hw)
    df_test = preprocessing(eval_data, hw)

    data_train, data_test = select_clu(df_train, df_test, cluster_option)

    kmeans = KMeans(n_clusters=num_tch, init='k-means++') 

    # print('train', data_train.shape)
    df_train['cluster'] = kmeans.fit_predict(data_train)
    # print('test', data_test.shape)
    df_test['cluster'] = kmeans.predict(data_test)

    train_MAP_stu = MAPDataset(df_train)
    train_loader_stu = DataLoader(train_MAP_stu, batch_size=n_batch, shuffle=True, collate_fn=train_MAP_stu.collate_fn)

    test_MAP_stu    = MAPDataset(df_test)
    test_loader_stu = DataLoader(test_MAP_stu, batch_size=n_batch, shuffle=False, collate_fn=test_MAP_stu.collate_fn)
    
    print("Beginning to save data for stu")

    with torch.no_grad():
        torch.save(df_test, os.path.join(processed_dir, cluster_option, f"{app_name}.df_stu.pt"))
        torch.save(train_loader_stu, os.path.join(processed_dir, cluster_option, f"{app_name}.train_stu.pt"))
        torch.save(test_loader_stu, os.path.join(processed_dir, cluster_option, f"{app_name}.test_stu.pt"))
    
    for tch in range(num_tch):
        train_idx = np.where(df_train['cluster'] == tch)[0]
        test_idx = np.where(df_test['cluster'] == tch)[0]

        df_train_tch = df_train.iloc[train_idx]
        df_test_tch = df_test.iloc[test_idx]

        train_data_tch = MAPDataset(df_train_tch)
        test_data_tch = MAPDataset(df_test_tch)

        train_loader = DataLoader(train_data_tch, batch_size=n_batch, shuffle=True, collate_fn=train_data_tch.collate_fn)
        test_loader = DataLoader(test_data_tch, batch_size=n_batch, shuffle=False, collate_fn=test_data_tch.collate_fn)

        print(f"Starting saving for tch:{tch+1} of {num_tch}")
        
        with torch.no_grad():
            torch.save(train_loader, os.path.join(processed_dir, cluster_option, f"{app_name}.train_{tch+1}.pt"))
            torch.save(test_loader, os.path.join(processed_dir, cluster_option, f"{app_name}.test_{tch+1}.pt"))
            torch.save(df_test_tch, os.path.join(processed_dir, cluster_option, f"{app_name}.df_{tch+1}.pt"))

        print(f"DONE FOR tch:{tch+1} of {num_tch}")

    print("All loaders saved!")

if __name__ == "__main__":
    main()
