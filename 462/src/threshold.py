import numpy as np
from numpy import nanargmax, sqrt
from tqdm import tqdm 
import yaml
from sklearn.metrics import f1_score
from sklearn.metrics import auc, f1_score, precision_score, recall_score, precision_recall_curve, roc_curve
import torch
from torch import nn
from torch.utils.data import DataLoader

from data_loader import init_dataloader, MAPDataset
from prep_single import read_load_trace_data, preprocessing
from utils import select_model

batch_size = 256
sigmoid = nn.Sigmoid()

def threshold_throttleing(train_df, throttle_type="f1", optimal_type="micro", topk=2, threshold=0.5):
    y_real_torch = train_df['future']
    y_real = [tensor.cpu() for tensor in y_real_torch]
    y_score = train_df["y_score"]
    if len(y_real[-1]) != 256:
        y_real = y_real[:-1]
        y_score = y_score[:-1*(len(y_score)-256*len(y_real))]
    
    # WANT
    y_real = np.stack(y_real) 
    y_score = np.stack(y_score)
    
    # -- F1 Score Throttling Using Precision-Recall Curve -- 
    print("F1 Score Throttling Using Precision-Recall Curve")
    
    p, r, threshold, fscore, ix = {}, {}, {}, {}, {}
    best_threshold = []
   
    p["micro"], r["micro"], threshold["micro"] = precision_recall_curve(y_real.ravel(), y_score.ravel())
    fscore["micro"] = (2 * p["micro"] * r["micro"]) / (p["micro"] + r["micro"])
    ix["micro"] = nanargmax(fscore["micro"])
    best_threshold = threshold["micro"][ix["micro"]]
   
    print(f"Best sampled threshold={best_threshold:.4f}") 
    
    y_pred_bin = (y_score-best_threshold > 0)*1
    train_df["predicted"] = list(y_pred_bin)
    
    return train_df, best_threshold
def find_optimal_threshold(model, device, train_loader, model_save_path, n_samples): 
    print("Finding optimal threshold based off of sampled train data")
    prediction = []
    model.load_state_dict(torch.load(model_save_path))
    model.to(device)
    model.eval()
    all_targets = [] 
    counter = 0
    for data, target in tqdm(train_loader):
        if counter > n_samples:
            break
        output = sigmoid(model(data))
        all_targets.append(target)
        prediction.extend(output.cpu().detach().numpy())
        counter += 1 
    
    new_sampled_df = {}
    new_sampled_df["future"] = all_targets
    new_sampled_df["y_score"] = prediction
    
    _, th = threshold_throttleing(new_sampled_df, throttle_type="f1", optimal_type="micro")
    return th

'''
# -- Extra Wrapper Method for Testing Main -- 
def data_generator(file_path,TRAIN_NUM,TOTAL_NUM,SKIP_NUM,only_val=False):
    with open('params.yaml', 'r') as p:
        params = yaml.safe_load(p)
    hardware = params['hardware']
    if only_val==True:
        print("only validation")
        _, eval_data = read_load_trace_data(file_path, TRAIN_NUM,TOTAL_NUM,SKIP_NUM)
        df_test = preprocessing(eval_data, hardware)
        test_dataset = MAPDataset(df_test)

        #logging.info("-------- Dataset Build! --------")
        dev_dataloader=DataLoader(test_dataset,batch_size=batch_size,shuffle=False,collate_fn=test_dataset.collate_fn)

        return dev_dataloader, df_test
    else:
        print("train and validation")
        train_data, eval_data = read_load_trace_data(file_path, TRAIN_NUM,TOTAL_NUM,SKIP_NUM)
        df_train = preprocessing(train_data, hardware)
        df_test = preprocessing(eval_data, hardware)

        train_dataset = MAPDataset(df_train)
        test_dataset = MAPDataset(df_test)

        #logging.info("-------- Dataset Build! --------")
        train_dataloader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True,collate_fn=train_dataset.collate_fn)
        dev_dataloader=DataLoader(test_dataset,batch_size=batch_size,shuffle=False,collate_fn=test_dataset.collate_fn)

        return train_dataloader, dev_dataloader, df_train,df_test

# -- Testing Main -- 
init_dataloader('0')
train_loader, test_loader, df_train, df_test = data_generator('/data/pengmiao/ML-DPC-S0/LoadTraces/410.bwaves-s0.txt.xz', 2, 3, 1)
model = select_model('ms')
device = torch.device('cuda:0')
print("threshold", find_optimal_threshold(model, device, train_loader, './model/410.bwaves-s0.ms.pkl', 10))
'''
