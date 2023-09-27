import sys
import os
import pandas as pd
import torch.nn as nn
import torch
import numpy as np
import pickle
from torchinfo import summary
import json
import pprint
import yaml
from sklearn.metrics import f1_score,recall_score,precision_score

from utils import select_model, replace_directory
from metrics import _cossim
import vq_amm

def evaluate(y_test,y_pred_bin):
    f1_score_res=f1_score(y_test, y_pred_bin, average='micro')
    #recall: tp / (tp + fn)
    recall_score_res=recall_score(y_test, y_pred_bin, average='micro')
    #precision: tp / (tp + fp)
    precision_score_res=precision_score(y_test, y_pred_bin, average='micro',zero_division=0)
    print("p,r,f1:",precision_score_res,recall_score_res,f1_score_res)
    return precision_score_res,recall_score_res,f1_score_res

def _eval_amm_layer(est, X, W, fixedB=True):
    est.reset_for_new_task()
    if fixedB:
        est.set_B(W)
    y_hat = est.predict(X, W)
    return y_hat

def evaluate_by_score(y_score, threshold, y_label):
    y_pred_bin = (y_score - np.array(threshold) > 0) * 1
    p, r, f1 = evaluate(y_label, y_pred_bin)
    return [p, r, f1]

def load_data_n_model(model_save_path, res_path):
    tensor_dict_path = model_save_path + '.tensor_dict.pkl'
    # Load the dictionary using pickle
    with open(tensor_dict_path, 'rb') as f:
        tensor_dict = pickle.load(f)
    train_data, train_target, test_data, test_target = \
        tensor_dict['train_data'], tensor_dict['train_target'], tensor_dict['test_data'], tensor_dict['test_target']

    # define and load model
    model.load_state_dict(torch.load(model_save_path))
    all_params = list(model.named_parameters())

    # load json in res dir. for threshold
    with open(res_path+".val_res.json", "r") as json_file:
        data = json.load(json_file)
    
    validation_list = data.get("validation")
    best_threshold = validation_list[0].get("threshold")
#    highest_f1 = -1  
#    best_threshold = None

#    for entry in validation_list:
#        f1 = entry.get("f1")
#        threshold = entry.get("threshold")
#        
#        if f1 is not None and f1 > highest_f1:
#            highest_f1 = f1
#            best_threshold = threshold
    return train_data, train_target, test_data, test_target, all_params, best_threshold

## Build a func. to pick layers to matmul vs. estimating
def mlp_inference_by_layers(input_data):
    mlp_flatten = nn.Flatten()
    mlp_relu = nn.ReLU()
    mlp_sigmoid = nn.Sigmoid()

    model_input = mlp_flatten(input_data)
    layer1_output = torch.matmul(model_input, all_params[0][1].t()) + all_params[1][1]  # note, need transpose
    layer1_output_activated = mlp_relu(layer1_output)

    layer2_output = torch.matmul(layer1_output_activated, all_params[2][1].t()) + all_params[3][1]
    layer2_output_activated = mlp_relu(layer2_output)

    layer3_output = torch.matmul(layer2_output_activated, all_params[4][1].t()) + all_params[5][1]
    layer3_output_activated = mlp_sigmoid(layer3_output)

    return [model_input, layer1_output, layer1_output_activated, layer2_output,
            layer2_output_activated, layer3_output, layer3_output_activated]


def validate_whole_model_vs_layers(train_data, train_target, test_data, test_target, best_threshold):
    y_score_by_whole_train = model(train_data).detach().numpy()
    y_score_by_whole_test = model(test_data).detach().numpy()

    mlp_inf_layer_res_train = mlp_inference_by_layers(train_data)
    mlp_inf_layer_res_test = mlp_inference_by_layers(test_data)

    y_score_by_layer_train = mlp_inf_layer_res_train[-1].detach().numpy()
    y_score_by_layer_test = mlp_inf_layer_res_test[-1].detach().numpy()

    f1_w_tr = evaluate_by_score(y_score_by_whole_train, best_threshold, train_target)
    f1_l_tr = evaluate_by_score(y_score_by_layer_train, best_threshold, train_target)

    f1_w_ts = evaluate_by_score(y_score_by_whole_test, best_threshold, test_target)
    f1_l_ts = evaluate_by_score(y_score_by_layer_test, best_threshold, test_target)

    def are_vectors_equal(vector1, vector2, tolerance=1e-9):
        return all(abs(x - y) < tolerance for x, y in zip(vector1, vector2))

    if are_vectors_equal(f1_w_tr, f1_l_tr) and are_vectors_equal(f1_w_ts, f1_l_ts) :
        print("Layer Validation Success: the whole model and layered model are nearly identical.")
    else:
        print("Layer Validation Fail! The models are different! ")

    return mlp_inf_layer_res_train, mlp_inf_layer_res_test, f1_l_tr, f1_l_ts


def layer_estimator(X_train, X_test, W, bias, act_fn, n=2, k=16):
    est = vq_amm.PQMatmul(ncodebooks=n, ncentroids=k)  # k must >= 16
    est.fit(X_train, W)
    #X_train.shape: (88462, 100), W.shape: (100, 64), output: (88462, 64)
    est_hat_train = _eval_amm_layer(est, X_train, W)
    est_hat_test = _eval_amm_layer(est, X_test, W)
    est_hat_output_train = act_fn(torch.Tensor(est_hat_train)+bias)
    est_hat_output_test = act_fn(torch.Tensor(est_hat_test) + bias)
    return [est, est_hat_train, est_hat_test, est_hat_output_train.detach().numpy(), est_hat_output_test.detach().numpy()]


def layer_cossim(mlp_layer_exact_res_train,mlp_layer_exact_res_test, est_l1_res, est_l2_res, est_l3_res):
    cossim_layer_train = [0] * 3
    cossim_layer_test = [0] * 3
    cossim_layer_train[0] = _cossim(mlp_layer_exact_res_train[2].detach().numpy(), est_l1_res[-2])
    cossim_layer_test[0] = _cossim(mlp_layer_exact_res_test[2].detach().numpy(), est_l1_res[-1])
    print("layer1 cos similarity on train data:", cossim_layer_train[0])
    print("layer1 cos similarity on test data:", cossim_layer_test[0])

    cossim_layer_train[1] = _cossim(mlp_layer_exact_res_train[4].detach().numpy(), est_l2_res[-2])
    cossim_layer_test[1] = _cossim(mlp_layer_exact_res_test[4].detach().numpy(), est_l2_res[-1])
    print("layer2 cos similarity on train data:", cossim_layer_train[1])
    print("layer2 cos similarity on test data:", cossim_layer_test[1])

    cossim_layer_train[2] = _cossim(mlp_layer_exact_res_train[6].detach().numpy(), est_l3_res[-2])
    cossim_layer_test[2] = _cossim(mlp_layer_exact_res_test[6].detach().numpy(), est_l3_res[-1])
    print("layer3 cos similarity on train data:", cossim_layer_train[2])
    print("layer3 cos similarity on test data:", cossim_layer_test[2])
    
    return [float(value) for value in cossim_layer_train], [float(value) for value in cossim_layer_test]


##

##################################################################################################
# main
##################################################################################################

with open("params.yaml", "r") as p:
    params = yaml.safe_load(p)

model_dir = params["system"]["model"]
res_dir = params["system"]["res"]

app = sys.argv[1]
option = sys.argv[2]

K_CLUSTER = [int(c) for c in sys.argv[3].split(",")]
N_SUBSPACE = [int(d) for d in sys.argv[4].split(",")]

gpu_id = int(sys.argv[5])

model = select_model(option.split(".")[0])
model_save_path = os.path.join(model_dir, f"{app[:-7]}.{option}.pkl") 
res_path = replace_directory(model_save_path, res_dir) 

train_data, train_target, test_data, test_target, all_params, best_threshold = load_data_n_model(model_save_path, res_path)

res_path += ".k."+".".join(map(str, K_CLUSTER))+".n."+".".join(map(str, N_SUBSPACE))

mlp_layer_exact_res_train, mlp_layer_exact_res_test, f1_l_tr, f1_l_ts = validate_whole_model_vs_layers(train_data, train_target,test_data, test_target, best_threshold)

est_l1_res = layer_estimator(X_train=mlp_layer_exact_res_train[0].detach().numpy(),
                             X_test=mlp_layer_exact_res_test[0].detach().numpy(),
                             W=all_params[0][1].t().detach().numpy(),
                             bias=all_params[1][1],
                             act_fn=nn.ReLU(),
                             n=N_SUBSPACE[0],
                             k=K_CLUSTER[0])


est_l2_res = layer_estimator(X_train=mlp_layer_exact_res_train[2].detach().numpy(),
                             X_test=est_l1_res[-1], #output of layer1
                             W=all_params[2][1].t().detach().numpy(),
                             bias=all_params[3][1],
                             act_fn=nn.ReLU(),
                             n=N_SUBSPACE[1],
                             k=K_CLUSTER[1])

est_l3_res = layer_estimator(X_train=mlp_layer_exact_res_train[4].detach().numpy(),
                             X_test=est_l2_res[-1], #output of layer2
                             W=all_params[4][1].t().detach().numpy(),
                             bias=all_params[5][1],
                             act_fn=nn.Sigmoid(),
                             n=N_SUBSPACE[2],
                             k=K_CLUSTER[2])

cossim_layer_train, cossim_layer_test = layer_cossim(mlp_layer_exact_res_train,mlp_layer_exact_res_test, est_l1_res, est_l2_res, est_l3_res)

print("exact p,r,f1: \n",f1_l_ts)
print("amm estimation:")
f1_est_ts = evaluate_by_score(est_l3_res[-1],best_threshold,test_target)

## size analysis
summary(model)
total_params = sum(p.numel() for p in model.parameters())
print ("MLP total parameters:", total_params)

# table size
lut_l1, lut_l2, lut_l3=est_l1_res[0].luts, est_l2_res[0].luts, est_l3_res[0].luts
print("LUT shapes = (output dimension, k clusters, n subspaces):", lut_l1.shape, lut_l2.shape, lut_l3.shape)
lut_total_size =  lut_l1.size + lut_l2.size + lut_l3.size
print("LUT total size = ", lut_total_size)

#%%
## output report
report = {
    'model': {
        'name': 'MLP',
        'threshold': best_threshold, 
        'layer': 3,
        'dim': params['model']['ms']['hidden_size'],
        'f1': f1_l_ts,
        'num_param': total_params
    },
    'estimator': {
        'method': 'PQ_KMEANS',
        'N_SUBSPACE': N_SUBSPACE,
        'K_CLUSTER': K_CLUSTER,
        'cossim_train': cossim_layer_train,
        'cossim_test': cossim_layer_test,
        'f1': f1_est_ts,
        'lut_size': lut_total_size
    }
}

pprint.pprint(report,sort_dicts=False)
with open(res_path+'.amm_report.json', 'w') as json_file:
    json.dump(report, json_file,indent=2)
