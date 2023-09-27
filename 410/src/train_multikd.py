import csv
import os
import sys
import warnings
import pickle
import yaml
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchinfo import summary
from statistics import mean

from data_loader import init_dataloader
from utils import select_model, replace_directory 
from validate import run_val

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

torch.manual_seed(100)

device = None
model = None
optimizer = None
scheduler = None
Temperature = None

sigmoid = nn.Sigmoid()
soft_loss = nn.KLDivLoss(reduction="mean", log_target=True)
#log = config.Logger()

def train(ep, alpha, train_loader, model_save_path, teacher_model):
    global steps
    epoch_loss = 0
    model.train()
    print('1', len(train_loader))
    print('2', len(train_loader.dataset))
    for batch_idx, (data, target) in enumerate(train_loader):#d,t: (torch.Size([64, 1, 784]),64)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        student_preds = model(data)

        with torch.no_grad():
            teacher_preds = teacher_model(data)

        student_loss = F.binary_cross_entropy(sigmoid(student_preds), target, reduction='mean')

        x_t_sig = sigmoid(teacher_preds / Temperature).reshape(-1)
        x_s_sig = sigmoid(student_preds / Temperature).reshape(-1)

        x_t_p = torch.stack((x_t_sig, 1 - x_t_sig), dim=1)
        x_s_p = torch.stack((x_s_sig, 1 - x_s_sig), dim=1)

        distillation_loss = soft_loss(x_s_p.log(), x_t_p.log())
        loss = alpha * student_loss + (1 - alpha) * distillation_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    epoch_loss/=len(train_loader)
    return epoch_loss

def test(test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = sigmoid(model(data))
            test_loss += F.binary_cross_entropy(output, target, reduction='mean').item()
            thresh=0.5
            output_bin=(output>=thresh)*1
            correct+=(output_bin&target.int()).sum()
        
        test_loss /=  len(test_loader)
        return test_loss   

def run_epoch(epochs, early_stop, alpha, loading, model_save_path, train_loaders, test_loaders, teacher_models, teacher_paths):
    if loading==True:
        model.load_state_dict(torch.load(model_save_path,map_location=device))
        print("-------------Model Loaded------------")
        
    best_loss=0
    curr_early_stop = early_stop
    num_clu = len(test_loaders)

    for epoch in range(epochs):
        print(f"------- START EPOCH {epoch+1} -------")

        train_losses = []
        test_losses = []

        logging = f"Epoch {epoch+1:2.0f}: "
        for i in range(num_clu): 
            teacher_model = teacher_models[i].to(device)
            teacher_model.load_state_dict(torch.load(teacher_paths[i], map_location=device))
            train_loss=train(epoch, alpha, train_loaders[i], model_save_path, teacher_model=teacher_model)
            test_loss=test(test_loaders[i])
            logging += f"T{i+1}: TrainL={train_loss:.5f}, TestL={test_loss:.5f}; "
            train_losses.append(train_loss)
            test_losses.append(test_loss) 

        print(logging)

        test_loss = mean(test_losses)
        if epoch == 0:
            best_loss=test_loss
        if test_loss<=best_loss:
            torch.save(model.state_dict(), model_save_path)    
            best_loss=test_loss
            print("-------- Save Best Model! --------")
            curr_early_stop = early_stop
        else:
            curr_early_stop -= 1
            print("Early Stop Left: {}".format(curr_early_stop))

        if curr_early_stop == 0:
            print("-------- Early Stop! --------")
            break

				
def save_data_for_amm(model_save_path, train_loader, test_loader, test_df):

    all_train_data,all_train_targets, all_test_data, all_test_targets = [],[],[],[]

    print("saving tensor pickle data")
    for batch_idx, (data, target) in enumerate(train_loader):
        all_train_data.append(data)
        all_train_targets.append(target)

    for batch_idx, (data, target) in enumerate(test_loader):
        all_test_data.append(data)
        all_test_targets.append(target)


    all_train_data = torch.cat(all_train_data, dim=0).cpu()
    all_train_targets = torch.cat(all_train_targets, dim=0).cpu()
    all_test_data = torch.cat(all_test_data, dim=0).cpu()
    all_test_targets = torch.cat(all_test_targets, dim=0).cpu()
    tensor_dict = {"train_data": all_train_data,
                   "train_target": all_train_targets,
                   "test_data": all_test_data,
                   "test_target": all_test_targets}

    with open(model_save_path+'.tensor_dict.pkl', 'wb') as f:
        pickle.dump(tensor_dict, f)

    print("tensor data saved")

    print("saving test dataframe")
    #train_df.to_pickle(res_root+"train_df.pkl")
    test_df[['id', 'cycle', 'addr', 'ip','block_address','future', 'y_score']].to_pickle(
        model_save_path+".test_df.pkl")

    print("done data saving for amm")

    return

def main():
    global device
    global model
    global optimizer
    global scheduler
    global Temperature

    with open("params.yaml", "r") as p:
        params = yaml.safe_load(p)

    app = sys.argv[1]
    app_name = app[:-7]
    
    t_option = sys.argv[2]
    option = sys.argv[3]
    
    gpu_id = sys.argv[4]
    init_dataloader(gpu_id)

    processed_dir = params["system"]["processed"]
    model_dir = params["system"]["model"]
    results_dir = params["system"]["res"]

    epochs = params["kd"]["epochs"]
    lr = params["kd"]["lr"]
    gamma = params["train"]["gamma"]
    step_size = params["train"]["step-size"]
    early_stop = params["train"]["early-stop"]

    clu_n = params["cluster"]["number"]
    cluster_option = params["cluster"]["option"]

    alpha = float(sys.argv[5]) 
    Temperature = params["kd"]["temperature"]
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

    os.makedirs(os.path.join(model_dir), exist_ok=True)

    teacher_models = [select_model(t_option) for _ in range(clu_n)]
    teacher_options = [t_option for _ in range(clu_n)]
    model = select_model(option)
    print(summary(model))
    model_save_path = os.path.join(model_dir, cluster_option, f"{app_name}.{option}.stu.{alpha*100}.pkl")
    res_path = os.path.join(results_dir, cluster_option, f"{app_name}.{option}.stu.{alpha*100}.pkl")
    teacher_paths = [os.path.join(model_dir, cluster_option, f"{app_name}.teacher_{i+1}.{option}.pth") for i, option in enumerate(teacher_options)]

    print("Loading data for model")
    
    train_loaders = []
    test_loaders = []

    for i, _ in enumerate(teacher_paths, start=1):
        curr_train_loader = torch.load(os.path.join(processed_dir, cluster_option, f"{app_name}.train_{i}.pt"))
        curr_test_loader = torch.load(os.path.join(processed_dir, cluster_option, f"{app_name}.test_{i}.pt"))
        train_loaders.append(curr_train_loader)
        test_loaders.append(curr_test_loader)
    
    test_df = torch.load(os.path.join(processed_dir, cluster_option, f"{app_name}.df_stu.pt"))
    test_loader = torch.load(os.path.join(processed_dir, cluster_option, f"{app_name}.test_stu.pt"))
    train_loader = torch.load(os.path.join(processed_dir, cluster_option, f"{app_name}.train_stu.pt"))
    
    print("Data loaded successfully for stu")

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

    loading = False

    run_epoch(epochs, early_stop, alpha, loading, model_save_path, train_loaders, test_loaders, teacher_models, teacher_paths)
    run_val(test_loader, test_df, app, model_save_path, res_path, option, gpu_id)
    save_data_for_amm(model_save_path, train_loader, test_loader, test_df)

if __name__ == "__main__":
    main()
