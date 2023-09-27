import numpy as np
import torch
from einops import rearrange
from torch.utils.data import Dataset
import yaml

## Potential Issues, Channels value is hardcoded in as 1 since DenseNet teacher model has 1 channel
with open("params.yaml", "r") as p:
    params = yaml.safe_load(p)
channels = params["model"]["rs"]["channels"]
image_size = (params["hardware"]["look-back"]+1, params["hardware"]["block-num-bits"]//params["hardware"]["split-bits"]+1)
## --

def init_dataloader(gpu_id):
    global device
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

class MAPDataset(Dataset):
    def __init__(self, df):
        self.past=list(df["past"].values)
        self.future=list(df["future"].values)

    def __getitem__(self, idx):
        past = self.past[idx]
        future = self.future[idx]
        return [past,future]

    def __len__(self):
        return len(self.past)
    
    
    def collate_fn(self, batch):
        past_b = [x[0] for x in batch]
        future_b = [x[1] for x in batch]
#        data=rearrange(past_b, '(b h c) w-> b c h w',c=cf.channels,b=batch,h=cf.image_size[0],w=cf.image_size[1])
#        print(np.array(past_b).shape)

        data=rearrange(np.array(past_b), '(b c) h w-> b c h w',c=channels, h=image_size[0], w=image_size[1])

        past_tensor=torch.Tensor(data).to(device)
        
        future_tensor=torch.Tensor(future_b).to(device)
        
        return past_tensor, future_tensor

class MAPDataset_gen(Dataset):
    def __init__(self, df):
        self.past=list(df["past"].values)
   
    def __getitem__(self, idx):
        past = self.past[idx]
        return [past]

    def __len__(self):
        return len(self.past)
    
    def collate_fn(self, batch):
        
        past_b = [x[0] for x in batch]
        data=rearrange(np.array(past_b), '(b c) h w-> b c h w', c=channels, h=image_size[0], w=image_size[1])
        past_tensor=torch.Tensor(data).to(device)
    
        return past_tensor
