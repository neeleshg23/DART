import numpy as np
import json
import os
import yaml

from models.mlp_mixer import MLPMixer
from models.mlp_simple import MLP
from models.mlp_teacher import MLPTeacher
from models.resnet import resnet_tiny, resnet50
from models.vit import TMAP

def select_model(option):
    with open("params.yaml", "r") as p:
        params = yaml.safe_load(p)
        
    image_size = (params["hardware"]["look-back"]+1, params["hardware"]["block-num-bits"]//params["hardware"]["split-bits"]+1)
    patch_size = (1, image_size[1])
    num_classes = 2*params["hardware"]["delta-bound"]
    
    if option == "mix":
        dim = params["model"][f"{option}"]["dim"]
        depth = params["model"][f"{option}"]["depth"]
        channels = params["model"][f"{option}"]["channel_dim"]
        return MLPMixer(
            in_channels=1, 
            image_size=image_size[0], 
            patch_size=2, 
            num_classes=num_classes,
            dim=dim,
            depth=depth,
            token_dim=dim, 
            channel_dim=channels 
        )
    elif option == "mixt":
        dim = params["model"][f"{option}"]["dim"]
        depth = params["model"][f"{option}"]["depth"]
        channels = params["model"][f"{option}"]["channel_dim"]
        return MLPMixer(
            in_channels=1, 
            image_size=image_size[0], 
            patch_size=2, 
            num_classes=num_classes,
            dim=dim,
            depth=depth,
            token_dim=dim, 
            channel_dim=channels 
        )
    elif option == "ms":
        input_size = params["model"][f"{option}"]["input_size"]
        hidden_size = params["model"][f"{option}"]["hidden_size"]
        num_classes = params["model"][f"{option}"]["num_classes"]
        return MLP(
            input_size = input_size,
            hidden_size = hidden_size,
            num_classes = num_classes
        )
    elif option == "mst":
        input_size = params["model"][f"{option}"]["input_size"]
        hidden_size = params["model"][f"{option}"]["hidden_size"]
        num_classes = params["model"][f"{option}"]["num_classes"]
        return MLPTeacher(
            input_size = input_size,
            hidden_size = hidden_size,
            num_classes = num_classes
        )
    elif option == "rs":
        channels = params["model"][option]["channels"]
        dim = params["model"][option]["dim"]
        return resnet_tiny(num_classes, channels, dim) 
    elif option == "rst":
        channels = params["model"][option]["channels"]
        dim = params["model"][option]["dim"]
        return resnet50(num_classes, channels, dim) 
    elif option == "vit":
        return TMAP (
            image_size=image_size,
            patch_size=patch_size,
            num_classes=num_classes,
            dim = params["model"][option]["dim"],
            depth = params["model"][option]["depth"],
            heads = params["model"][option]["heads"],
            mlp_dim = params["model"][option]["mlp_dim"],
            channels = params["model"][option]["channels"],
            dim_head = params["model"][option]["mlp_dim"]
        )
    elif option == "vitt":
        return TMAP (
            image_size=image_size,
            patch_size=patch_size,
            num_classes=num_classes,
            dim = params["model"][option]["dim"],
            depth = params["model"][option]["depth"],
            heads = params["model"][option]["heads"],
            mlp_dim = params["model"][option]["mlp_dim"],
            channels = params["model"][option]["channels"],
            dim_head = params["model"][option]["mlp_dim"]
        )


def replace_directory(path, new_directory):
    parts = path.split('/')
    parts[-2] = new_directory
    new_path = '/'.join(parts)
    return new_path

def select_clu(df_train, df_test, option):
    if option == "p":
        data_train = df_train['past_page'].values
        data_train = np.array(data_train.tolist())
        
        data_test = df_test['past_page'].values
        data_test = np.array(data_test.tolist())
        return data_train, data_test

    if option == "i":
        data_train = df_train['past_ip'].values
        data_train = np.array(data_train.tolist())

        data_test = df_test['past_ip'].values
        data_test = np.array(data_test.tolist())
        return data_train, data_test

    if option == "d":
        data_train = df_test['past_delta'].values
        data_train = np.array(data_train.tolist())
        
        data_test = df_test['past_delta'].values
        data_test = np.array(data_test.tolist())
        return data_train, data_test
    
    if option == "a":
        data_train = df_test['past_block_addr'].values
        data_train = np.array(data_train.tolist())
        
        data_test = df_test['past_block_addr'].values
        data_test = np.array(data_test.tolist())
        return data_train, data_test

