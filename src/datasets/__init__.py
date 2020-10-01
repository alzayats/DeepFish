
# from . import trancos, fish_reg
from . import fish_clf, fish_reg, fish_loc, fish_seg
from torchvision import transforms
import cv2, os
from src import utils as ut
import pandas as pd
import numpy as np

import pandas as pd
import  numpy as np
import os

def get_dataset(dataset_name, split, transform=None, 
                datadir=None, habitat=None):

    transform = get_transformer(transform, split)

    if dataset_name == "fish_clf":
        dataset = fish_clf.FishClf(split,
                               transform=transform,
                               datadir=datadir + "/Classification/", 
                               habitat=habitat)

    elif dataset_name == "fish_reg":
        dataset = fish_reg.FishReg(split,
                               transform=transform,
                              datadir=datadir + "/Localization/", 
                              habitat=habitat)

    elif dataset_name == "fish_loc":
        dataset = fish_loc.FishLoc(split,
                               transform=transform,
                               datadir=datadir+ "/Localization/", 
                               habitat=habitat)
    
    elif dataset_name == "fish_seg":
        dataset = fish_seg.FishSeg(split,
                              transform=transform,
                              datadir=datadir+ "/Segmentation/", 
                              habitat=habitat)

    # ===========================================================
    # tiny
    if dataset_name == "tiny_fish_clf":
        dataset = fish_clf.FishClf(split="train",
                               transform=transform,
                               datadir=datadir + "/Classification/", 
                               n_samples=10, 
                               habitat=habitat)

    elif dataset_name == "tiny_fish_reg":
        dataset = fish_reg.FishReg(split="train",
                               transform=transform,
                              datadir=datadir + "/Localization/", 
                              n_samples=10, 
                              habitat=habitat)

    elif dataset_name == "tiny_fish_loc":
        dataset = fish_loc.FishLoc(split="train",
                               transform=transform,
                               datadir=datadir+ "/Localization/", 
                               n_samples=10, 
                               habitat=habitat)
    
    elif dataset_name == "tiny_fish_seg":
        dataset = fish_seg.FishSeg(split="train",
                              transform=transform,
                              datadir=datadir+ "/Segmentation/", 
                              n_samples=10,
                              habitat=habitat)

    return dataset


# ===================================================
# helpers
def slice_df(df, habitat):
    if habitat is None:
        return df
    return df[df['ID'].apply(lambda x: True if x.split("/")[0] == habitat else False)]

def slice_df_reg(df, habitat):
    if habitat is None:
        return df
    return df[df['ID'].apply(lambda x: True if x.split("/")[1].split("_")[0] 
                        == habitat else False)]

def get_transformer(transform, split):
    if transform == "resize_normalize":
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        normalize_transform = transforms.Normalize(mean=mean, std=std)

        return transforms.Compose(
            [transforms.Resize((224,224)),
             transforms.ToTensor(),
             normalize_transform])

    if transform == "rgb_normalize":
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        normalize_transform = transforms.Normalize(mean=mean, std=std)

        return transforms.Compose(
            [
             transforms.ToTensor(),
             normalize_transform])

