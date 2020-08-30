# This file should be made to view and aggregate results
#%%
import itertools
import pprint
import argparse
import sys
import os
import pylab as plt
import pandas as pd
import sys
import numpy as np
import hashlib 
import pickle
import json
import glob

import copy

savedir_base = '/mnt/datasets/public/issam/prototypes/'\
'underwater_fish/borgy'


sys.path.append("/home/issam/Research_Ground/FishCount")
import exp_configs as cg 

EXP_GROUPS = cg.EXP_GROUPS

import torch
import pprint
from src import utils as ut
from src import datasets, models
from torch.utils.data import DataLoader
from src import mlkit
from src import wrappers
import exp_configs


savedir_base = '/mnt/datasets/public/issam/prototypes/'\
               'underwater_fish/borgy'
datadir = "/mnt/datasets/public/issam/FishCount_annotated/"


def vis_experiments(exp_dict, savedir, datadir, savedir_images):
    """Main."""
    pprint.pprint(exp_dict)
    print("exp_id: %s" % mlkit.hash_dict(exp_dict))
    print("Experiment saved in %s" % savedir)
    # stop
    # Load val set and train set
    val_set = datasets.get_dataset(dataset_name=exp_dict["dataset"], split="val",
                                   transform=exp_dict.get("transform"),
                                   datadir=datadir)
    
    vis_loader = DataLoader(val_set, sampler=ut.SubsetSampler(val_set,
                                                     indices=np.arange(10)),
                            batch_size=1)
    # Create model, opt, wrapper
    model_original = models.get_model(exp_dict["model"], exp_dict=exp_dict).cuda()
    model = wrappers.get_wrapper(exp_dict["wrapper"], model=model_original).cuda()

    score_list = mlkit.load_pkl(savedir + "/score_list.pkl")
    model.load_state_dict(torch.load(savedir + "/model_state_dict.pth"))
    # stop
    
    # stop
    # print(savedir_images)
    # stop

    model.vis_on_loader(vis_loader, 
            savedir=savedir_images)
    # stop
if __name__ == '__main__':
    exp_list = []
    exp_group_list = [
        # "clf", "reg", 
        # "loc", 
        "fisheries"
        # "seg"
        ]
    for exp_group_name in exp_group_list:
        exp_list += exp_configs.EXP_GROUPS[exp_group_name]
    
    # print(exp_list)

    # stop
    
    for exp_dict in exp_list:
        # exp_id = mlkit.hash_dict(exp_dict)
        # savefolder = savedir_list(exp_id)
        # savedir = "%s/%s" % (savedir, exp_id)
        savedir_images = (savedir_base.replace("/borgy","/fisheries") + 
                                "/%s_%s" % (exp_dict["task"],exp_dict["dataset"]))
        print(savedir_images)
        # stop
        vis_experiments(exp_dict=exp_dict, 
                 savedir=savedir_base + "/%s" % mlkit.hash_dict(exp_dict), 
                 datadir=datadir,
                 savedir_images=savedir_images)

    