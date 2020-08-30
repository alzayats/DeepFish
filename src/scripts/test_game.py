#%%
import torch
import argparse
import pandas as pd
import sys
import numpy as np
sys.path.append("/mnt/home/issam/Research_Ground/FishCount")
from torch import nn
from torch.nn import functional as F
import tqdm
import pprint
from src import trainers
from src import utils as ut
import torchvision
from src import datasets, models
from torch.utils.data import DataLoader
import configs as cg
from src.scripts import vis
from torch.utils.data.sampler import RandomSampler
import os 

# mlkit
from mlkit import checkpoint_manager as cm
from mlkit import exp_manager as em

val_key = "val_game"
tasks=("loc",)
exp_list = [{"dataset":"Fish",
            "model":"DeepFish",
            "batch_size": 1,
            "tasks":("loc"),
            "max_epochs": 1000,
            "wrapper":"DeepFishWrapper"},
           
            ]


def main(exp_dict, savedir):
    fname_score_dict = savedir + "/test_score_dict_game.json"
    if not os.path.exists(fname_score_dict):
        pprint.pprint(exp_dict)
        score_dict = {}
        dataset_dir = cg.DATADIR_BASE_DICT[exp_dict["dataset"]]["issam"]
        model_original = models.get_model(exp_dict["model"], 
                                            exp_dict=exp_dict).cuda()
        opt = torch.optim.Adam(model_original.parameters(), 
                        lr=1e-5, weight_decay=0.0005)
        model = models.get_wrapper(exp_dict["wrapper"], 
                                    model=model_original, 
                                    opt=opt).cuda()

        val_set = datasets.get_dataset(dataset_name=exp_dict["dataset"], 
                        split="val",
                                    transformer_name=exp_dict.get("transform"),
                                    dataset_dir=dataset_dir,
                                    tasks=exp_dict["tasks"])
        model.load_state_dict(torch.load(savedir + "/model.pth"))
        habitats = set([l.split("/")[1].split("_")[0] for l in val_set.loc.img_names])
        
        habitat_name = "all"
        score_dict["epoch"] = ut.load_pkl(savedir + "/score_list.pkl")[-1]["epoch"]
        print("loaded model at epoch %d" % score_dict["epoch"])
        score_dict["%s_n_samples" % (habitat_name)] = len(val_set)
        val_loader = DataLoader(val_set, shuffle=False, batch_size=exp_dict["batch_size"])

        val_dict = trainers.val_epoch(model, val_loader)
        score_dict["%s" % (habitat_name)] = val_dict[val_key]

        for habitat in habitats:
            # print(habitat)
            val_set = datasets.get_dataset(dataset_name=exp_dict["dataset"], split="val",
                                    transformer_name=exp_dict.get("transform"),
                                    dataset_dir=dataset_dir,
                                habitat=habitat,
                                tasks=exp_dict["tasks"])
            val_loader = DataLoader(val_set, shuffle=False, batch_size=exp_dict["batch_size"])
            # print(val_set.loc.img_names)
            habitat_name = cg.Habitats_dict[habitat]
            # print(val_set.loc.labels)
            # score_dict["%s_%s" % (habitat_name, model)] = func(val_set.loc.labels)
            score_dict["%s_n_samples" % (habitat_name)] = len(val_set)
            val_dict = trainers.val_epoch(model, val_loader)
            score_dict["%s" % (habitat_name)] = val_dict[val_key]

            pprint.pprint(score_dict)
        ut.save_pkl(fname_score_dict, score_dict)
    score_dict = ut.load_pkl(fname_score_dict)
    score_dict["model"] = exp_dict["model"]
    return score_dict



habitats = ["Low complexity reef"      ,                
"Sandy mangrove prop roots"  ,             
"Complex reef"    ,                         
"Seagrass bed"    ,                         
"Low algal bed"       ,                     
"Reef trench"     ,                         
"Boulders"             ,                    
"Mixed substratum mangrove - prop roots" ,   
"Rocky Mangrove - prop roots"       ,        
"Upper Mangrove – medium Rhizophora"    ,    
"Rock shelf"           ,                     
"Mangrove - mixed pneumatophore prop root" , 
"Sparse algal bed"     ,                     
"Muddy mangrove - pneumatophores and trunk" ,
"Large boulder and pneumatophores"       ,   
"Rocky mangrove - large boulder and trunk"  ,
"Bare substratum"       ,                    
"Upper mangrove - tall rhizophora"  ,        
"Large boulder"       ,                      
"Muddy mangrove – pneumatophores" ,
"all"]


if __name__ == "__main__":
    savedir_base = "/mnt/datasets/public/issam/prototypes/underwater_fish/"
    results = []

    for exp_dict in exp_list:
        exp_id = em.get_exp_id(exp_dict)
        savedir = savedir_base + "/borgy/%s/" % (exp_id)

        results += [main(exp_dict, savedir)]
    df = pd.DataFrame(results).set_index("model")
    # print(df.columns)
    display(df)

    result_list = []
    for habitat in habitats:
        if habitat == "all":
            df_filter = df[[col for col in
                            df.columns if habitat
                            == col][:1]]
        else:
            df_filter = df[[col for col in
                            df.columns if habitat
                            in col][:1]]
        # print(df_filter.to_dict())
        # qweqw
        for k, v in df_filter.to_dict().items():
            v["habitat"] = k
            result_list += [v]
    df_filter = pd.DataFrame(result_list)
    df_filter = df_filter[["habitat",  "DeepFish"]]

    df_filter = df_filter.set_index("habitat")
    display(df_filter)
