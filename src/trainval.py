import torch
import argparse
import pandas as pd
import sys
import os
from torch import nn
from torch.nn import functional as F
import tqdm
import pprint
from src import utils as ut
import torchvision

from src import mlkit
from src import datasets, models
from torch.utils.data import DataLoader
import exp_configs
from torch.utils.data.sampler import RandomSampler
from src import mlkit
from src import wrappers


def trainval(exp_dict, savedir, datadir, val_flag=True, vis_flag=True):
    """Main."""
    pprint.pprint(exp_dict)
    print("Experiment saved in %s" % savedir)

    # Load val set and train set
    val_set = datasets.get_dataset(dataset_name=exp_dict["dataset"], split="val",
                                   transform=exp_dict.get("transform"),
                                   datadir=datadir)
    train_set = datasets.get_dataset(dataset_name=exp_dict["dataset"],
                                     split="train", 
                                     transform=exp_dict.get("transform"),
                                     datadir=datadir)
    
    # Load train loader, val loader, and vis loader
    train_loader = DataLoader(train_set, 
                            sampler=RandomSampler(train_set,
                            replacement=True, num_samples=max(min(500, 
                                                            len(train_set)), 
                                                            len(val_set))),
                            batch_size=exp_dict["batch_size"])

    val_loader = DataLoader(val_set, shuffle=False, batch_size=exp_dict["batch_size"])
    vis_loader = DataLoader(val_set, sampler=ut.SubsetSampler(train_set,
                                                     indices=[0, 1, 2]),
                            batch_size=1)

    # Create model, opt, wrapper
    model_original = models.get_model(exp_dict["model"], exp_dict=exp_dict).cuda()
    opt = torch.optim.Adam(model_original.parameters(), 
                        lr=1e-5, weight_decay=0.0005)

    model = wrappers.get_wrapper(exp_dict["wrapper"], model=model_original, opt=opt).cuda()

    score_list = []

    # Restart or Resume from last saved state_dict
    if (not os.path.exists(savedir + "/run_dict.pkl") or 
        not os.path.exists(savedir + "/score_list.pkl")):
        mlkit.save_pkl(savedir + "/run_dict.pkl", {"running":1})
        score_list = []
        s_epoch = 0
    else:
        score_list = mlkit.load_pkl(savedir + "/score_list.pkl")
        model.load_state_dict(torch.load(savedir + "/model_state_dict.pth"))
        opt.load_state_dict(torch.load(savedir + "/opt_state_dict.pth"))
        s_epoch = score_list[-1]["epoch"] + 1

    # Run training and validation
    for epoch in range(s_epoch, exp_dict["max_epoch"]):
        score_dict = {"epoch": epoch}

        # visualize
        if vis_flag:
            model.vis_on_loader(vis_loader, savedir=savedir+"/images/")

        # validate
        if val_flag:
            score_dict.update(model.val_on_loader(val_loader))
        
        # train
        score_dict.update(model.train_on_loader(train_loader))

        # Add score_dict to score_list
        score_list += [score_dict]

        # Report and save
        print(pd.DataFrame(score_list).tail())
        mlkit.save_pkl(savedir + "/score_list.pkl", score_list)
        mlkit.torch_save(savedir + "/model_state_dict.pth", model.state_dict())
        mlkit.torch_save(savedir + "/opt_state_dict.pth", opt.state_dict())
        print("Saved: %s" % savedir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-e', '--exp_group_list', nargs="+")
    parser.add_argument('-sb', '--savedir_base', default="/mnt/datasets/public/issam/prototypes/underwater_fish/non_borgy/")
    parser.add_argument('-d', '--datadir', default="/mnt/datasets/public/issam/FishCount_annotated/")
    parser.add_argument("-r", "--reset",  default=0, type=int)
    parser.add_argument("-ei", "--exp_id", default=None)
    parser.add_argument("-vf", "--val_flag", default=1, type=int)
    parser.add_argument("-vs", "--vis_flag", default=1, type=int)
   
    args = parser.parse_args()
    exp_list = []
    for exp_group_name in args.exp_group_list:
        exp_list += exp_configs.EXP_GROUPS[exp_group_name]

    # loop over experiments
    for exp_dict in exp_list:
        exp_id = mlkit.hash_dict(exp_dict)

        if args.exp_id is not None and args.exp_id != exp_id:
            continue
        
        savedir = args.savedir_base + "/%s/" % exp_id
        os.makedirs(savedir, exist_ok=True)
        mlkit.save_json(savedir+"/exp_dict.json", exp_dict)

        # check if experiment exists
        if args.reset:
            if os.path.exists(savedir + "/score_list.pkl"):
                os.remove(savedir + "/score_list.pkl")
            if os.path.exists(savedir + "/run_dict.pkl"):
                os.remove(savedir + "/run_dict.pkl")

        # do trainval
        trainval(exp_dict=exp_dict, 
                 savedir=savedir, 
                 datadir=args.datadir, 
                 val_flag=args.val_flag,
                 vis_flag=args.vis_flag)

