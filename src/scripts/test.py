import torch
import argparse
import pandas as pd
import sys

from torch import nn
from torch.nn import functional as F
import tqdm
import pprint
from src import trainers
from src import utils as ut
import torchvision
from src import datasets, models
from torch.utils.data import DataLoader
import configs
from src.scripts import vis
from torch.utils.data.sampler import RandomSampler

# mlkit
from mlkit import checkpoint_manager as cm
from mlkit import exp_manager as em


def test(exp_dict, savedir, args_dict):
    """Main."""
    ac_stp = exp_dict["batch_size"] 
    pprint.pprint(exp_dict)
    print("Experiment saved in %s" % savedir)

    dataset_dir = configs.DATADIR_BASE_DICT[exp_dict["dataset"]][args_dict["userid"]]

    val_set = datasets.get_dataset(dataset_name=exp_dict["dataset"], split="val",
                                transformer_name=exp_dict["transformer"],
                                dataset_dir=dataset_dir)

    val_loader = DataLoader(val_set, shuffle=False, batch_size=exp_dict["batch_size"])


    # create model
    s_epoch = 0
    model_original = models.get_model(exp_dict["model"], exp_dict=exp_dict).cuda()
    opt = torch.optim.Adam(model_original.parameters(), lr=1e-5, weight_decay=0.0005)

    model = models.get_wrapper(exp_dict["wrapper"], model=model_original, opt=opt).cuda()

    score_list = []

    chk = cm.CheckpointManager(exp_dict, savedir=savedir)
    # resume experiment
    save_dict = chk.load(["model.pth", "score_list.pkl"])
    model.load_state_dict(save_dict["model.pth"])
    score_list = save_dict["score_list.pkl"]
    epoch = len(score_list)

    assert epoch == exp_dict["max_epochs"]

    print("Loaded model at epoch: %d" % epoch)


    habitats = set([l.split("_")[0] for l in val_set.img_names])
    score_dict = {}
    # Get test scores for all
    val_dict = trainers.val_epoch(model, val_loader)
    habitat_name = "all"
    for k,v in val_dict.items():
        score_dict["%s_%s" % (habitat_name, k)] = v
        score_dict["%s_n_samples" % (habitat_name)] = len(val_set)
    # Get test scores per habitat
    vis_loader = DataLoader(val_set,
                            sampler=ut.SubsetSampler(val_set,
                                                     indices=[0, 1, 2]),
                            batch_size=1)
    trainers.vis_epoch(model, vis_loader, savedir=savedir + "/vis_test/")
    for habitat in habitats:
        val_set = datasets.get_dataset(dataset_name=exp_dict["dataset"], split="val",
                                transformer_name=exp_dict["transformer"],
                                dataset_dir=dataset_dir,
                                habitat=habitat)

        val_loader = DataLoader(val_set, shuffle=False, batch_size=exp_dict["batch_size"])
        val_dict = trainers.val_epoch(model, val_loader)
        for k,v in val_dict.items():
            habitat_name = configs.Habitats_dict[habitat]
            score_dict["%s_%s" % (habitat_name, k)] = v
            score_dict["%s_n_samples" % (habitat_name)] = len(val_set)

        pprint.pprint(score_dict)
    ut.save_pkl(savedir + "/test_score_dict.json", score_dict)
    

if __name__ == "__main__":
    # parsing
    parser = argparse.ArgumentParser()

    parser.add_argument('-e', '--exp_config_name', default="baseline")
    parser.add_argument('-s', '--savedir')
    
    parser.add_argument('-vsf', '--vis_flag', default=0, type=int)
    parser.add_argument('-vf', '--val_flag', default=1, type=int)
    parser.add_argument('-u', '--userid', default="issam")

    args_dict = vars(parser.parse_args())
    pprint.pprint(args_dict)

    # specify project directory
    savedir_base = configs.SAVEDIR_BASE_DICT[args_dict["userid"]]

    # =================================================================
    # Run experiments                
    exp_list = em.cartesian_exp_config(configs.EXP_GROUPS[args_dict["exp_config_name"]])
    for exp_dict in exp_list:
        exp_id = em.get_exp_id(exp_dict)
        savedir = savedir_base + "/borgy/%s/" % (exp_id)
        em.save_exp_dict(exp_dict, savedir)

        # delete exp folder
        exp_dict = em.load_exp_dict(savedir)

        test(exp_dict, savedir=savedir, args_dict=args_dict)
