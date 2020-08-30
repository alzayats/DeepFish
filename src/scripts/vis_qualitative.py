import os
import sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)


import torch
import pprint
from src import utils as ut
from src import datasets, models
from torch.utils.data import DataLoader
from src import mlkit
from src import wrappers
import exp_configs

def exp_list():
    exp_list = []
    exp_group_list = ["clf", "reg", "loc", "seg"]
    for exp_group_name in exp_group_list:
        exp_list += exp_configs.EXP_GROUPS[exp_group_name]
    return exp_list

def savedir_list(exp_dict):
    task = exp_dict.get("task")
    model_name = exp_dict.get("model")
    if task == "clf":
        if model_name == "inception":
           return "/38f969357bc81f1853391f76a4f5035a/"   # inception
        else:
            return "/38f969357bc81f1853391f76a4f5035a/"   # resnet

    if task == "reg":
        if model_name == "inception":
            return "/38f969357bc81f1853391f76a4f5035a/"  # inception
        else:
            return "/38f969357bc81f1853391f76a4f5035a/"   # resnet

    if task == "loc":
        if model_name == "unet":
            return "/38f969357bc81f1853391f76a4f5035a/"   #  unet
        else:
            return "/38f969357bc81f1853391f76a4f5035a/"   # fcn8

    if task == "seg":
        if model_name == "unet":
            return "/38f969357bc81f1853391f76a4f5035a/"    #  unet
        else:
            return "/38f969357bc81f1853391f76a4f5035a/"    # fcn8


def vis_experiments(exp_dict, savedir, datadir):
    """Main."""
    pprint.pprint(exp_dict)
    print("Experiment saved in %s" % savedir)

    # Load val set and train set
    val_set = datasets.get_dataset(dataset_name=exp_dict["dataset"], split="val",
                                   transform=exp_dict.get("transform"),
                                   datadir=datadir)
    
    vis_loader = DataLoader(val_set, sampler=ut.SubsetSampler(val_set,
                                                     indices=[0, 1, 2]),
                            batch_size=1)
    # Create model, opt, wrapper
    model_original = models.get_model(exp_dict["model"], exp_dict=exp_dict).cuda()
    opt = torch.optim.Adam(model_original.parameters(),
                        lr=1e-5, weight_decay=0.0005)

    model = wrappers.get_wrapper(exp_dict["wrapper"], model=model_original, opt=opt).cuda()

    
    score_list = mlkit.load_pkl(savedir + "/score_list.pkl")
    model.load_state_dict(torch.load(savedir + "/model_state_dict.pth"))

    model.vis_on_loader(vis_loader, savedir=savedir+"/qualitative/")

if __name__ == "__main__":
    # savedir = "D:/prototypes/ed8904d21f206d7da4c6e32433ee00b3/"
    datadir = "D:/Datasets/JCU_FISH/Public_JCU_Fish/"
    exp_list = exp_list()

    for exp_id in exp_list:
        savefolder = savedir_list(exp_id)
        savedir = "D:/prototypes/%s/" % savefolder
        vis_experiments(exp_id, savedir, datadir)
