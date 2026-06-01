"""
Visualize a trained DeepFish model on the DeepFish validation set.

This script runs a trained model over the *DeepFish dataset* validation split
and saves side-by-side visualizations (ground truth vs. prediction). It therefore
requires the DeepFish dataset directory structure.

To run predictions on your OWN images or video (without the DeepFish dataset
structure, CSVs or masks), use scripts/predict_simple.py instead.

Usage:
    python scripts/predict.py -e loc -d ${PATH_TO_DATASET} -m model.pth -sb ./predict_output -uc 1
"""

import sys, os, pprint

path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, path)
from haven import haven_utils as hu
import numpy as np
from DeepFish import datasets, models, wrappers
import argparse
from tqdm.auto import tqdm

from PIL import Image
import matplotlib.pyplot as plt

import torch
from torch.backends import cudnn
from torch import nn
import torchvision.transforms as T
import exp_configs

cudnn.benchmark = True

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--datadir',
                        type=str, default='/mnt/public/datasets/DeepFish')
    parser.add_argument("-e", "--exp_config", default='loc')
    parser.add_argument('-m', '--model_path', required=True,
                        help='Path to the trained model checkpoint (.pth file).')
    parser.add_argument('-sb', '--savedir_base', default='./predict_output',
                        help='Base directory where the visualizations will be saved.')
    parser.add_argument("-uc", "--use_cuda", type=int, default=0)
    args = parser.parse_args()

    device = torch.device('cuda' if args.use_cuda and torch.cuda.is_available() else 'cpu')
    print('Running on device: %s' % device)

    exp_dict = exp_configs.EXP_GROUPS[args.exp_config][0]

    val_set = datasets.get_dataset(dataset_name=exp_dict["dataset"], split="val",
                                   transform=exp_dict.get("transform"),
                                   datadir=args.datadir)

    # Load the trained model. The checkpoint may be a full wrapper, a raw model,
    # or a state_dict, so handle each case.
    loaded = torch.load(args.model_path, map_location=device)

    if isinstance(loaded, dict):
        # state_dict of the wrapper (what trainval.py saves via model.state_dict()):
        # rebuild the wrapper and load the weights into it (keys are prefixed "model.").
        model_original = models.get_model(exp_dict["model"], exp_dict=exp_dict)
        model = wrappers.get_wrapper(exp_dict["wrapper"], model=model_original).to(device)
        model.load_state_dict(loaded)
    elif hasattr(loaded, 'model'):
        # already a wrapper
        model = loaded.to(device)
    else:
        # a raw model
        model = wrappers.get_wrapper(exp_dict["wrapper"], model=loaded).to(device)

    # Create DataLoader
    vis_loader = torch.utils.data.DataLoader(val_set, shuffle=False, batch_size=1)

    # Visualize on loader
    model.vis_on_loader(vis_loader, savedir=os.path.join(args.savedir_base, "images"))
    print("Saved visualizations to %s" % os.path.join(args.savedir_base, "images"))
