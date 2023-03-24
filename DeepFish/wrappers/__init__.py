import torch
import tqdm
import argparse
import pandas as pd
import pickle, os
import numpy as np
from . import clf_wrapper, reg_wrapper, loc_wrapper, seg_wrapper


def get_wrapper(wrapper_name, model, opt=None):
    if wrapper_name == "clf_wrapper":
        return clf_wrapper.ClfWrapper(model, opt)

    if wrapper_name == "reg_wrapper":
        return reg_wrapper.RegWrapper(model, opt)

    if wrapper_name == "loc_wrapper":
        return loc_wrapper.LocWrapper(model, opt)

    if wrapper_name == "seg_wrapper":
        return seg_wrapper.SegWrapper(model, opt)

# ===============================================
# Trainers
def train_on_loader(model, train_loader):
    model.train()

    n_batches = len(train_loader)
    train_monitor = TrainMonitor()
    print('Training')
    for e in range(1):
        for i, batch in enumerate(tqdm.tqdm(train_loader)):
            score_dict = model.train_on_batch(batch)
            
            train_monitor.add(score_dict)
        
    return train_monitor.get_avg_score()

@torch.no_grad()
def val_on_loader(model, val_loader, val_monitor):
    model.eval()

    n_batches = len(val_loader)
    print('Validating')
    for i, batch in enumerate(tqdm.tqdm(val_loader)):
        # score, gt_points, blobs = model.val_on_batch(batch)
        score = model.val_on_batch(batch)
        val_monitor.add(score)
        # game_monitor.add_batch(gt_points, blobs)

    # return val_monitor.get_avg_score(game_monitor.get_score_dict())
    return val_monitor.get_avg_score()


@torch.no_grad()
def vis_on_loader(model, vis_loader, savedir):
    model.eval()

    n_batches = len(vis_loader)
    split = vis_loader.dataset.split
    for i, batch in enumerate(vis_loader):
        print("%d - visualizing %s image - savedir:%s" % (i, batch["meta"]["split"][0], savedir.split("/")[-2]))
        model.vis_on_batch(batch, 
        savedir_image=os.path.join(savedir, f'{i}.png'))
        

@torch.no_grad()
def test_on_loader(model, test_loader):
    model.eval()
    ae = 0.
    n_samples = 0.

    n_batches = len(test_loader)
    pbar = tqdm.tqdm(total=n_batches)
    for i, batch in enumerate(test_loader):
        pred_count = model.predict(batch, method="counts")

        ae += abs(batch["counts"].cpu().numpy().ravel() - pred_count.ravel()).sum()
        n_samples += batch["counts"].shape[0]

        pbar.set_description("TEST mae: %.4f" % (ae / n_samples))
        pbar.update(1)

    pbar.close()
    score = ae / n_samples
    print({"test_score": score, "test_mae":score})

    return {"test_score": score, "test_mae":score}


class TrainMonitor:
    def __init__(self):
        self.score_dict_sum = {}
        self.n = 0

    def add(self, score_dict):
        for k,v in score_dict.items():
            if k not in self.score_dict_sum:
                self.score_dict_sum[k] = score_dict[k]
            else:
                self.n += 1
                self.score_dict_sum[k] += score_dict[k]

    def get_avg_score(self):
        return {k:v/(self.n + 1) for k,v in self.score_dict_sum.items()}

    