from managers import checkpoint_manager as cm
from managers import exp_manager as em

from torch.utils.data.sampler import SubsetRandomSampler
import argparse
import utils as ut
import pprint
import tqdm
import sys, getpass
import numpy as np
import datasets, models
import torch
import os
import exp_dicts


EXP_CONFIG_naive_baselines = {

            "baseline": {
                   "dataset": ["TinyTrancos"],
                    "model":["Max", "Min", "Mean", "Always0", "Always1"],
                   "metric": ["MAE"],
                   "batch_size": [1],
                   "batch_size_val": [1],
               },
            "fish": {"dataset": ["FishCount"],
                    "model":["Max", "Min", "Mean", "Always0", "Always1"],
                     "metric": ["MAE"],
                     "batch_size": [1],
                     "batch_size_val": [1]},
}
def main(exp_dict, savedir, path_base):
    """Main."""

    # val set
    train_set = datasets.get_dataset(dataset_name=exp_dict["dataset"], split="train", exp_dict=exp_dict, path_base=path_base)
    train_loader = torch.utils.data.DataLoader(train_set, shuffle=False, batch_size=1)

    # val set
    val_set = datasets.get_dataset(dataset_name=exp_dict["dataset"], split="val", exp_dict=exp_dict, path_base=path_base)
    val_loader = torch.utils.data.DataLoader(val_set, shuffle=False, batch_size=1)

    model = MODELS_DICT[exp_dict["model"]]()

    # Train
    if hasattr(model, "train_step"):
        pbar = tqdm.tqdm(total=len(train_loader))
        for i, batch in enumerate(train_loader):
            float(model.train_step(batch))

            pbar.set_description("Training")
            pbar.update(1)

        pbar.close()

    # Validate
    ae = 0.
    n_samples = 0.

    pbar = tqdm.tqdm(total=len(val_loader))
    for i, batch in enumerate(val_loader):
        pred_count = model.predict(batch, method="counts")

        ae += abs(batch["counts"].cpu().numpy().ravel() - pred_count.ravel()).sum()
        n_samples += batch["counts"].shape[0]

        pbar.set_description("Val mae: %.4f" % (ae / n_samples))
        pbar.update(1)

    pbar.close()

    result_dict = {}
    result_dict["mae"] = ae / n_samples
    result_dict["n_train"] = len(train_loader.dataset)
    result_dict["n_val"] = len(val_loader.dataset)

    return result_dict

# ==============================================
# naive models
class Always0:
    def __init__(self):
        pass

    def predict(self, batch, **options):
        return np.array([0.])

class Always1:
    def __init__(self):
        pass

    def predict(self, batch, **options):
        return np.array([1.])

class Mean:
    def __init__(self):
        self.sum_count = 0.
        self.n_samples = 0.

    def train_step(self, batch):
        self.sum_count += batch["counts"].item()
        self.n_samples += batch["counts"].shape[0]

        return -1

    def predict(self, batch, **options):
        mean_count = self.sum_count / self.n_samples
        return np.array([mean_count])

class Max:
    def __init__(self):
        self.max_count = -1.

    def train_step(self, batch):
        self.max_count = max(self.max_count, batch["counts"].item())

        return -1

    def predict(self, batch, **options):
        return np.array([self.max_count])

class Min:
    def __init__(self):
        self.min_count = np.inf

    def train_step(self, batch):
        self.min_count = min(self.min_count, batch["counts"].item())

        return -1

    def predict(self, batch, **options):
        return np.array([self.min_count])


MODELS_DICT = {"Always0":Always0, "Always1":Always1,
               "Mean":Mean, "Max":Max, "Min":Min}

if __name__ == "__main__":
    # parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--exp_config_name', default="baseline")
    parser.add_argument('-r', '--reset', default=0, type=int)
    parser.add_argument('-u', '--userid', default="issam")
    args_dict = vars(parser.parse_args())

    # specify project directory
    EXP_CONFIG = exp_dicts.EXP_CONFIG_naive_baselines
    datadir_base = exp_dicts.DATADIR_BASE_DICT[args_dict["exp_config_name"]][args_dict["userid"]]
    savedir_base = exp_dicts.SAVEDIR_BASE_DICT[args_dict["userid"]]

    exp_list = em.cartesian_exp_config(EXP_CONFIG[args_dict["exp_config_name"]])
    results_dict = {}

    # loop over the baseline experiments
    for exp_dict in exp_list:
        print("\n")
        exp_id = em.get_exp_id(exp_dict)
        savedir = savedir_base + "/%s/%s/" % (args_dict["exp_config_name"],exp_id)
        em.save_exp_dict(exp_dict, savedir)

        model_name = exp_dict["model"]
        fname = savedir + "/%s.pkl" % model_name

        if os.path.exists(fname) and not args_dict["reset"]:
            print("Exists: %s" %fname)
            result_dict = ut.load_pkl(fname)
        else:
            print("Running: %s" % fname)
            result_dict =  main(exp_dict=exp_dict, savedir=savedir, path_base=datadir_base)
            ut.save_pkl(fname, result_dict)

        results_dict["%s" % (exp_dict["model"])] = result_dict
        pprint.pprint(results_dict)



