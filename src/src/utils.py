import torch
import tqdm
import argparse
import pandas as pd
import pickle, os
import numpy as np
from torch.utils.data import sampler
import json

# ========================================================
# Sampler functions
# ========================================================
class SubsetSampler(sampler.Sampler):
    def __init__(self, data_source, indices):
        self.data_source = data_source
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)

class BalancedSampler(sampler.Sampler):
    def __init__(self, data_source, n_samples):
        self.data_source = data_source
        self.n_samples = n_samples
        self.n = len(self.data_source)
        self.nf = (self.data_source.labels!=0).sum()
        self.nb = (self.data_source.labels==0).sum()

        self.nb_ind = (self.data_source.labels==0)
        self.nf_ind = (self.data_source.labels!=0)
        
    def __iter__(self):
        p = np.ones(len(self.data_source))
        p[self.nf_ind] =  self.nf 
        p[self.nb_ind] =  self.nb
        p = p / p.sum()

        indices = np.random.choice(np.arange(self.n), 
                                   self.n_samples, 
                                   replace=False, 
                                   p=p)
        # self.data_source.labels[indices]
        return iter(indices)

    def __len__(self):
        return self.n_samples

Habitats_list = \
["7117",
"7393",
"7398",
"7426",
"7434",
"7463",
"7482",
"7490",
"7585",
"7623",
"9852",
"9862",
"9866",
"9870",
"9892",
"9894",
"9898",
"9907",
"9908"]

Habitats_dict = \
{"7117":	"Rocky Mangrove - prop roots"
,"7268":	"Sparse algal bed"
,"7393":	"Upper Mangrove – medium Rhizophora"
,"7398":	"Sandy mangrove prop roots"
,"7426":	"Complex reef"
,"7434":	"Low algal bed"
,"7463":	"Seagrass bed"
,"7482":	"Low complexity reef"
,"7490":	"Boulders"
,"7585":	"Mixed substratum mangrove - prop roots"
,"7623":	"Reef trench"
,"9852":	"Upper mangrove - tall rhizophora"
,"9862":	"Large boulder"
,"9866":	"Muddy mangrove - pneumatophores and trunk"
,"9870":	"Muddy mangrove – pneumatophores"
,"9892":	"Bare substratum"
,"9894":	"Mangrove - mixed pneumatophore prop root"
,"9898":	"Rocky mangrove - large boulder and trunk"
,"9907":	"Rock shelf"
,"9908":	"Large boulder and pneumatophores"}


try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url