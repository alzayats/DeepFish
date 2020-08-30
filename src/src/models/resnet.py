from torch import nn
from skimage import morphology as morph
from torchvision import models
from torch.autograd import Variable
from torch.autograd import Function
# from haven.base import base_model
import torch.nn.functional as F
import torch
from torch import optim
import torchvision
from . import lcfcn
# from haven._toolbox import misc as ms
import numpy as np
import shutil
from src import utils as ut
from src import models as md
# from models.counts2points import helpers
from torchvision.transforms import functional as FT
from torch.autograd import Function
from . import resfcn

class ResNet(torch.nn.Module):
    def __init__(self, n_classes=1):
        super().__init__()    
        # features 
        self.n_outputs = n_outputs = 1
        
        # backbone
        self.backbone = resfcn.ResBackbone()
        layers = list(map(int, str("100-100").split("-")))
        layers = [401408] + layers
        n_hidden = len(layers) - 1

        layerList = []      
        for i in range(n_hidden): 
            layerList += [nn.Linear(layers[i], 
                          layers[i+1]), nn.ReLU()]
        
        layerList += [nn.Linear(layers[i+1], n_outputs)]
        self.mlp = nn.Sequential(*layerList)


        # # FREEZE BATCH NORMS
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.requires_grad = False
                m.bias.requires_grad = False


    def forward(self, x):
        n = x.shape[0]
        logits_32s, logits_16s, logits_8s = self.backbone.extract_features(x)

        # 1. EXTRACT resnet features
        x = logits_32s.view(n, -1)
       
        
        # 2. GET MLP OUTPUT
        x = self.mlp(x)
        return x 


