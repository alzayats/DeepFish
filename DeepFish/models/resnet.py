from torch import nn
from skimage import morphology as morph
from torchvision import models
from torch.autograd import Variable
from torch.autograd import Function
# from haven.base import base_model
import torch.nn.functional as F
import torch
# from torch import optim
# import torchvision
# # from . import lcfcn
# # from haven._toolbox import misc as ms
# import numpy as np
# import shutil
# # from src import utils as ut
# # from src import models as md
# # from models.counts2points import helpers
# from torchvision.transforms import functional as FT
# from torch.autograd import Function
from models import resfcn

class ResNet(torch.nn.Module):
    def __init__(self, n_classes=1):
        super().__init__()    
        # features 
        self.n_outputs = n_outputs = 1
        
        # backbone
        self.backbone = resfcn.ResBackbone()
        layers = list(map(int, str("100-100").split("-")))
        layers = [100352] + layers
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
        # logits_32s, logits_16s, logits_8s = self.backbone.extract_features(x)
        logits_8s, logits_16s, logits_32s = self.backbone.extract_features(x)

        # 1. EXTRACT resnet features
        x = logits_32s.view(n, -1)
       
        
        # 2. GET MLP OUTPUT
        x = self.mlp(x)
        return x 

if __name__ == '__main__':
    model = ResNet()
    x = torch.rand((8,3,224,224))
    y = model(x)
    print(y.shape)
