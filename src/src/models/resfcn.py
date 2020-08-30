import torch 
import torchvision

from torch import nn


class ResBackbone(torch.nn.Module):
    
    def __init__(self):
        super().__init__()
        
        resnet50_32s = torchvision.models.resnet50(pretrained=True)
        
        # Create a linear layer -- we don't need logits in this case
        resnet50_32s.fc = torch.nn.Sequential()
        
        self.resnet50_32s = resnet50_32s


        resnet_block_expansion_rate = resnet50_32s.layer1[0].expansion

        # # FREEZE BATCH NORMS
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.requires_grad = False
                m.bias.requires_grad = False

    
    def extract_features(self, x_input):
        self.resnet50_32s.eval()
        x = self.resnet50_32s.conv1(x_input)
        x = self.resnet50_32s.bn1(x)
        x = self.resnet50_32s.relu(x)
        x = self.resnet50_32s.maxpool(x)

        x = self.resnet50_32s.layer1(x)
        
        x_8s = self.resnet50_32s.layer2(x)
        x_16s = self.resnet50_32s.layer3(x_8s)
        x_32s = self.resnet50_32s.layer4(x_16s)

        return x_8s, x_16s, x_32s
