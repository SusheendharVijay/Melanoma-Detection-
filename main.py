import os 
import torch
import numpy 
import pandas as pd 
import torch.nn as nn
import pretrainedmodels

class SEResNext50_32x4d(nn.Module):
    def __init(self,pretrained="imagenet"):
        super(SEResNext50_32x4d,self).__init__()
        self.model = pretrainedmodels.__dict__["se_resnext50_32x4d"](pretrained=pretrained)
        self.out = nn.Linear(2048,1)
    
    def forward(self,image):
        






