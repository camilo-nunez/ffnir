import torch
import torch.nn as nn
import torch.nn.functional as F

from model.backbone import Backbone
from model.neck import LGFFN

from collections import OrderedDict
from omegaconf.dictconfig import DictConfig

class GeM(nn.Module):
    ### Original code extracted from:
    # Title: CNN Image Retrieval in PyTorch: Training and evaluating CNNs for Image Retrieval in PyTorch
    # Retrieved Date: feb-2024
    # Availability: https://github.com/filipradenovic/cnnimageretrieval-pytorch/blob/master/cirtorch/layers/pooling.py
    ###
    def __init__(self,
                 p: float,
                 eps: float):
        super(GeM,self).__init__()
        self.p = nn.parameter.Parameter(torch.ones(1)*p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p, eps):
        return F.adaptive_avg_pool2d(x.clamp(min=eps).pow(p), 1).pow(1./p)

class MiniHead(nn.Module):
    def __init__(self, 
                 base_config: DictConfig):
        super().__init__()
        
        self._BASECONFIG = base_config
        
        self.gem = GeM(p=self._BASECONFIG.MODEL.HEAD.P, eps=self._BASECONFIG.MODEL.HEAD.EPS)
        self.bn = nn.BatchNorm1d(self._BASECONFIG.MODEL.NECK.NUM_CHANNELS)
        self.do = nn.Dropout(p=0.6)
        self.fc = nn.Linear(self._BASECONFIG.MODEL.NECK.NUM_CHANNELS, self._BASECONFIG.MODEL.NECK.NUM_CHANNELS)
        self.fc_norm = nn.Linear(self._BASECONFIG.MODEL.NECK.NUM_CHANNELS, self._BASECONFIG.MODEL.NECK.NUM_CHANNELS)
        
    def forward(self, x):
        x = self.gem(x).squeeze(2,3)
        x = self.bn(x)
        x = self.do(x)
        x = self.fc(x)
        x = F.normalize(x, p=2, dim=-1) ## l2-norm
        x = self.fc_norm(x)

        return x
        
class GeMHead(nn.Module):
    def __init__(self, 
                 base_config: DictConfig):
        
        super().__init__()
        
        self._BASECONFIG = base_config
        
        self.h1 = MiniHead(self._BASECONFIG)
        self.h2 = MiniHead(self._BASECONFIG)
        self.h3 = MiniHead(self._BASECONFIG)
        self.h4 = MiniHead(self._BASECONFIG)
    
    def forward(self, x):
        
        if isinstance(x, OrderedDict): x = list(x.values())
        p1, p2, p3, p4 = x
        
        h1 = self.h1(p1)
        h2 = self.h2(p2)
        h3 = self.h3(p3)
        h4 = self.h4(p4)

        return torch.cat([h1,h2,h3,h4], dim=1)
        
        
class LGFFEM(nn.Module):
    
    def __init__(self, 
                 base_config: DictConfig):
        
        super().__init__()
        
        self._BASECONFIG = base_config
        
        # Prepare the backbone
        self.backbone = Backbone(self._BASECONFIG.MODEL.BACKBONE.MODEL_NAME, self._BASECONFIG.MODEL.BACKBONE.OUT_INDICES)
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Prepare the neck
        self.neck = nn.Sequential(OrderedDict([(f"neck_layer_{i}",
                                                LGFFN(self._BASECONFIG.MODEL.NECK.NUM_CHANNELS,
                                                      self._BASECONFIG.MODEL.NECK.IN_CHANNELS, 
                                                      first_time=True if i == 0 else False))
                                               for i in range(self._BASECONFIG.MODEL.NECK.NUM_LAYERS)]))
        # Prepare head
        self.head = GeMHead(self._BASECONFIG)


    def forward(self, x):
        features = self.backbone(x)
        fusioned_features = self.neck(features)
        embeddings = self.head(fusioned_features)

        return embeddings