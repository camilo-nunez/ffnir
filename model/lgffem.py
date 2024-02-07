import torch
import torch.nn as nn
import torch.nn.functional as F

from model.backbone import Backbone
from model.neck import LGFFN

from collections import OrderedDict
from omegaconf.dictconfig import DictConfig

## https://github.com/filipradenovic/cnnimageretrieval-pytorch/blob/master/cirtorch/layers/pooling.py
class GeM(nn.Module):
    def __init__(self,
                 p: float,
                 eps: float):
        super(GeM,self).__init__()
        self.p = nn.parameter.Parameter(torch.ones(1)*p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p, eps):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)
        # return F.lp_pool2d(F.threshold(x, eps, eps), p, (x.size(-2), x.size(-1))) # alternative
        
class GeMHead(nn.Module):
    def __init__(self, 
                 base_config: DictConfig):
        
        super().__init__()
        
        self._BASECONFIG = base_config
        
        self.h1 = GeM(p=self._BASECONFIG.MODEL.HEAD.P, eps=self._BASECONFIG.MODEL.HEAD.EPS)
        self.h2 = GeM(p=self._BASECONFIG.MODEL.HEAD.P, eps=self._BASECONFIG.MODEL.HEAD.EPS)
        self.h3 = GeM(p=self._BASECONFIG.MODEL.HEAD.P, eps=self._BASECONFIG.MODEL.HEAD.EPS)
        self.h4 = GeM(p=self._BASECONFIG.MODEL.HEAD.P, eps=self._BASECONFIG.MODEL.HEAD.EPS)
        
        self.h1_fc = nn.Linear(self._BASECONFIG.MODEL.NECK.NUM_CHANNELS, self._BASECONFIG.MODEL.NECK.NUM_CHANNELS)
        self.h2_fc = nn.Linear(self._BASECONFIG.MODEL.NECK.NUM_CHANNELS, self._BASECONFIG.MODEL.NECK.NUM_CHANNELS)
        self.h3_fc = nn.Linear(self._BASECONFIG.MODEL.NECK.NUM_CHANNELS, self._BASECONFIG.MODEL.NECK.NUM_CHANNELS)
        self.h4_fc = nn.Linear(self._BASECONFIG.MODEL.NECK.NUM_CHANNELS, self._BASECONFIG.MODEL.NECK.NUM_CHANNELS)
    
    def forward(self, x):
        
        if isinstance(x, OrderedDict): x = list(x.values())
        p1, p2, p3, p4 = x
        
        h1 = self.h1(p1).squeeze(0,2,3)
        h1 = F.normalize(h1, p=2, dim=-1)
        h1 = self.h1_fc(h1)
        
        h2 = self.h2(p2).squeeze(0,2,3)
        h2 = F.normalize(h2, p=2, dim=-1)
        h2 = self.h2_fc(h2)
        
        h3 = self.h3(p3).squeeze(0,2,3)
        h3 = F.normalize(h3, p=2, dim=-1)
        h3 = self.h3_fc(h3)
        
        h4 = self.h4(p4).squeeze(0,2,3)
        h4 = F.normalize(h4, p=2, dim=-1)
        h4 = self.h4_fc(h4)

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