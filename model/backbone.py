from timm import create_model
from collections import OrderedDict

import torch.nn as nn

TIMM_AVAILABLE_MODELS = [
                         ## convnext
                         'convnext_tiny', ## 27,818,592 params
                         'convnext_base', ## 87,564,416 params
                         ## convnextv2
                         'convnextv2_tiny.fcmae', ## 27,864,960
                         'convnextv2_base.fcmae', ##  87,690,752 params
                         ## caformer
                         'caformer_s18.sail_in22k_ft_in1k_384', ## 23,236,912 params
                         'caformer_b36.sail_in22k_ft_in1k_384', ## 93,310,566 params
                         ## regnety
                         'regnety_080_tv.tv2_in1k', ## 27,749,944 params
                         'regnety_320.tv2_in1k', ## 106,852,654 params
                         ## coatnet
                         'coatnet_1_rw_224.sw_in1k', ## 28,098,162 params
                         'coatnet_3_rw_224.sw_in12k', ## 112,262,724 params
                        ]

# General builder
class Backbone(nn.Module):
    
    def __init__(self, 
                 model_name: str,
                 out_indices: list[int] = [0, 1, 2, 3],
                ):
        super(Backbone, self).__init__()
        
        if model_name in TIMM_AVAILABLE_MODELS: # Create timm models
            self.backbone = create_model(model_name, pretrained=True, features_only=True, out_indices=out_indices)
        else:
            raise Exception(f'The model name does not exist. The available models are: {TIMM_AVAILABLE_MODELS}')
        
    def forward(self, x):
        features = self.backbone(x)

        return OrderedDict(zip([f'P{i}' for i in range(len(features))], features))