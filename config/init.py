from dataclasses import dataclass, MISSING
from omegaconf import OmegaConf

@dataclass
class BACKBONE:
    MODEL_NAME: str = MISSING
    OUT_INDICES: list[int] = MISSING
        
@dataclass
class NECK:
    IN_CHANNELS: list[int] = MISSING
    NUM_CHANNELS: int = MISSING
    NUM_LAYERS: int = MISSING
        
@dataclass
class HEAD:
    P: float = MISSING
    EPS: float = MISSING

def default_config():
    # Main Config
    _C = OmegaConf.create()

    ## Model config
    _C.MODEL = OmegaConf.create()
    _C.MODEL.BACKBONE = OmegaConf.structured(BACKBONE)
    _C.MODEL.NECK = OmegaConf.structured(NECK)
    _C.MODEL.HEAD = OmegaConf.structured(HEAD)
    
    return _C

## CONFIG FOR TRAIN_IN1K.py
def create_train_in1k_config(args):
    model_backbone_conf = OmegaConf.load(args.cfg_model_backbone)
    model_neck_conf = OmegaConf.load(args.cfg_model_neck)
    model_head_conf = OmegaConf.load(args.cfg_model_head)
    
    base_config = default_config()
    
    base_config.MODEL = OmegaConf.merge(base_config.MODEL, model_backbone_conf, model_neck_conf, model_head_conf)

    return base_config

## CONFIG FOR TRAIN_IN1K.py
def create_baseconfig_from_checkpoint(checkpoint: dict):
 
    model_backbone_conf = OmegaConf.load(checkpoint['fn_cfg_model_backbone'])
    model_neck_conf = OmegaConf.load(checkpoint['fn_cfg_model_neck'])
    model_head_conf = OmegaConf.load(checkpoint['fn_cfg_model_head'])
    
    base_config = default_config()
    
    base_config.MODEL = OmegaConf.merge(base_config.MODEL, model_backbone_conf, model_neck_conf, model_head_conf)

    return base_config