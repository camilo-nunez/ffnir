from dataclasses import dataclass, MISSING
from omegaconf import OmegaConf

@dataclass
class BACKBONE:
    MODEL_NAME: str = MISSING
    OUT_INDICES: list[int] = MISSING
        
@dataclass
class NECK:
#     MODEL_NAME: str = MISSING
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