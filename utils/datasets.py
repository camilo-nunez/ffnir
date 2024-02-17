from torch.utils.data import Dataset

import albumentations as A
from torchvision.transforms import v2

from pathlib import Path, PurePath
from natsort import natsorted
from PIL import Image

import numpy as np

class KIMIAPath24CDataset(Dataset):

    def __init__(self, root_dir, transform=None):
        
        self.KIMIA_CLASSES_IDX = dict([(f's{i}',i) for i in range(24)])
        
        self.root_dir = root_dir
        self.transform = transform
        
        img_l = []
        for i in range(0,24):
            path_i = PurePath(self.root_dir, f's{i}')
            img_l+=natsorted([path for path in Path(path_i).rglob('*.jpg')], key=str)
        self.img_l = img_l

    def __len__(self):
        return len(self.img_l)

    def __getitem__(self, idx):
        
        path_img = self.img_l[idx]
        img_class = self.KIMIA_CLASSES_IDX[str(path_img).split('/')[-2]]
        
        img = Image.open(path_img).convert('RGB')
        
        if self.transform:
            if isinstance(self.transform, A.core.composition.Compose):
                transformed = self.transform(image=np.asarray(img))
                img = transformed['image']
            elif isinstance(self.transform, v2._container.Compose):
                img = self.transform(img)
            else:
                RuntimeError("[+] The transform compose must by an Albumentations or Pytorch V2 type.!")
        
        return img, img_class