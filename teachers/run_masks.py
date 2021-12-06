import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from teachers.mask_rcnn import images_to_masks
from datalake.datalake import Datalake
import torch
import os

data = Datalake(100, ['image'], Path(__file__).parent.parent/'datalake/data_sample')
dataloader = DataLoader(data, batch_size=16)

for X, y, pths in dataloader:
    #print((X['image']/255).shape)
    masks = images_to_masks(X['image'].permute((0,3,1,2)).cuda()/255)

    for mask, idx, loadpth in zip(masks, y, pths):
        savepth = loadpth[:-4].replace('/raw/','/masks/')
        os.makedirs(savepth, exist_ok=True)
        np.save(f"{savepth}.npy", mask)