import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader

from mask_rcnn import images_to_masks
sys.path.append(os.path.abspath('../datalake'))
from datalake import Datalake


data = Datalake(100, ['image'], '../datalake/data')
dataloader = DataLoader(data, batch_size=16)

for X, y, paths in dataloader:
    #print((X['image']/255).shape)
    masks = images_to_masks(X['image'].permute((0,3,1,2)).cuda()/255)

    for mask, index, loadpath in zip(masks, y, paths):
        savepath = loadpath[:-4].replace('/raw/','/masks/')
        os.makedirs('/'.join(savepath.split('/')[:-1]), exist_ok=True)
        np.savez_compressed(f'{savepath}', object_class_mask=mask)
