from torch.utils.data import random_split, DataLoader
from classroomnet.classroomnet import create_classroom_net
from datalake.datalake import Datalake
from teachers.spvnas import get_projected_features_from_point_clouds
import cv2 
import time
import numpy as np
import torch    
import torch.nn.functional as F
import torch.optim as optim
import time
import pickle
from tqdm import tqdm
from advertorch.attacks import L2PGDAttack


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

model = create_classroom_net(2, 96, [(0, 1, 0, 1), (0, 1, 0, 1)], [76, 5], [48, 48, 48], 64, 10)
model.to(device)
model.load_state_dict(torch.load('nonablated.pt'))

data = Datalake(50000, ['image', 'bounding_boxes', 'object_classes', 'object_depths', 'object_class_mask', 'image_point_cloud_map_unscaled'], 'datalake/data')
# print(data[0])

loader = DataLoader(data, batch_size=1, collate_fn=Datalake.collate_fn, shuffle=True)

for data_batch in loader:

    data_dict = [x for x, _, _ in data_batch if len(x['bounding_boxes'].shape) == 2]
    data_instance = {
        'image': torch.stack([torch.tensor(x['image']) for x in data_dict], dim=0),
        'bounding_boxes': [x['bounding_boxes'] for x in data_dict],
        'object_classes': [x['object_classes'] for x in data_dict],
        'object_depths': [x['object_depths'] for x in data_dict],
        'object_class_mask': torch.stack([torch.tensor(x['object_class_mask']) for x in data_dict], dim=0),
        'image_point_cloud_map_unscaled': [x['image_point_cloud_map_unscaled'] for x in data_dict],
    }


    image = data_instance['image']
    bounding_boxes = data_instance['bounding_boxes']
    object_classes = data_instance['object_classes']
    object_depths = data_instance['object_depths']
    mask = data_instance['object_class_mask']
    pc = data_instance['image_point_cloud_map_unscaled']

    image_reshape = torch.permute(image, (0, 3, 1, 2)) / 256
    image_reshape.to(device)

    targets = []
    for idx, bbox_sample in enumerate(bounding_boxes):
        bbox = torch.tensor(bbox_sample)
        x1 = bbox[:, 0:1]
        x2 = bbox[:, 1:2]
        y1 = bbox[:, 2:3]
        y2 = bbox[:, 3:4]
        bbox = torch.cat([x1, y1, x2, y2], dim=1).to(device)

        labels = torch.tensor(object_classes[idx]).to(device)
        depths = torch.tensor(object_depths[idx]).to(device)

        #print(bbox, labels, depths)

        keep_idxs = (x2[:, 0]-x1[:, 0]>4) & (y2[:, 0]-y1[:, 0]>4)
        bbox, labels, depths = bbox[keep_idxs], labels[keep_idxs], depths[keep_idxs]
        
        # if (x2 - x1 > 4) and (y2 - y1 > 4):
        targets.append({'boxes': bbox, 'labels': labels, 'depths': depths})

    l, z = model(image_reshape, targets)

    pc = [torch.flip(torch.tensor(pc_), (1,)) for pc_ in pc]

    feats_3d = get_projected_features_from_point_clouds(pc).detach()

    distill_loss_3d = F.mse_loss(z[0], feats_3d)
    distill_loss_mask = F.mse_loss(z[1], mask.cuda().detach())

    l['distill 3d'] = distill_loss_3d
    l['distill mask'] = distill_loss_mask

    for k, v in l.items():
        attack = L2PGDAttack(model, epsilon=8.0, eps_iter=0.1, nb_iter=50,    
                rand_init=False, targeted=True, clip_min = -3.0, clip_max = 3.0)
        adv_image = attack.perturb(image_reshape, targets)

