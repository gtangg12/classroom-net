from torch.utils.data.dataloader import DataLoader
from classroomnet.classroomnet import create_classroom_net
from datalake.datalake import Datalake
from teachers.spvnas import get_unprojected_features_from_point_clouds
import cv2 
import time
import numpy as np
import torch    
import torch.nn.functional as F
import torch.optim as optim

# torch.cuda.set_enabled_lms(True)

def draw_bounding_boxes(image, bounding_boxes):
    for box in bounding_boxes:
        image = cv2.rectangle(image, (box[0], box[2]), (box[1], box[3]), (255, 0, 0), 2)
    return image


def imshow(name, image, enc='RGB'):
    image = image.astype('float32')
    image /= np.max(image)
    if enc == 'RGB':
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imshow(name, image)
    cv2.waitKey(0)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

model = create_classroom_net(2, 128, [(0, 1, 0, 1), (0, 1, 0, 1)], [78, 5], [96, 78, 5], 128, 10)
model.to(device)

optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.7, patient=8)

data = Datalake(10, ['image', 'bounding_boxes', 'object_classes', 'object_depths', 'object_class_mask', 'image_point_cloud_map'], 'datalake/data_sample')
# print(data[0])

dataloader = DataLoader(data, batch_size=2)

num_epochs = 150

for epochs in range(num_epochs):
    for data_instance, idxs, pths in dataloader:

        optimizer.zero_grad()

        image = data_instance['image']
        bounding_boxes = data_instance['bounding_boxes']
        object_classes = data_instance['object_classes']
        object_depths = data_instance['object_depths']
        mask = data_instance['object_class_mask']
        pc = data_instance['image_point_cloud_map']

        # draw_bounding_boxes(image, bounding_boxes)
        # imshow('Image', image)
        # print(object_depths)
        # print(object_classes)

        print(bounding_boxes)

        image_reshape = torch.permute(torch.from_numpy(image), (0, 3, 1, 2)) / 256
        image_reshape.to(device)

        print(image_reshape)

        targets = []

        for bbox_sample in bounding_boxes:
            bbox = torch.from_numpy(bbox_sample)
            x1 = bbox[:, 0:1]
            x2 = bbox[:, 1:2]
            y1 = bbox[:, 2:3]
            y2 = bbox[:, 3:4]
            bbox = torch.cat([x1, y1, x2, y2], dim=1).to(device)

            labels = torch.from_numpy(object_classes).to(device)
            depths = torch.from_numpy(object_depths).to(device)

            print(bbox, labels, depths)

            targets.append({'boxes': bbox, 'labels': labels, 'depths': depths})

        # model.eval()

        l, z = model(image_reshape, targets)

        feats_3d = get_unprojected_features_from_point_clouds(pc)
        distill_loss_3d = F.mse_loss(z[0], feats_3d)
        distill_loss_mask = F.mse_loss(z[1], mask)
        print('distill_loss_3d', distill_loss_3d)
        print('distill_loss_mask', distill_loss_mask)
        total_loss = distill_loss_3d + distill_loss_mask
        for k, v in l.items():
            print(k, v)
            total_loss += v
        
        total_loss.backward()
        optimizer.step()
        scheduler.step()