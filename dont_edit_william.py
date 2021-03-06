from classroomnet.classroomnet import create_classroom_net
from datalake.datalake import Datalake
import cv2 
import time
import numpy as np
import torch    

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


data = Datalake(1, ['image', 'bounding_boxes', 'object_classes', 'object_depths'], 'datalake/data_sample')
# print(data[0])

data_instance = data[0][0]
image = data_instance['image']
bounding_boxes = data_instance['bounding_boxes']
object_classes = data_instance['object_classes']
object_depths = data_instance['object_depths']

# draw_bounding_boxes(image, bounding_boxes)
# imshow('Image', image)
# print(object_depths)
# print(object_classes)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

print(bounding_boxes)

model = create_classroom_net(2, 128, [(0, 1, 0, 1), (0, 1, 0, 1)], [96, 96, 96], 128, 10)
model.to(device)

image_reshape = torch.reshape(torch.from_numpy(image), (1, 3, 256, 384)) / 256
image_reshape.to(device)

print(image_reshape)

bbox = torch.from_numpy(bounding_boxes)
bbox1 = bbox[:, 0:1]
bbox2 = bbox[:, 1:2]
bbox3 = bbox[:, 2:3]
bbox4 = bbox[:, 3:4]
bbox = torch.cat([bbox1, bbox3, bbox2, bbox4], dim=1).to(device)

labels = torch.from_numpy(object_classes).to(device)
depths = torch.from_numpy(object_depths).to(device)

print(bbox, labels, depths)

targets = {'boxes': bbox, 'labels': labels, 'depths': depths}

# model.eval()

l, z = model(image_reshape, [targets])

print(l, z[0].shape)

# print(p[0]['boxes'].shape, p[0]['labels'].shape, p[0]['scores'].shape, p[0]['depths'].shape, z[0].shape)