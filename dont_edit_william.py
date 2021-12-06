from classroomnet.classroomnet import create_classroom_net
from datalake.datalake import Datalake
import cv2 
import time
import numpy as np

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
print(data[0])


data_instance = data[0]
image = data_instance['image']
bounding_boxes = data_instance['bounding_boxes']
object_classes = data_instance['object_classes']
object_depths = data_instance['object_depths']

draw_bounding_boxes(image, bounding_boxes)
imshow('Image', image)
print(object_depths)
print(object_classes)

# model = create_classroom_net(2, 512, [(0, 1, 0, 1), (0, 1, 0, 1)], [192, 96, 96], 512, 10)