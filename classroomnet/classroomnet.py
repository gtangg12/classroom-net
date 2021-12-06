import torch
import torch.nn as nn
import torch.nn.functional as F

from classroomnet.backbones.backbone_utils import StudentBackbone
from classroomnet.faster_rcnn.faster_rcnn import FasterRCNN

# feature_dim should be 512, in_channels_list should be [192, 96, 96, ...], out_channels (512? can probably be anything)
def create_classroom_net(num_teachers, feature_dim, statistics_list, in_channels_list, out_channels, num_classes, **kwargs):
    backbone = StudentBackbone(num_teachers, feature_dim, statistics_list, in_channels_list, out_channels)
    model = FasterRCNN(backbone, num_classes, **kwargs)

    return model