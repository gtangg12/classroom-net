import os
import glob
import numpy as np

import torch
from torch.utils.data import Dataset

RAW_DATA_TYPES = ['image, bounding_boxes, object_classes, object_depths, image_point_cloud_map']
TEACHER_DATA_TYPES = []

class Datalake(Dataset):
    def __init__(self, num_data, data_types, path):
        """ Pass in a list of strings of feature names

            image
            bounding_boxes
            object_classes
            object_depths
            image_point_cloud_map

            the boxes to object classes and depths have same indexing correspond
            to same object

            William add ur feature names

            check for bugs tmrw
        """
        self.num_data = num_data
        self.data_types = data_types
        self.waymo_data = \
            sorted(glob.glob('{}/raw/*/*/*.npz'.format(path))[:num_data])
        self.teacher_features = \
            sorted(glob.glob('{}/teacher_features/*/*/*.npz'.format(path))[:num_data])

        self.has_teacher_features = len(set(self.data_types) & set(RAW_DATA_TYPES))

    def __len__(self):
        return self.num_data

    def __getitem__(self, idx):
        """ """
        raw = np.load(self.waymo_data[idx])
        if self.has_teacher_features:
            features = np.load(self.teacher_features[idx])

        ret = {}
        for data_type in self.data_types:
            if data_type in raw:
                ret[data_type] = raw[data_type]
            elif data_type in self.features:
                ret[data_type] = features[data_type]
            else:
                raise Exception('Datalake: requested datatype {} not present'.format(data_type))
        return ret, idx
