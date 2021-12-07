import os
import glob
import numpy as np

import torch
from torch.utils.data import Dataset

RAW_DATA_TYPES = ['image', 'bounding_boxes', 'object_classes', 'object_depths', 'image_point_cloud_map_unscaled']
TEACHER_DATA_TYPES = ['object_class_mask']

class Datalake(Dataset):
    def __init__(self, num_data, data_types, path):
        """ Pass in a list of strings of feature names."""
        self.num_data = num_data
        self.data_types = data_types
        self.waymo_data = \
            sorted(glob.glob('{}/raw/*/*/*.npz'.format(path))[:num_data])
        self.teacher_features = \
            sorted(glob.glob('{}/teacher_features/*/*/*.npz'.format(path))[:num_data])

        self.has_teacher_features = len(set(self.data_types) & set(TEACHER_DATA_TYPES))

    def __len__(self):
        return self.num_data

    @staticmethod
    def collate_fn(samples):
        return samples

    def __getitem__(self, idx):
        """ """
        raw = np.load(self.waymo_data[idx])
        if self.has_teacher_features:
            features = np.load(self.teacher_features[idx])

        ret = {}
        loadpath = self.waymo_data[idx]
        
        for data_type in self.data_types:
            if data_type in raw:
                ret[data_type] = raw[data_type]
            elif data_type in features:
                ret[data_type] = features[data_type]
            else:
                raise Exception('Datalake: requested datatype {} not present'.format(data_type))
        return ret, idx, loadpath
