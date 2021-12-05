import os
import glob
import numpy as np

import torch
from torch.utils.data import Dataset

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

    def __len__(self):
        return self.num_data

    def __getitem__(self, idx):
        """ """
        raw = np.load(self.waymo_data[idx])
        features = np.load(self.teacher_features[idx])
        ret = {}
        for data_type in data_types:
            if data_type in raw:
                ret[data_type] = raw[data_type]
            elif datatype in features:
                ret[data_type] = features[data_type]
            else:
                raise Exception('Datalake: requested datatype not present')
        return ret
