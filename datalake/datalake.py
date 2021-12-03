import os
import glob
import numpy as np

import torch
from torch.utils.data import Dataset

class Datalake(Dataset):
    def __init__(self, num_data, datatypes):
        """ Pass in a list of strings of feature names

            image
            bounding_boxes
            object_classes
            object_depths

            the boxes to object classes and depths have same indexing correspond
            to same object

            William add ur feature names

            check for bugs tmrw 
        """
        self.num_data = num_data
        self.datatypes = datatypes
        self.waymo_data = \
            sorted(glob.glob('data/raw_data/*/*/*.npz')[:num_data])
        self.teacher_features = \
            sorted(glob.glob('data/teacher_features/*/*/*.npz')[:num_data])

    def __len__(self):
        return self.num_data

    def __getitem__(self, idx):
        """ """
        data_dict = np.load(self.waymo_data[idx])
        feature_dict = np.load(self.teacher_features[idx])
        ret = {}
        for datatype in datatypes:
            if datatype in data_dict:
                ret[datatype] = data_dict[datatype]
            elif datatype in feature_dict:
                ret[datatype] = feature_dict[datatype]
            else:
                raise Exception('Datalake: requested datatype not present')
        return ret
