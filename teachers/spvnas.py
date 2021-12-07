import torchsparse
import torchsparse.nn as spnn
from torchsparse import PointTensor, SparseTensor

from torchsparse.utils.collate import sparse_collate_fn
from torchsparse.utils.quantize import sparse_quantize

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
sys.path.append(str(Path(__file__).resolve().parents[0]))
sys.path.append(str(Path(__file__).resolve().parents[0] / 'external/spvnas'))

from external.spvnas.core.models.utils import point_to_voxel, voxel_to_point
from external.spvnas.model_zoo import minkunet, spvcnn, spvnas_specialized


import torch
import numpy as np

h_orig, w_orig = 1280, 1920

h_new, w_new = 256, 384

scale = 5

MODEL = spvnas_specialized('SemanticKITTI_val_SPVNAS@65GMACs')
def unproject_features(features_3d, point_cloud):
    """
    Given 3d features, shape (N_i, D), downproject to 2D features, using 
    point cloud, shape (N_i, 5)
    result: (H, W, D)
    """
    #print(point_cloud)
    h, w = int(torch.max(point_cloud[:, 3]).item()) + 1, int(torch.max(point_cloud[:, 4]).item()) + 1
    n, d = features_3d.shape
    features = torch.zeros((h_new, w_new, d), device='cuda')

    _, idx_sort = torch.sort(point_cloud[:, 1], descending=True)
    pc_ = point_cloud[idx_sort]
    features_ = features_3d[idx_sort]
    features[(pc_[:, 3]/5).type(torch.cuda.LongTensor), (pc_[:, 4]/5).type(torch.cuda.LongTensor)] = features_

    """
    for x in range(h):
        for y in range(w):
            point_cloud_with_idx = torch.cat((point_cloud, torch.arange(point_cloud.shape[0])[:, None]), dim=1)
            idxs_relevant = (5*x <= point_cloud[:, 3]) & (point_cloud[:, 3] < 5*x+5) & (5*y <= point_cloud[:, 4]) & (point_cloud[:, 4] < 5*y+5) 
            points_relevant = point_cloud_with_idx[idxs_relevant]
            if points_relevant.shape[0]:
                point_to_use = points_relevant[torch.argmin(points_relevant[:, 1]), :]
                features[x, y] = features_3d[point_to_use[5]].cpu().detach()
    """

    #batch_idx = np.tile(np.arange(b), (1, n))
    #features[point_cloud[:, 3].type(torch.LongTensor), point_cloud[:, 4].type(torch.LongTensor)] = features_3d.cpu().detach()
    return features

def preprocess_block(block):
    voxel_size = 0.05

    pc_ = np.round(block[:, :3] / voxel_size).astype(np.int32)
    pc_ -= pc_.min(0, keepdims=1)

    feat_ = block

    #visualize_point_cloud(block)

    _, inds, inverse_map = sparse_quantize(pc_,
                                            return_index=True,
                                            return_inverse=True)
    
    pc = pc_[inds]
    feat = feat_[inds]
    #labels = labels_[inds]
    lidar = SparseTensor(feat, pc)
    #labels = SparseTensor(labels, pc)
    #labels_ = SparseTensor(labels_, pc_)
    inverse_map = SparseTensor(inverse_map, pc_)

    return {
        'lidar': lidar,
        'targets': None,
        'targets_mapped': None,
        'inverse_map': inverse_map,
        'file_name': None
    }

def extract_features(model_spvnas, x):
    """Given a spvnas model and a point cloud, extract features"""
    # x: SparseTensor z: PointTensor
    z = PointTensor(x.F, x.C.float())
    x0 = point_to_voxel(x, z)

    x0 = model_spvnas.stem(x0)
    z0 = voxel_to_point(x0, z)
    z0.F = z0.F

    x1 = point_to_voxel(x0, z0)
    x1 = model_spvnas.downsample[0](x1)
    x2 = model_spvnas.downsample[1](x1)
    x3 = model_spvnas.downsample[2](x2)
    x4 = model_spvnas.downsample[3](x3)

    # point transform 32 to 256
    z1 = voxel_to_point(x4, z0)
    z1.F = z1.F + model_spvnas.point_transforms[0](z0.F)

    y1 = point_to_voxel(x4, z1)
    y1.F = model_spvnas.dropout(y1.feats)
    y1 = model_spvnas.upsample[0].transition(y1)
    y1 = torchsparse.cat([y1, x3])
    y1 = model_spvnas.upsample[0].feature(y1)

    y2 = model_spvnas.upsample[1].transition(y1)
    y2 = torchsparse.cat([y2, x2])
    y2 = model_spvnas.upsample[1].feature(y2)
    # point transform 256 to 128
    z2 = voxel_to_point(y2, z1)
    z2.F = z2.F + model_spvnas.point_transforms[1](z1.F)

    y3 = point_to_voxel(y2, z2)
    y3.F = model_spvnas.dropout(y3.F)
    y3 = model_spvnas.upsample[2].transition(y3)
    y3 = torchsparse.cat([y3, x1])
    y3 = model_spvnas.upsample[2].feature(y3)

    y4 = model_spvnas.upsample[3].transition(y3)
    y4 = torchsparse.cat([y4, x0])
    y4 = model_spvnas.upsample[3].feature(y4)
    z3 = voxel_to_point(y4, z2)
    z3.F = z3.F + model_spvnas.point_transforms[2](z2.F)

    return z3

def get_projected_features_from_point_clouds(pcs):
    """Abstracts away other parts of code. Get the unprojected features from the raw point clouds"""

    # convert to sparse tensor
    preprocessed_inputs = []

    for i in range(len(pcs)):
        preprocessed_inputs.append(preprocess_block(
            np.hstack(( pcs[i][:, :3], np.full((pcs[i].shape[0], 1), 0, dtype=np.float32) )) 
            ))

    # collate
    inputs = sparse_collate_fn(preprocessed_inputs)

    # extract features
    features = extract_features(MODEL, inputs['lidar'].cuda()).F

    #print(features.shape, inputs['lidar'].F.shape, inputs['inverse_map'].C.shape)
    # unproject features
    unprojected_features = []
    for idx in range(inputs['inverse_map'].C[:, -1].max() + 1):
        # get the inverse indices
        inv_idx = inputs['inverse_map'].F[inputs['inverse_map'].C[:, -1] == idx]
        # get the features
        feat = features[inputs['lidar'].C[:, -1] == idx][inv_idx]
        # unproject
        unprojected_features.append(unproject_features(feat, pcs[idx][inv_idx]))
    
    return torch.permute(torch.stack(unprojected_features, dim=0), (0, 3, 1, 2))

    
    
