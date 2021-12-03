import os
import time

frames_path = 'waymo_open_dataset/frames/training_training_0000/segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord'

import numpy as np
import tensorflow as tf

from waymo_open_dataset.utils import frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset

dataset = tf.data.TFRecordDataset(frames_path, compression_type='')

for data in dataset:
    frame = open_dataset.Frame()
    frame.ParseFromString(bytearray(data.numpy()))
    break


def get_front_image(frame):
    return tf.image.decode_jpeg(frame.images[0].image).numpy()


def get_point_cloud_front_image_mapping(frame):
    """Return Nx5 array of points and their pixel coords (col, row) in the front image."""
    (range_images, camera_projections, range_image_top_pose) = (
        frame_utils.parse_range_image_and_camera_projection(frame))

    points, cp_points = frame_utils.convert_range_image_to_point_cloud(frame,
                                                                       range_images,
                                                                       camera_projections,
                                                                       range_image_top_pose)

    # 3d points in vehicle frame.
    points_all = np.concatenate(points, axis=0)
    # camera projection corresponding to each point.
    cp_points_all = np.concatenate(cp_points, axis=0)

    points_all_tensor = tf.constant(points_all)
    cp_points_all_tensor = tf.constant(cp_points_all, dtype=tf.int32)

    # mask only points visibile by front image
    mask = tf.equal(cp_points_all_tensor[..., 0], frame.images[0].name)

    cp_points_all_tensor = tf.cast(
        tf.gather_nd(cp_points_all_tensor, tf.where(mask)), dtype=tf.float32)
    points_all_tensor = tf.gather_nd(points_all_tensor, tf.where(mask))

    return tf.concat(
        [points_all_tensor, cp_points_all_tensor[..., 1:3]], axis=-1).numpy()

print(get_point_cloud_front_image_mapping(frame))
