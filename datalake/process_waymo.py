import os
import time
import glob

import numpy as np
import tensorflow as tf
import cv2

from waymo_open_dataset.utils import frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset import label_pb2 as open_label


WAYMO_DATA_PATH = 'waymo_open_dataset/frames/'
DATALAKE_PATH  = 'data/'


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


class WaymoFrame:
    def __init__(self, index, frame):
        self.index = index
        self.frame = frame
        self.range_images, self.camera_projections, self.range_image_top_pose = \
            frame_utils.parse_range_image_and_camera_projection(frame)

    def process_image(self):
        """ """
        self.image = tf.image.decode_jpeg(self.frame.images[0].image).numpy()

    def process_point_cloud(self):
        """ """
        points, cp_points = \
            frame_utils.convert_range_image_to_point_cloud(self.frame,
                                                           self.range_images,
                                                           self.camera_projections,
                                                           self.range_image_top_pose)
        # 3d points in vehicle frame.
        points = np.concatenate(points, axis=0)
        # camera projection corresponding to each point.
        cp_points = np.concatenate(cp_points, axis=0)

        points_tensor = tf.constant(points)
        cp_points_tensor = tf.constant(cp_points, dtype=tf.int32)

        # The distance between lidar points and vehicle frame origin.
        depth_tensor = tf.norm(points, axis=-1, keepdims=True)

        # mask only points visibile by front image
        mask = tf.equal(cp_points_tensor[..., 0], self.frame.images[0].name)

        points_tensor = tf.gather_nd(points_tensor, tf.where(mask))
        cp_points_tensor = tf.cast(
            tf.gather_nd(cp_points_tensor, tf.where(mask)), dtype=tf.float32)
        depth_tensor = tf.gather_nd(depth_tensor, tf.where(mask))

        # channel format: (x, y, X, Y, Z) where (x, y) are pixel coords and (X, Y, Z) world
        self.image_point_cloud_mapping = tf.concat(
            [cp_points_tensor[..., 1:3], points_tensor], axis=-1).numpy()
        # (x, y, depth)
        self.image_depth_mapping = tf.concat(
            [cp_points_tensor[..., 1:3], depth_tensor], axis=-1).numpy()

    def process_labels(self):
        """ """
        self.labels = self.frame.camera_labels[0]

        bounding_boxes = []
        object_classes = []
        for label in self.labels.labels:
            bounding_boxes.append([
                int(label.box.center_x - 0.5 * label.box.length),
                int(label.box.center_x + 0.5 * label.box.length),
                int(label.box.center_y - 0.5 * label.box.width),
                int(label.box.center_y + 0.5 * label.box.width)
            ])
            object_classes.append(label.type)

        assert len(bounding_boxes) == len(object_classes), "Process Labels"

        bounding_boxes = np.array(bounding_boxes)
        bounding_box_depth_average = np.zeros(len(self.labels.labels))

        if bounding_boxes.shape[0] > 0:
            points_x, points_y = \
                self.image_depth_mapping[..., 0], self.image_depth_mapping[..., 1]

            compare_x_lower = np.expand_dims(points_x, -1) > \
                np.expand_dims(bounding_boxes[..., 0], 0)
            compare_x_upper = np.expand_dims(points_x, -1) < \
                np.expand_dims(bounding_boxes[..., 1], 0)
            compare_y_lower = np.expand_dims(points_y, -1) > \
                np.expand_dims(bounding_boxes[..., 2], 0)
            compare_y_upper = np.expand_dims(points_y, -1) < \
                np.expand_dims(bounding_boxes[..., 3], 0)

            point_in_bounding_box = \
                compare_x_lower & compare_x_upper & compare_y_lower & compare_y_upper
            point_indicies, bounding_box_indicies = np.where(point_in_bounding_box)

            LIDAR_POINT_THRESHOLD = 10

            for bounding_box_index in range(len(bounding_boxes)):
                contained_points = np.where(bounding_box_indicies == bounding_box_index)[0]

                if len(contained_points) > LIDAR_POINT_THRESHOLD:
                    bounding_box_depth_average[bounding_box_index] = \
                        np.mean(self.image_depth_mapping[point_indicies[contained_points], 2])

        self.bounding_boxes = bounding_boxes
        self.object_classes = object_classes
        self.bounding_box_depth_average = bounding_box_depth_average

    def save_frame(self, path, downsample=True):
        """ """
        # originally (1280, 1920)
        if downsample:
            self.image = cv2.resize(self.image, (384, 256))
            self.bounding_boxes //= 5

        np.savez_compressed(
            path + '{}'.format(self.index),
            image=self.image,
            bounding_boxes=self.bounding_boxes,
            object_classes=self.object_classes,
            object_depths=self.bounding_box_depth_average
        )


def batch_read(path):
    """ """
    frames = []
    dataset = tf.data.TFRecordDataset(path, compression_type='')
    for index, data in enumerate(dataset):
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))
        frames.append(WaymoFrame(index, frame))
    return frames


def batch_process(frames):
    """ """
    for frame in frames:
        frame.process_image()
        frame.process_point_cloud()
        frame.process_labels()
        '''
        draw_bounding_boxes(frame.image, frame.bounding_boxes)
        imshow('Test Image', frame.image)
        print(frame.bounding_box_depth_average)
        exit()
        '''

def batch_write(frames, path):
    """ """
    path = '/'.join(path.split('/')[2:]).split('.')[0]
    path = DATALAKE_PATH + path + '/'
    os.makedirs(path, exist_ok=True)
    for frame in frames:
        frame.save_frame(path)


import multiprocessing

def process_batch(path):
    print('Processing batch')
    frames = batch_read(path)
    batch_process(frames)
    batch_write(frames, path)


def process_waymo():
    """ """
    start_time = time.time()

    batches = glob.glob(WAYMO_DATA_PATH + '*/*.tfrecord')

    p = multiprocessing.Pool(processes = 4)
    p.map(process_batch, batches)
    p.close()
    p.join()

    print(time.time() - start_time)


if __name__ == '__main__':
    process_waymo()
    '''
    batches = glob.glob(DATALAKE_PATH + '/*/*/*.npz')
    for batch in batches:
        tmp = np.load(batch)
        image = tmp['image']
        bounding_boxes = tmp['bounding_boxes']
        object_classes = tmp['object_classes']
        object_depths = tmp['object_depths']
        print(object_classes)
        print(object_depths)
        draw_bounding_boxes(image, bounding_boxes)
        imshow('Frame', image)
    '''
