# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import numpy as np
import warnings
from mmcv import Config, DictAction, mkdir_or_exist, track_iter_progress
from os import path as osp
from pyquaternion.quaternion import Quaternion

from torch.utils import data

from mmdet3d.core.bbox import (Box3DMode, CameraInstance3DBoxes, Coord3DMode,
                               DepthInstance3DBoxes, LiDARInstance3DBoxes)
from mmdet3d.core.visualizer import (show_multi_modality_result, show_result,
                                     show_seg_result)#, Visualizer, BoundingBox3D)
from mmdet3d.datasets import build_dataset
import mmcv

# ROS imports 
import tf
import os
# import cv2
import rospy
import rosbag
from tf2_msgs.msg import TFMessage
from datetime import datetime
from std_msgs.msg import Header
from sensor_msgs.msg import CameraInfo, Imu, PointField, NavSatFix
import sensor_msgs.point_cloud2 as pcl2
from geometry_msgs.msg import TransformStamped, TwistStamped, Transform
from cv_bridge import CvBridge
from nav_msgs.msg import Odometry

def parse_args():
    parser = argparse.ArgumentParser(description='Browse a dataset')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--skip-type',
        type=str,
        nargs='+',
        default=['Normalize'],
        help='skip some useless pipeline')
    parser.add_argument(
        '--output-dir',
        default=None,
        type=str,
        help='If there is no display interface, you can save it')
    parser.add_argument(
        '--task',
        type=str,
        choices=['det', 'seg', 'multi_modality-det', 'mono-det'],
        help='Determine the visualization method depending on the task.')
    parser.add_argument(
        '--online',
        action='store_true',
        help='Whether to perform online visualization. Note that you often '
        'need a monitor to do so.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument('--det_file', help='output result file in pickle format')
    
    args = parser.parse_args()
    return args


def build_data_cfg(config_path, skip_type, cfg_options):
    """Build data config for loading visualization data."""
    cfg = Config.fromfile(config_path)
    if cfg_options is not None:
        cfg.merge_from_dict(cfg_options)
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])
    # extract inner dataset of `RepeatDataset` as `cfg.data.train`
    # so we don't need to worry about it later
    if cfg.data.train['type'] == 'RepeatDataset':
        cfg.data.train = cfg.data.train.dataset
    # use only first dataset for `ConcatDataset`
    if cfg.data.train['type'] == 'ConcatDataset':
        cfg.data.train = cfg.data.train.datasets[0]
    train_data_cfg = cfg.data.train
    # eval_pipeline purely consists of loading functions
    # use eval_pipeline for data loading
    train_data_cfg['pipeline'] = [
        x for x in cfg.eval_pipeline if x['type'] not in skip_type
    ]

    return cfg

def save_velo_data(bag, infos, prepared_data):
    scan_l = prepared_data['points']._data.numpy()
    # retain only 3 din
    scan_l = scan_l[:,:3]
    l2e_r_mat = Quaternion(infos['lidar2ego_rotation']).rotation_matrix
    l2e_mat = get_transformation_matrix(l2e_r_mat, infos['lidar2ego_translation'])
    scan_l[:,:3] = transform_pc(scan_l[:,:3].copy(), l2e_mat)

    # add color
    if 'color_to_lidar' in prepared_data and 'color_mask' in prepared_data:
        color_mask = prepared_data['color_mask']
        scan_l = scan_l[color_mask,:].tolist()
        color_to_lidar = prepared_data['color_to_lidar']
        color_visible = [struct.unpack('I', struct.pack('BBBB', c[0][2], c[0][1], c[0][0], 255)) for c in zip(color_to_lidar[color_mask,:])]
        # color_visible = np.array([struct.unpack('I', struct.pack('BBBB', c[0][2], c[0][1], c[0][0], 255))[0] for c in zip(color_to_lidar[color_mask,:])])
        # scan_l = np.hstack((scan_l, color_visible.reshape(-1,1)))
        [scan.extend(rgb) for (scan, rgb) in zip(scan_l, color_visible)]

    # add semantic labels
    if 'pts_semantickittiformat_mask' in prepared_data:
        # scan_l = np.hstack((scan_l, prepared_data['pts_semantickittiformat_mask'].reshape(-1,1)))
        labels = prepared_data['pts_semantickittiformat_mask'].reshape(-1,1)
        labels = labels[color_mask]
        [scan.extend(l) for (scan, l) in zip(scan_l, labels)]
        # [scan.extend([0]) for scan in scan_l]
        # for (scan, l) in zip(scan_l, labels):
        #     scan[-1]=l.tolist()[0]

    # else:
    #     [scan.extend(np.array([0])) for scan in scan_l]

    # if 'color_mask' in prepared_data:
    #     color_mask = prepared_data['color_mask']
    #     scan_l = scan_l[color_mask,:]

    # create header
    header = Header()
    header.frame_id = 'base_link'
    header.stamp = rospy.Time.from_sec(infos['timestamp']*1e-6)

    # fill pcl msg
    fields = [PointField('x', 0, PointField.FLOAT32, 1),
                PointField('y', 4, PointField.FLOAT32, 1),
                PointField('z', 8, PointField.FLOAT32, 1),
                # PointField('i', 12, PointField.FLOAT32, 1),
                # PointField('t', 16, PointField.FLOAT32, 1)
                PointField('rgb', 12, PointField.UINT32, 1),
                PointField('label', 16, PointField.UINT32, 1)
                ]
    pcl_msg = pcl2.create_cloud(header, fields, scan_l)

    bag.write('/lidar_top', pcl_msg, t=pcl_msg.header.stamp)


def inv(transform):
    "Invert rigid body transformation matrix"
    R = transform[0:3, 0:3]
    t = transform[0:3, 3]
    t_inv = -1 * R.T.dot(t)
    transform_inv = np.eye(4)
    transform_inv[0:3, 0:3] = R.T
    transform_inv[0:3, 3] = t_inv
    return transform_inv

def get_transformation_matrix(rot, trans):
    T = np.eye(4)
    T[:3,:3] = rot
    T[:3,-1] = trans
    return T

def transform_pc(pc, T):
    pc_hom = np.hstack((pc, np.zeros((pc.shape[0],1))))
    pc_hom = (T @ pc_hom.T).T
    return pc_hom[:,:3]


def get_tf_stamped(frame, child_frame, timestamp, trans, rot):
    # translation in x,y,z
    # rot in quaternion: 2,x,y,z
    tf_stamped = TransformStamped()
    tf_stamped.header.stamp = rospy.Time.from_sec(timestamp)
    tf_stamped.header.frame_id = frame #'map'
    tf_stamped.child_frame_id = child_frame #'odom'
    
    transform = Transform()
    transform.translation.x = trans[0]
    transform.translation.y = trans[1]
    transform.translation.z = trans[2]
    transform.rotation.x = rot[1]
    transform.rotation.y = rot[2]
    transform.rotation.z = rot[3]
    transform.rotation.w = rot[0]

    tf_stamped.transform = transform
    return tf_stamped

def save_tf(bag, infos):
    tf_msg = TFMessage()

    # define tf bw map and odom (identity if GT available)
    tf_stamped = get_tf_stamped('map', 'odom', infos['timestamp']*1e-6, [0,0,0], [1,0,0,0])
    tf_msg.transforms.append(tf_stamped)

    tf_stamped = get_tf_stamped('odom', 'base_link', infos['timestamp']*1e-6, infos['ego2global_translation'], infos['ego2global_rotation'])
    # tf_stamped = get_tf_stamped('odom', 'base_link', infos['timestamp']*1e-6, e2g_mat_inv[:3,3], Quaternion(matrix=e2g_mat_inv))
    tf_msg.transforms.append(tf_stamped)

    l2e_r_mat = Quaternion(infos['lidar2ego_rotation']).rotation_matrix
    l2e_mat = get_transformation_matrix(l2e_r_mat, infos['lidar2ego_translation'])
    l2e_mat_inv = inv(l2e_mat)

    # define static tf bw base_link and sensors
    # tf_stamped = get_tf_stamped('base_link', 'lidar_top', infos['timestamp']*1e-6, infos['lidar2ego_translation'], infos['lidar2ego_rotation'])
    tf_stamped = get_tf_stamped('base_link', 'lidar_top', infos['timestamp']*1e-6, l2e_mat_inv[:3,3], Quaternion(matrix=l2e_mat_inv))
    tf_msg.transforms.append(tf_stamped)

    for cam_name, cam_infos in infos['cams'].items():
        tf_stamped = get_tf_stamped('base_link', cam_name.lower()+'/raw', infos['timestamp']*1e-6, cam_infos['sensor2ego_translation'], cam_infos['sensor2ego_rotation'])
        tf_msg.transforms.append(tf_stamped)

    # TODO: add radar topics 

    bag.write('/tf', tf_msg, tf_msg.transforms[0].header.stamp)

def save_odom(bag, infos):
    odom_msg = Odometry()
    odom_msg.header.stamp = rospy.Time.from_sec(infos['timestamp']*1e-6)
    odom_msg.header.frame_id = 'odom'
    odom_msg.child_frame_id = 'base_link'
    odom_msg.pose.pose.position.x = infos['ego2global_translation'][0]
    odom_msg.pose.pose.position.y = infos['ego2global_translation'][1]
    odom_msg.pose.pose.position.z = infos['ego2global_translation'][2]
    rot = infos['ego2global_rotation'].copy()
    # rot[0], rot[3] = rot[3], rot[0] 
    odom_msg.pose.pose.orientation.w = infos['ego2global_rotation'][0]
    odom_msg.pose.pose.orientation.x = infos['ego2global_rotation'][1]
    odom_msg.pose.pose.orientation.y = infos['ego2global_rotation'][2]
    odom_msg.pose.pose.orientation.z = infos['ego2global_rotation'][3]

    bag.write('/odom', odom_msg, odom_msg.header.stamp)

def main():
    args = parse_args()

    if args.output_dir is not None:
        mkdir_or_exist(args.output_dir)

    cfg = build_data_cfg(args.config, args.skip_type, args.cfg_options)
    try:
        dataset = build_dataset(
            cfg.data.train, default_args=dict(filter_empty_gt=False))
    except TypeError:  # seg dataset doesn't have `filter_empty_gt` key
        dataset = build_dataset(cfg.data.train)
    data_infos = dataset.data_infos
    dataset_type = cfg.dataset_type

    CONF_THRESH = 0.5
    data_list = []
    # det_results = mmcv.load(args.det_file)
    # prog_bar = mmcv.ProgressBar(len(det_results))
    prog_bar = mmcv.ProgressBar(len(dataset.data_infos))

    compression = rosbag.Compression.NONE
    # compression = rosbag.Compression.BZ2
    # compression = rosbag.Compression.LZ4

    # create first rosbag
    bag = rosbag.Bag(os.path.join(args.output_dir, "nuscenes_{}.bag".format(dataset.data_infos[0]['scene_name'])), 'w', compression=compression)
    # prev_timestamp = dataset.data_infos[0]['timestamp']
    scene_name = dataset.data_infos[0]['scene_name']
    for index, infos in enumerate(dataset.data_infos):
        if infos['scene_name'] != scene_name:
            print("## OVERVIEW ##")
            print(bag)
            bag.close()
            scene_name = infos['scene_name']
            # create a new rosbag
            bag = rosbag.Bag(os.path.join(args.output_dir, "nuscenes_{}.bag".format(infos['scene_name'])), 'w', compression=compression)
       
        prepared_data = dataset.prepare_train_data(index)
        ann_info = dataset.get_ann_info(index)

        # delta_t = infos['timestamp'] - prev_timestamp
        # print("Exporting data for scene ", infos['name'])
        # print("Exporting data, delta_t: ", delta_t)
        
        # save_static_tf(bag, infos)
        save_tf(bag, infos)
        save_odom(bag, infos)
        # save_dynamic_tf(bag, kitti, args.kitti_type, initial_time=current_epoch)
        save_velo_data(bag, infos, prepared_data)
        #     for camera in used_cameras:
        #         save_camera_data(bag, args.kitti_type, kitti, util, bridge, camera=camera[0], camera_frame_id=camera[1], topic=camera[2], initial_time=current_epoch) 
        # prev_timestamp = infos['timestamp']
        prog_bar.update()

if __name__ == '__main__':
    main()
