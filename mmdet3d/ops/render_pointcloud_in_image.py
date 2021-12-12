# nuScenes dev-kit.
# Code written by Oscar Beijbom, Holger Caesar & Fong Whye Kit, 2020.

import json
import math
import os
import os.path as osp
import sys
import time
from datetime import datetime
from typing import Tuple, List, Iterable

import mmcv
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics
from PIL import Image
from matplotlib import rcParams
from matplotlib.axes import Axes
from pyquaternion import Quaternion
from tqdm import tqdm
from PIL import Image
import torch

def render_pointcloud_in_image(#self,
                                pointcloud,
                                image,
                                T_lidar2cam,
                                cam_intrinsic,
                                #    sample_token: str,
                                # dot_size: int = 5,
                                #    pointsensor_channel: str = 'LIDAR_TOP',
                                #    camera_channel: str = 'CAM_FRONT',
                                #    out_path: str = None,
                                #    render_intensity: bool = False,
                                #    show_lidarseg: bool = False,
                                #    filter_lidarseg_labels: List = None,
                                # ax: Axes = None,
                                #    show_lidarseg_legend: bool = False,
                                verbose: bool = False,
                                #    lidarseg_preds_bin_path: str = None,
                                #    show_panoptic: bool = False
                                min_dist = 1,
                                title : str = 'render_pointcloud_in_image' 
                                ):
    """
    Scatter-plots a pointcloud on top of image.
    :param sample_token: Sample token.
    :param dot_size: Scatter plot dot size.
    :param pointsensor_channel: RADAR or LIDAR channel name, e.g. 'LIDAR_TOP'.
    :param camera_channel: Camera channel name, e.g. 'CAM_FRONT'.
    :param out_path: Optional path to save the rendered figure to disk.
    :param render_intensity: Whether to render lidar intensity instead of point depth.
    :param show_lidarseg: Whether to render lidarseg labels instead of point depth.
    :param filter_lidarseg_labels: Only show lidar points which belong to the given list of classes.
    :param ax: Axes onto which to render.
    :param show_lidarseg_legend: Whether to display the legend for the lidarseg labels in the frame.
    :param verbose: Whether to display the image in a window.
    :param lidarseg_preds_bin_path: A path to the .bin file which contains the user's lidar segmentation
                                    predictions for the sample.
    :param show_panoptic: When set to True, the lidar data is colored with the panoptic labels. When set
        to False, the colors of the lidar data represent the distance from the center of the ego vehicle.
        If show_lidarseg is True, show_panoptic will be set to False.
    """
    # if show_lidarseg:
    #     show_panoptic = False
    # sample_record = self.nusc.get('sample', sample_token)

    # # Here we just grab the front camera and the point sensor.
    # pointsensor_token = sample_record['data'][pointsensor_channel]
    # camera_token = sample_record['data'][camera_channel]

    points, mask, coloring, im = map_pointcloud_to_image(pointcloud, 
                                                        image,
                                                        T_lidar2cam,
                                                        cam_intrinsic,
                                                        min_dist
                                                        # pointsensor_token, camera_token,
                                                        # render_intensity=render_intensity,
                                                        # show_lidarseg=show_lidarseg,
                                                        # filter_lidarseg_labels=filter_lidarseg_labels,
                                                        # lidarseg_preds_bin_path=lidarseg_preds_bin_path,
                                                        # show_panoptic=show_panoptic
                                                        )
    if verbose:
        show_overlay(points, mask, im, coloring, title=title)
    return points[:2,:], mask, coloring 

def show_overlay(points, 
                mask, 
                im,
                coloring,
                dot_size: int = 5,
                ax: Axes = None,
                title : str = 'render_pointcloud_in_image' 
                ):
    # Init axes.
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(9, 16))
        # if lidarseg_preds_bin_path:
        #     fig.canvas.set_window_title(sample_token + '(predictions)')
        # else:
        #     fig.canvas.set_window_title(sample_token)
        fig.canvas.set_window_title(title)
    # else:  # Set title on if rendering as part of render_sample.
    #     ax.set_title(camera_channel)
    im = mmcv.bgr2rgb(im)
    ax.imshow(im)
    ax.scatter(points[0, mask], points[1, mask], c=coloring[mask], s=dot_size)
    ax.axis('off')

    # # Produce a legend with the unique colors from the scatter.
    # if pointsensor_channel == 'LIDAR_TOP' and (show_lidarseg or show_panoptic) and show_lidarseg_legend:
    #     # If user does not specify a filter, then set the filter to contain the classes present in the pointcloud
    #     # after it has been projected onto the image; this will allow displaying the legend only for classes which
    #     # are present in the image (instead of all the classes).
    #     if filter_lidarseg_labels is None:
    #         if show_lidarseg:
    #             # Since the labels are stored as class indices, we get the RGB colors from the
    #             # colormap in an array where the position of the RGB color corresponds to the index
    #             # of the class it represents.
    #             color_legend = colormap_to_colors(self.nusc.colormap, self.nusc.lidarseg_name2idx_mapping)
    #             filter_lidarseg_labels = get_labels_in_coloring(color_legend, coloring)
    #         else:
    #             # Only show legends for all stuff categories for panoptic.
    #             filter_lidarseg_labels = stuff_cat_ids(len(self.nusc.lidarseg_name2idx_mapping))

    #     if filter_lidarseg_labels and show_panoptic:
    #         # Only show legends for filtered stuff categories for panoptic.
    #         stuff_labels = set(stuff_cat_ids(len(self.nusc.lidarseg_name2idx_mapping)))
    #         filter_lidarseg_labels = list(stuff_labels.intersection(set(filter_lidarseg_labels)))

    #     create_lidarseg_legend(filter_lidarseg_labels, self.nusc.lidarseg_idx2name_mapping, self.nusc.colormap,
    #                             loc='upper left', ncol=1, bbox_to_anchor=(1.05, 1.0))

    # if out_path is not None:
    #     plt.savefig(out_path, bbox_inches='tight', pad_inches=0, dpi=200)
    # if verbose:
    plt.show()
    

def map_pointcloud_to_image(#self,
                                lidar_points, 
                                im,
                                T_lidar2cam,
                                cam_intrinsic,                   
                                # pointsensor_token: str,
                                # camera_token: str,
                                min_dist: float = 1.0,
                                # render_intensity: bool = False,
                                # show_lidarseg: bool = False,
                                # filter_lidarseg_labels: List = None,
                                # lidarseg_preds_bin_path: str = None,
                                # show_panoptic: bool = False) -> Tuple:
                                ):
    """
    Given a point sensor (lidar/radar) token and camera sample_data token, load pointcloud and map it to the image
    plane.
    :param pointsensor_token: Lidar/radar sample_data token.
    :param camera_token: Camera sample_data token.
    :param min_dist: Distance from the camera below which points are discarded.
    :param render_intensity: Whether to render lidar intensity instead of point depth.
    :param show_lidarseg: Whether to render lidar intensity instead of point depth.
    :param filter_lidarseg_labels: Only show lidar points which belong to the given list of classes. If None
        or the list is empty, all classes will be displayed.
    :param lidarseg_preds_bin_path: A path to the .bin file which contains the user's lidar segmentation
                                    predictions for the sample.
    :param show_panoptic: When set to True, the lidar data is colored with the panoptic labels. When set
        to False, the colors of the lidar data represent the distance from the center of the ego vehicle.
        If show_lidarseg is True, show_panoptic will be set to False.
    :return (pointcloud <np.float: 2, n)>, coloring <np.float: n>, image <Image>).
    """
    # cam = self.nusc.get('sample_data', camera_token)
    # pointsensor = self.nusc.get('sample_data', pointsensor_token)
    # pcl_path = osp.join(self.nusc.dataroot, pointsensor['filename'])
    # if pointsensor['sensor_modality'] == 'lidar':
    #     if show_lidarseg or show_panoptic:
    #         gt_from = 'lidarseg' if show_lidarseg else 'panoptic'
    #         assert hasattr(self.nusc, gt_from), f'Error: nuScenes-{gt_from} not installed!'

    #         # Ensure that lidar pointcloud is from a keyframe.
    #         assert pointsensor['is_key_frame'], \
    #             'Error: Only pointclouds which are keyframes have lidar segmentation labels. Rendering aborted.'

    #         assert not render_intensity, 'Error: Invalid options selected. You can only select either ' \
    #                                         'render_intensity or show_lidarseg, not both.'

    #     pc = LidarPointCloud.from_file(pcl_path)
    # else:
    #     pc = RadarPointCloud.from_file(pcl_path)
    # im = Image.open(osp.join(self.nusc.dataroot, cam['filename']))

    # # Points live in the point sensor frame. So they need to be transformed via global to the image plane.
    # # First step: transform the pointcloud to the ego vehicle frame for the timestamp of the sweep.
    # cs_record = self.nusc.get('calibrated_sensor', pointsensor['calibrated_sensor_token'])
    # pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
    # pc.translate(np.array(cs_record['translation']))

    # # Second step: transform from ego to the global frame.
    # poserecord = self.nusc.get('ego_pose', pointsensor['ego_pose_token'])
    # pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix)
    # pc.translate(np.array(poserecord['translation']))

    # # Third step: transform from global into the ego vehicle frame for the timestamp of the image.
    # poserecord = self.nusc.get('ego_pose', cam['ego_pose_token'])
    # pc.translate(-np.array(poserecord['translation']))
    # pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix.T)

    # # Fourth step: transform from ego into the camera.
    # cs_record = self.nusc.get('calibrated_sensor', cam['calibrated_sensor_token'])
    # pc.translate(-np.array(cs_record['translation']))
    # pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix.T)

    # Fifth step: actually take a "picture" of the point cloud.
    # Grab the depths (camera frame z axis points away from the camera).
    # depths = pc.points[2, :]

    im = Image.fromarray(im)
    # points in lidar frame
    P_t_lidar = np.vstack((lidar_points.T, np.ones((lidar_points.shape[0]))))
    # points in camera frame
    P_t_cam = (T_lidar2cam @ P_t_lidar)[:3,:]
    # # points in image frame
    # P_t_img = cam_intrinsic @ P_t_cam
    pc = P_t_cam
    depths = pc[2, :]

    # if render_intensity:
    #     assert pointsensor['sensor_modality'] == 'lidar', 'Error: Can only render intensity for lidar, ' \
    #                                                         'not %s!' % pointsensor['sensor_modality']
    #     # Retrieve the color from the intensities.
    #     # Performs arbitary scaling to achieve more visually pleasing results.
    #     intensities = pc.points[3, :]
    #     intensities = (intensities - np.min(intensities)) / (np.max(intensities) - np.min(intensities))
    #     intensities = intensities ** 0.1
    #     intensities = np.maximum(0, intensities - 0.5)
    #     coloring = intensities
    # elif show_lidarseg or show_panoptic:
    #     assert pointsensor['sensor_modality'] == 'lidar', 'Error: Can only render lidarseg labels for lidar, ' \
    #                                                         'not %s!' % pointsensor['sensor_modality']

    #     gt_from = 'lidarseg' if show_lidarseg else 'panoptic'
    #     semantic_table = getattr(self.nusc, gt_from)

    #     if lidarseg_preds_bin_path:
    #         sample_token = self.nusc.get('sample_data', pointsensor_token)['sample_token']
    #         lidarseg_labels_filename = lidarseg_preds_bin_path
    #         assert os.path.exists(lidarseg_labels_filename), \
    #             'Error: Unable to find {} to load the predictions for sample token {} (lidar ' \
    #             'sample data token {}) from.'.format(lidarseg_labels_filename, sample_token, pointsensor_token)
    #     else:
    #         if len(semantic_table) > 0:  # Ensure {lidarseg/panoptic}.json is not empty (e.g. in case of v1.0-test).
    #             lidarseg_labels_filename = osp.join(self.nusc.dataroot,
    #                                                 self.nusc.get(gt_from, pointsensor_token)['filename'])
    #         else:
    #             lidarseg_labels_filename = None

    #     if lidarseg_labels_filename:
    #         # Paint each label in the pointcloud with a RGBA value.
    #         if show_lidarseg:
    #             coloring = paint_points_label(lidarseg_labels_filename,
    #                                             filter_lidarseg_labels,
    #                                             self.nusc.lidarseg_name2idx_mapping,
    #                                             self.nusc.colormap)
    #         else:
    #             coloring = paint_panop_points_label(lidarseg_labels_filename,
    #                                                 filter_lidarseg_labels,
    #                                                 self.nusc.lidarseg_name2idx_mapping,
    #                                                 self.nusc.colormap)

    #     else:
    #         coloring = depths
    #         print(f'Warning: There are no lidarseg labels in {self.nusc.version}. Points will be colored according '
    #                 f'to distance from the ego vehicle instead.')
    # else:
    # # Retrieve the color from the depth.
    coloring = depths

    # Take the actual picture (matrix multiplication with camera-matrix + renormalization).
    # points = view_points(pc.points[:3, :], np.array(cs_record['camera_intrinsic']), normalize=True)
    points = view_points(pc, cam_intrinsic, normalize=True)

    # Remove points that are either outside or behind the camera. Leave a margin of 1 pixel for aesthetic reasons.
    # Also make sure points are at least 1m in front of the camera to avoid seeing the lidar points on the camera
    # casing for non-keyframes which are slightly out of sync.
    mask = np.ones(depths.shape[0], dtype=bool)
    mask = np.logical_and(mask, depths > min_dist)
    mask = np.logical_and(mask, points[0, :] > 1)
    mask = np.logical_and(mask, points[0, :] < im.size[0] - 1)
    mask = np.logical_and(mask, points[1, :] > 1)
    mask = np.logical_and(mask, points[1, :] < im.size[1] - 1)
    # points = points[:, mask]
    # coloring = coloring[mask]

    return points, mask, coloring, im

def map_pointcloud_to_image_torch(#self,
                                lidar_points, 
                                im,
                                T_lidar2cam,
                                cam_intrinsic,                   
                                # pointsensor_token: str,
                                # camera_token: str,
                                min_dist: float = 1.0,
                                # render_intensity: bool = False,
                                # show_lidarseg: bool = False,
                                # filter_lidarseg_labels: List = None,
                                # lidarseg_preds_bin_path: str = None,
                                # show_panoptic: bool = False) -> Tuple:
                                ):
    """
    """
    
    #im = Image.fromarray(im)
    im = im.permute(2,1,0)
    # points in lidar frame
    # P_t_lidar = np.vstack((lidar_points.T, np.ones((lidar_points.shape[0]))))
    P_t_lidar = torch.cat([lidar_points, torch.ones((lidar_points.size(0),1)).to(lidar_points.device)], dim=1)
    P_t_lidar = P_t_lidar.T
    # points in camera frame
    P_t_cam = torch.mm(T_lidar2cam.float(), P_t_lidar)[:3,:] 
    # # points in image frame
    # P_t_img = cam_intrinsic @ P_t_cam
    # pc = P_t_cam
    depths = P_t_cam[2, :]

    coloring = depths

    # Take the actual picture (matrix multiplication with camera-matrix + renormalization).
    # points = view_points(pc.points[:3, :], np.array(cs_record['camera_intrinsic']), normalize=True)
    # points = view_points(pc, cam_intrinsic, normalize=True)
    P_t_img = torch.mm(cam_intrinsic.float(), P_t_cam)
    points = torch.divide(P_t_img[:2,:], P_t_img[2,:])

    # Remove points that are either outside or behind the camera. Leave a margin of 1 pixel for aesthetic reasons.
    # Also make sure points are at least 1m in front of the camera to avoid seeing the lidar points on the camera
    # casing for non-keyframes which are slightly out of sync.
    mask = torch.ones(depths.size(0), dtype=bool).to(points.device)
    mask = torch.logical_and(mask, depths > min_dist)
    mask = torch.logical_and(mask, points[0, :] > 1)
    mask = torch.logical_and(mask, points[0, :] < im.size(0) - 1)
    mask = torch.logical_and(mask, points[1, :] > 1)
    mask = torch.logical_and(mask, points[1, :] < im.size(1) - 1)
    # points = points[:, mask]
    # coloring = coloring[mask]

    return points, mask, coloring, im

def view_points(points: np.ndarray, view: np.ndarray, normalize: bool) -> np.ndarray:
    """
    This is a helper class that maps 3d points to a 2d plane. It can be used to implement both perspective and
    orthographic projections. It first applies the dot product between the points and the view. By convention,
    the view should be such that the data is projected onto the first 2 axis. It then optionally applies a
    normalization along the third dimension.

    For a perspective projection the view should be a 3x3 camera matrix, and normalize=True
    For an orthographic projection with translation the view is a 3x4 matrix and normalize=False
    For an orthographic projection without translation the view is a 3x3 matrix (optionally 3x4 with last columns
     all zeros) and normalize=False

    :param points: <np.float32: 3, n> Matrix of points, where each point (x, y, z) is along each column.
    :param view: <np.float32: n, n>. Defines an arbitrary projection (n <= 4).
        The projection should be such that the corners are projected onto the first 2 axis.
    :param normalize: Whether to normalize the remaining coordinate (along the third axis).
    :return: <np.float32: 3, n>. Mapped point. If normalize=False, the third coordinate is the height.
    """

    assert view.shape[0] <= 4
    assert view.shape[1] <= 4
    assert points.shape[0] == 3

    viewpad = np.eye(4)
    viewpad[:view.shape[0], :view.shape[1]] = view

    nbr_points = points.shape[1]

    # Do operation in homogenous coordinates.
    points = np.concatenate((points, np.ones((1, nbr_points))))
    points = np.dot(viewpad, points)
    points = points[:3, :]

    if normalize:
        points = points / points[2:3, :].repeat(3, 0).reshape(3, nbr_points)

    return points

