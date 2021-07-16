#!/usr/bin/env python
# Source: https://github.com/intel-isl/Open3D/blob/master/examples/python/gui/video.py
from concurrent.futures import thread
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import os
import time
import threading

from mmcv import track_iter_progress
import mmcv
from os import path as osp
import torch
from mmdet3d.core.bbox import (Box3DMode, CameraInstance3DBoxes, Coord3DMode,
                               DepthInstance3DBoxes, LiDARInstance3DBoxes)
from .image_vis import (draw_camera_bbox3d_on_img, draw_depth_bbox3d_on_img,
                        draw_lidar_bbox3d_on_img)
import warnings
from matplotlib import pyplot as plt
import cv2

class GUIWindow:
    """
    This class serves as an experimental Graphical User Interface for an mmdetection3d dataset

    Args:
        dataset (mmdet3d.datasets.xx): The dataset for which visualizer is created.

    """
    def __init__(self, dataset, dataset_type, point_cloud_range, monitor_width = 1920, monitor_height = 1080):

        self.dataset = dataset
        self.dataset_type = dataset_type
        self.bounds = o3d.geometry.AxisAlignedBoundingBox(point_cloud_range[:3], point_cloud_range[3:])
        self.set_dataset_specific_prop()

        self.window = gui.Application.instance.create_window(
            "GUI - Visualizer", monitor_width, monitor_height) #TODO: try to get monitor res?
        self.em = self.window.theme.font_size
        self.window.set_on_layout(self._on_layout)
        self.window.set_on_close(self._on_close)

        self.init_data_str()
        self.init_user_interface()

        self.pause = False
        self.index = 0
        self.is_done = False
        threading.Thread(target=self._update_thread).start()

        self.event = threading.Event()

    def _on_layout(self, layout_context):
        contentRect = self.window.content_rect

        img_panel_width = 75 * self.em  # where 1 em = 16 px
        metadata_panel_height = 7.5 * self.em

        self.widget3d.frame = gui.Rect(contentRect.x, contentRect.y,
                                       contentRect.width - img_panel_width,
                                       contentRect.height - metadata_panel_height)
        self.img_panel.frame = gui.Rect(self.widget3d.frame.get_right(),
                                    contentRect.y, img_panel_width,
                                    contentRect.height)
        self.metadata_panel.frame = gui.Rect(contentRect.x, self.widget3d.frame.get_bottom(),
                                    self.widget3d.frame.get_right(),
                                    metadata_panel_height)
        
        # factor by which to rescale images to fit the image panel
        self.img_rescaling_factor = img_panel_width / self.img_panel_cols / self.img_size[1]    

    def init_data_str(self):
        data_infos = self.dataset.data_infos
        modality = self.dataset.modality
            
        if modality['use_lidar']:
            # TODO: change logic for multi-lidar setup?
            self.pcd = o3d.t.geometry.PointCloud(o3d.core.Device("CPU:0"))
            self.pcd.point["points"] = self._make_tcloud_array(np.random.rand(100,3)*10)

            self.pcd_mat = rendering.Material()
            self.pcd_mat.shader = "defaultLit"

            self.bbox_mat = rendering.Material()
            self.bbox_mat.shader = "unlitLine"
            self.bbox_mat.line_width = 2 * self.window.scaling

            self.bbox_count = 0

        if modality['use_camera']:
            self.img_dict = dict()
            # TODO: what if only some cameras need to be displayed?
            for key, value in self.panel_2_cam.items():
                self.img_dict[value] = {
                    'img': np.zeros(self.img_size).astype(np.uint8),
                    'overlay': np.zeros(self.img_size).astype(np.uint8),
                    'bbox': np.zeros(self.img_size).astype(np.uint8),
                    'widget': gui.ImageWidget(o3d.geometry.Image())
                }

        if 'use_radar' in modality and modality['use_radar']:  # TODO: this doesn't work yet.
            self.radar_dict = dict()
            # for key, _ in data_infos[0]['radars'].items():
            for key, _ in self.data_info_radar.items():
                self.radar_dict[key] = o3d.geometry.PointCloud

        #TODO: its possible to initialize more structures based on metadata in self.dataset
        # eg HD Map, or based on class_names of annotated 3D bboxes, semantic segmentation etc

    def init_user_interface(self):
        self.widget3d = gui.SceneWidget()
        self.widget3d.scene = rendering.Open3DScene(self.window.renderer)

        # sample = self.get_stitched_pcd()
        # self.widget3d.scene.add_geometry("Stitched PC", sample, lit) # TODO: (',').join( list(data_infos[0]['cams'].keys()))
        self.widget3d.scene.add_geometry("Stitched PC", self.pcd, self.pcd_mat)
        #NOTE: set bounds carefully! else the visualizer zooms in excessively
        self.widget3d.setup_camera(60.0, self.bounds, self.bounds.get_center()) 
        self.widget3d.scene.show_axes(True)
        self.window.add_child(self.widget3d)

        margin = 0.25 * self.em
        self.img_panel = gui.VGrid(self.img_panel_cols, margin)

        for index, _ in enumerate(self.panel_2_cam):
            self.img_panel.add_child(self.img_dict[self.panel_2_cam[index]]['widget'])

        self.window.add_child(self.img_panel)

        # self.metadata_panel = gui.Horiz()
        self.metadata_panel = gui.VGrid(3, margin)

        prev_button = gui.Button("<")
        prev_button.horizontal_padding_em = 0.5
        prev_button.vertical_padding_em = 0.5
        prev_button.set_on_clicked(self._on_prev)
        self.metadata_panel.add_child(prev_button)

        pause_button = gui.Button("Pause")
        pause_button.horizontal_padding_em = 0.5
        pause_button.vertical_padding_em = 0.5
        pause_button.set_on_clicked(self._on_pause)
        # self.metadata_panel.add_stretch()
        self.metadata_panel.add_child(pause_button)
        
        next_button = gui.Button(">")
        next_button.horizontal_padding_em = 0.5
        next_button.vertical_padding_em = 0.5
        next_button.set_on_clicked(self._on_next)
        self.metadata_panel.add_child(next_button)
        
        self.window.add_child(self.metadata_panel)
        #TODO: add for radar and HD Maps?

    def _on_pause(self):
        self.pause = not(self.pause)
        if not self.pause:
            self.event.set()
            self.event = threading.Event() 
        # print(self.pause)
        return

    def _on_next(self):
        if self.pause:
            print("Processing Next frame")
            self.index += 1
            self.get_prepared_data(self.index)

            if not self.is_done:
                gui.Application.instance.post_to_main_thread(
                    self.window, self.update)
        else:
            print('Press Pause first')
        return

    def _on_prev(self):
        if self.pause:
            print("Processing Previous frame")
        else:
            print('Press Pause first')
        return

    def set_dataset_specific_prop(self):
        if self.dataset_type in ['NuScenesDataset', 'LyftDataset']: #TODO: ArgoDataset
            self.panel_2_cam = {
                0: 'CAM_FRONT_LEFT',
                1: 'CAM_FRONT',
                2: 'CAM_FRONT_RIGHT',
                3: 'CAM_BACK_LEFT',
                4: 'CAM_BACK',
                5: 'CAM_BACK_RIGHT'
            }

            self.img_panel_cols = 3     # number of columns in the image panel
            self.img_size = mmcv.imread(self.dataset.data_infos[0]['cams'][self.panel_2_cam[0]]['data_path']).shape  # height, width, # of channels

        elif self.dataset_type in ['KittiDataset']:
            self.panel_2_cam = {
                0: 'CAM_FRONT_LEFT',
                #1: 'CAM_FRONT_RIGHT',  # TODO: mmdetection doesn't read the right image 
            }
            
            self.img_panel_cols = 1     # number of columns in the image panel
            self.img_size = mmcv.imread(osp.join(self.dataset.data_root, self.dataset.data_infos[0]['image']['image_path'])).shape    # height, width, # of channels

            # TODO: data_path and file_path can be set here too

    def get_stitched_pcd(self):
        # TODO: this needs to be modified for multi-lidar setup
        # return self.pcd_dict['LIDAR_TOP']
        a = o3d.t.geometry.PointCloud(o3d.core.Device("CPU:0"))
        a.point["points"] = o3d.core.Tensor.from_numpy(np.ascontiguousarray(self.pcd_dict['LIDAR_TOP'][:,:3]))
        return a

    def _make_tcloud_array(self, np_array, copy=False):
        if copy or not np_array.data.c_contiguous:
            return o3d.core.Tensor(np_array)
        else:
            return o3d.core.Tensor.from_numpy(np_array)

    def _on_close(self):
        self.is_done = True
        return True  # False would cancel the close

    def _draw_bboxes(self, bbox3d,
                 vis=None,
                 points_colors=None,
                 pcd=None,
                 bbox_color=(0, 1, 0),
                 points_in_box_color=(1, 0, 0),
                 rot_axis=2,
                 center_mode='lidar_bottom',
                 mode='xyz'):
        """Draw bbox on visualizer and change the color of points inside bbox3d.

        Args:
            bbox3d (numpy.array | torch.tensor, shape=[M, 7]):
                3d bbox (x, y, z, dx, dy, dz, yaw) to visualize.
            vis (:obj:`open3d.visualization.Visualizer`): open3d visualizer.
            points_colors (numpy.array): color of each points.
            pcd (:obj:`open3d.geometry.PointCloud`): point cloud. Default: None.
            bbox_color (tuple[float]): the color of bbox. Default: (0, 1, 0).
            points_in_box_color (tuple[float]):
                the color of points inside bbox3d. Default: (1, 0, 0).
            rot_axis (int): rotation axis of bbox. Default: 2.
            center_mode (bool): indicate the center of bbox is bottom center
                or gravity center. avaliable mode
                ['lidar_bottom', 'camera_bottom']. Default: 'lidar_bottom'.
            mode (str):  indicate type of the input points, avaliable mode
                ['xyz', 'xyzrgb']. Default: 'xyz'.
        """
        if isinstance(bbox3d, torch.Tensor):
            bbox3d = bbox3d.cpu().numpy()
        bbox3d = bbox3d.copy()

        self.bbox_count = len(bbox3d)

        # in_box_color = np.array(points_in_box_color)
        for i in range(len(bbox3d)):
            center = bbox3d[i, 0:3]
            dim = bbox3d[i, 3:6]
            yaw = np.zeros(3)
            yaw[rot_axis] = -bbox3d[i, 6]
            rot_mat = o3d.geometry.get_rotation_matrix_from_xyz(yaw)

            if center_mode == 'lidar_bottom':
                center[rot_axis] += dim[
                    rot_axis] / 2  # bottom center to gravity center
            elif center_mode == 'camera_bottom':
                center[rot_axis] -= dim[
                    rot_axis] / 2  # bottom center to gravity center
            box3d = o3d.geometry.OrientedBoundingBox(center, rot_mat, dim)

            line_set = o3d.geometry.LineSet.create_from_oriented_bounding_box(box3d)
            line_set.paint_uniform_color(bbox_color)

            # TODO: this is a hack, need to somehow update_geometry
            # if self.widget3d.scene.has_geometry("BBox_"+str(i)):
            #     self.widget3d.scene.remove_geometry("BBox_"+str(i))


            self.widget3d.scene.add_geometry("Bbox_"+str(i), line_set, self.bbox_mat)
        #     # draw bboxes on visualizer
        #     vis.add_geometry(line_set)

        #     # change the color of points which are in box
        #     if pcd is not None and mode == 'xyz':
        #         indices = box3d.get_point_indices_within_bounding_box(pcd.points)
        #         points_colors[indices] = in_box_color

        # # update points colors
        # if pcd is not None:
        #     pcd.colors = o3d.utility.Vector3dVector(points_colors)
        #     vis.update_geometry(pcd)

    def add_bboxes(self, bbox3d, bbox_color=None, points_in_box_color=None):
        """Add bounding box to visualizer.

        Args:
            bbox3d (numpy.array, shape=[M, 7]):
                3D bbox (x, y, z, dx, dy, dz, yaw) to be visualized.
                The 3d bbox is in mode of Box3DMode.DEPTH with
                gravity_center (please refer to core.structures.box_3d_mode).
            bbox_color (tuple[float]): the color of bbox. Defaule: None.
            points_in_box_color (tuple[float]): the color of points which
                are in bbox3d. Defaule: None.
        """
        # if bbox_color is None:
        #     bbox_color = self.bbox_color
        # if points_in_box_color is None:
        #     points_in_box_color = self.points_in_box_color
        # _draw_bboxes(bbox3d, self.o3d_visualizer, self.points_colors, self.pcd,
        #              bbox_color, points_in_box_color, self.rot_axis,
        #              self.center_mode, self.mode)
        self._draw_bboxes(bbox3d)

    def clean_widget_3d(self):
        # TODO: this is a hack, need to some how update_geometry
        self.widget3d.scene.remove_geometry("Stitched PC")
        for i in range(self.bbox_count):
            self.widget3d.scene.remove_geometry("Bbox_"+str(i))

    def project_pts_on_img(self,
                        points,
                       raw_img,
                       lidar2img_rt,
                       max_distance=70,
                       thickness=-1):
        """Project the 3D points cloud on 2D image.

        Args:
            points (numpy.array): 3D points cloud (x, y, z) to visualize.
            raw_img (numpy.array): The numpy array of image.
            lidar2img_rt (numpy.array, shape=[4, 4]): The projection matrix
                according to the camera intrinsic parameters.
            max_distance (float): the max distance of the points cloud.
                Default: 70.
            thickness (int, optional): The thickness of 2D points. Default: -1.
        """
        img = raw_img.copy()
        num_points = points.shape[0]
        pts_4d = np.concatenate([points[:, :3], np.ones((num_points, 1))], axis=-1)
        pts_2d = pts_4d @ lidar2img_rt.T

        # cam_points is Tensor of Nx4 whose last column is 1
        # transform camera coordinate to image coordinate
        pts_2d[:, 2] = np.clip(pts_2d[:, 2], a_min=1e-5, a_max=99999)
        pts_2d[:, 0] /= pts_2d[:, 2]
        pts_2d[:, 1] /= pts_2d[:, 2]

        fov_inds = ((pts_2d[:, 0] < img.shape[1])
                    & (pts_2d[:, 0] >= 0)
                    & (pts_2d[:, 1] < img.shape[0])
                    & (pts_2d[:, 1] >= 0))

        imgfov_pts_2d = pts_2d[fov_inds, :3]  # u, v, d

        cmap = plt.cm.get_cmap('hsv', 256)
        cmap = np.array([cmap(i) for i in range(256)])[:, :3] * 255
        for i in range(imgfov_pts_2d.shape[0]):
            depth = imgfov_pts_2d[i, 2]
            color = cmap[np.clip(int(max_distance * 10 / depth), 0, 255), :]
            cv2.circle(
                img,
                center=(int(np.round(imgfov_pts_2d[i, 0])),
                        int(np.round(imgfov_pts_2d[i, 1]))),
                radius=1,
                color=tuple(color),
                thickness=thickness,
            )
        # cv2.imshow('project_pts_img', img.astype(np.uint8))
        # cv2.waitKey(100)
        return img

    def process_img(self,
                       index,
                       key,
                       #dataset,
                       #out_dir,
                       #filename,
                       #show=False,
                       #img,
                       prepared_data,
                       #calib,
                       #img_metas,
                       gt_bboxes=None,
                       pred_bboxes=None,
                       gt_bbox_color=(0, 255, 0),  # (61, 102, 255) is Blue
                       pred_bbox_color=(241, 101, 72),
                       is_nus_mono=False):
        """
        Visualize 3D bboxes on 2D image by projection.
        source: show_proj_bbox_img and show_multi_modality_result
        """
        img_metas = prepared_data['img_metas']._data

        length_img_list = len(prepared_data['img']._data.numpy().shape) 
        if length_img_list > 3:
            img = prepared_data['img']._data.numpy()[index,:,:,:]
        else:
            img = prepared_data['img']._data.numpy()

        # need to transpose channel to first dim
        img = img.transpose(1,2,0)
        img = mmcv.bgr2rgb(img)
        self.img_dict[key]['img'] = img.copy()

        if gt_bboxes.tensor.shape[0] == 0:
            gt_bboxes = None
        if isinstance(gt_bboxes, DepthInstance3DBoxes):
            draw_bbox = draw_depth_bbox3d_on_img
            proj_mat = prepared_data['calib'][index] if length_img_list > 3 else prepared_data['calib']
        elif isinstance(gt_bboxes, LiDARInstance3DBoxes):
            draw_bbox = draw_lidar_bbox3d_on_img
            proj_mat = img_metas['lidar2img'][index] if length_img_list > 3 else img_metas['lidar2img']
        elif isinstance(gt_bboxes, CameraInstance3DBoxes):
            # TODO: remove the hack of box from NuScenesMonoDataset
            if is_nus_mono:
                from mmdet3d.core.bbox import mono_cam_box2vis
                gt_bboxes = mono_cam_box2vis(gt_bboxes)
            draw_bbox = draw_camera_bbox3d_on_img
            proj_mat = img_metas['cam_intrinsic'][index] if length_img_list > 3 else img_metas['cam_intrinsic']
        else:
            # can't project, just show img
            warnings.warn(
                f'unrecognized gt box type {type(gt_bboxes)}, only show image')
        img_bbox = img.copy()

        if gt_bboxes is not None:
            img_bbox = draw_bbox(
                gt_bboxes, img_bbox, proj_mat, img_metas, color=gt_bbox_color, thickness=3)
            
        if pred_bboxes is not None:
            img_bbox = draw_bbox(
                pred_bboxes, img_bbox.copy(), proj_mat, img_metas, color=pred_bbox_color, thickness=1)

        self.img_dict[key]['bbox'] = img_bbox.copy()

        ## TODO: incorporate odometry to lidar points to get their correct pose. also make the projection faster  
        # self.img_dict[key]['overlay'] = self.project_pts_on_img(self.pcd.point["points"].numpy(),
        #                     img.copy(),
        #                     proj_mat,
        #                     thickness=2)
        
    
    # Update the scene. This must be done on the UI thread.
    def update(self):
        # self.pcd_dict['LIDAR_TOP'] = prepared_data['points']._data.numpy()
        # self.widget3d.scene.remove_geometry("Stitched PC")
        self.clean_widget_3d()
        self.pcd.point["points"] = self._make_tcloud_array(self.prepared_data['points']._data.numpy()[:,:3])
        # self.widget3d.scene.add_geometry("Stitched PC", self.get_stitched_pcd(), lit)
        self.widget3d.scene.add_geometry("Stitched PC", self.pcd, self.pcd_mat)
        # TODO: add a button to select/deselct option of adding bbox, add as a if loop here.
        self.add_bboxes(bbox3d=self.gt_bboxes.tensor, bbox_color=(0, 0, 1))

        for index, key in enumerate(self.img_dict):
            self.process_img(index=index,
                            key=key,
                            prepared_data=self.prepared_data,
                            gt_bboxes=self.gt_bboxes)

            # TODO: add a button to select/deselct option of adding bbox, add as a if loop here.
            img = self.img_dict[key]['bbox'].copy()
            img = mmcv.imrescale(img, (self.img_rescaling_factor))
            # self.img_dict[key].update_image(o3d.geometry.Image(np.ascontiguousarray(img)))
            self.img_dict[key]['widget'].update_image(o3d.geometry.Image(np.ascontiguousarray(img)))

    def get_prepared_data(self, index):
        self.prepared_data = self.dataset.prepare_train_data(index)  # this already has loaded pc and img, img_meta
        self.gt_bboxes = self.dataset.get_ann_info(index)['gt_bboxes_3d']
        return True

    def _update_thread(self):
        # # This is NOT the UI thread, need to call post_to_main_thread() to update
        # # the scene or any part of the UI.
        # data_infos = self.dataset.data_infos
        
        # for idx, data_info in enumerate(track_iter_progress(data_infos)):
            # self.prepared_data = self.dataset.prepare_train_data(idx)  # this already has loaded pc and img, img_meta
            # self.gt_bboxes = self.dataset.get_ann_info(idx)['gt_bboxes_3d']

        while True:
            if self.pause:
                self.event.wait()
            
            # Prepare data, passing index is symbolic
            self.get_prepared_data(self.index)

            self.index += 1
            if self.index >= len(self.dataset.data_infos):
                self.is_done = True 
            
            if not self.is_done:
                gui.Application.instance.post_to_main_thread(
                    self.window, self.update)
            else:
                gui.Application.instance.quit()
                break
        # self.is_done = True

def main():
    app = o3d.visualization.gui.Application.instance
    app.initialize()

    win = GUIWindow()

    app.run()


if __name__ == "__main__":
    main()
