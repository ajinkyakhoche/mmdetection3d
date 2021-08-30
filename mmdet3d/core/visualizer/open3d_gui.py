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
    def __init__(self, dataset_type, modality, point_cloud_range, monitor_width=1920, monitor_height=1080):

        # self.dataset = dataset
        # self.dataset_type = dataset_type
        self.modality = modality
        self.dataset_type = dataset_type
        self.point_cloud_range = point_cloud_range
        
        self.pause = True
        self.index = 0
        self.is_done = False
        # threading.Thread(target=self._update_thread).start()
        self.event = threading.Event()
        self._lock = threading.Lock()
        self._started = False
        self._start()

    def _start(self):
        if not self._started:
            self._thread = threading.Thread(target=self._thread_main)
            self._thread.start()
            self._started = True

    def _thread_main(self):
        app = gui.Application.instance
        app.initialize()

        # initialize an app instance of gui
        # gui.Application.instance.initialize()
        self.window = gui.Application.instance.create_window(
            "GUI - Visualizer for "+ self.dataset_type, 1920, 1208)  # TODO: try to get monitor res?
        self.window.set_on_layout(self._on_layout)
        self.window.set_on_close(self._on_close)

        self.em = self.window.theme.font_size
        self.margin = 0.25 * self.em

        # self.bounds = o3d.geometry.AxisAlignedBoundingBox(
        #     point_cloud_range[:3], point_cloud_range[3:])
        self._set_dataset_specific_prop(self.dataset_type)

        if self.modality['use_lidar']:
            # TODO: change logic for multi-lidar setup?
            self.pcd = o3d.t.geometry.PointCloud(o3d.core.Device("CPU:0"))
            self.pcd_mat = rendering.Material()
            self.pcd_mat.shader = "defaultLit"
            # Add point cloud widget to gui
            self.widget3d = gui.SceneWidget()
            self.widget3d.scene = rendering.Open3DScene(self.window.renderer)
            self.widget3d.scene.add_geometry(
                "Stitched PC", self.pcd, self.pcd_mat)
            # NOTE: set bounds carefully! else the visualizer zooms in excessively
            bounds = o3d.geometry.AxisAlignedBoundingBox(
                self.point_cloud_range[:3], self.point_cloud_range[3:])
            self.widget3d.setup_camera(60.0, bounds, bounds.get_center())
            self.widget3d.scene.show_axes(True)
            self.window.add_child(self.widget3d)

            # 3D bboxes
            self.bbox_mat = rendering.Material()
            self.bbox_mat.shader = "unlitLine"
            self.bbox_mat.line_width = 2 * self.window.scaling
            self.bbox_count_gt = 0
            self.bbox_count_pred = 0
            self.bbox_gt_list = []
            self.bbox_pred_list = []
            

        if self.modality['use_camera']:
            self.img_dict = dict()
            for cam in self.cam_names:      # TODO: what if only some cameras need to be displayed?
                self.img_dict[cam] = {
                    'img': np.zeros((1, 1, 3)).astype(np.uint8),
                    'overlay': np.zeros((1, 1, 3)).astype(np.uint8),
                    'bbox': np.zeros((1, 1, 3)).astype(np.uint8),
                    'widget': gui.ImageWidget(o3d.geometry.Image())
                    # TODO: can add more attributes for sem seg, 2d bbox etc..
                }
            # Add img widget to gui
            self.img_panel = gui.VGrid(self.img_panel_cols, self.margin)
            for index, _ in enumerate(self.panel_2_cam):
                self.img_panel.add_child(
                    self.img_dict[self.panel_2_cam[index]]['widget'])
            self.window.add_child(self.img_panel)

        # TODO: this doesn't work yet.
        if 'use_radar' in self.modality and self.modality['use_radar']:
            self.radar_dict = dict()
            # for key, _ in data_infos[0]['radars'].items():
            # for key, _ in self.data_info_radar.items():
            #     self.radar_dict[key] = o3d.geometry.PointCloud

        # TODO: its possible to initialize more structures
        # eg HD Map or class_names of annotated 3D bboxes, semantic segmentation etc

        # self.init_data_str()
        self._init_user_interface()

        app.run()


    def _set_dataset_specific_prop(self, dataset_type):
        if dataset_type in ['NuScenesDataset', 'LyftDataset']:  # TODO: ArgoDataset
            # order in which imgs/cameras are arranged internally in the dataset
            self.cam_names = [
                'CAM_FRONT',
                'CAM_FRONT_RIGHT',
                'CAM_FRONT_LEFT',
                'CAM_BACK',
                'CAM_BACK_LEFT',
                'CAM_BACK_RIGHT'
            ]
            # order in which imgs/cameras should be displayed in the GUI
            self.panel_2_cam = {
                0: 'CAM_FRONT_LEFT',
                1: 'CAM_FRONT',
                2: 'CAM_FRONT_RIGHT',
                3: 'CAM_BACK_LEFT',
                4: 'CAM_BACK',
                5: 'CAM_BACK_RIGHT'
            }
            self.img_panel_cols = 3     # number of columns in the image panel
            # height, width, # of channels
            # self.img_size = mmcv.imread(
            #     self.dataset.data_infos[0]['cams'][self.panel_2_cam[0]]['data_path']).shape

        elif dataset_type in ['KittiDataset']:
            self.cam_names = [   # This is a hack
                'CAM_FRONT_LEFT',
                # 'CAM_FRONT_RIGHT'
            ]
            self.panel_2_cam = {
                0: 'CAM_FRONT_LEFT',
                # 1: 'CAM_FRONT_RIGHT',  # TODO: mmdetection doesn't read the right image
            }
            self.img_panel_cols = 1     # number of columns in the image panel
            # height, width, # of channels
            # self.img_size = mmcv.imread(osp.join(
            #     self.dataset.data_root, self.dataset.data_infos[0]['image']['image_path'])).shape

            # TODO: data_path and file_path can be set here too

    def _on_layout(self, layout_context):
        contentRect = self.window.content_rect
        img_panel_width = 75 * self.em  # where 1 em = 16 px
        metadata_panel_height = 7.5 * self.em

        if self.modality['use_lidar']:
            self.widget3d.frame = gui.Rect(contentRect.x, contentRect.y,
                                           contentRect.width - img_panel_width,
                                           contentRect.height - metadata_panel_height)
        if self.modality['use_camera']:
            self.img_panel.frame = gui.Rect(self.widget3d.frame.get_right(),
                                            contentRect.y, img_panel_width,
                                            contentRect.height)
            # factor by which to rescale images to fit the image panel
            # self.img_rescaling_factor = img_panel_width / \
            #     self.img_panel_cols / self.img_size[1]
            self.img_rescaling_factor = 0.3 # TODO: this is a hack

        self.metadata_panel.frame = gui.Rect(contentRect.x, self.widget3d.frame.get_bottom(),
                                             self.widget3d.frame.get_right(),
                                             metadata_panel_height)

    def _init_user_interface(self):
        self.metadata_panel = gui.VGrid(3, self.margin)

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
        # TODO: add for radar and HD Maps?

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
            self.index -= 1
            self.get_prepared_data(self.index)
            if not self.is_done:
                gui.Application.instance.post_to_main_thread(
                    self.window, self.update)
        else:
            print('Press Pause first')
        return
    
    def _on_close(self):
        gui.Application.instance.quit()
        self.is_done = True
        return True  # False would cancel the close




    def update_pcd(self, attr):
        self.pcd.point["points"] = self._make_tcloud_array(attr)
        # Update scalar values, source: ml3d/vis/visualizer.py
        if attr is not None:
            if len(attr.shape) == 1:
                scalar = attr
            else:
                # channel = max(0, self._colormap_channel.selected_index)
                channel = 0  # TODO: set this from GUI
                scalar = attr[:, channel]
        else:
            # shape = [len(tcloud.point["points"].numpy())]
            shape = [len(self.pcd.point["points"].numpy())]
            scalar = np.zeros(shape, dtype='float32')
        # tcloud.point["__visualization_scalar"] = Visualizer._make_tcloud_array(
        #     scalar)
        self.pcd.point["__visualization_scalar"] = self._make_tcloud_array(
            scalar)

        # flag |= rendering.Scene.UPDATE_UV0_FLAG

        # Update RGB values
        if attr is not None and (len(attr.shape) == 2 and attr.shape[1] >= 3):
            # max_val = float(self._rgb_combo.selected_text)
            max_val = 255   # TODO: set this from GUI
            if max_val <= 0:
                max_val = 255.0
            colors = attr[:, [0, 1, 2]] * (1.0 / max_val)
            # tcloud.point["colors"] = Visualizer._make_tcloud_array(colors)
            self.pcd.point["colors"] = self._make_tcloud_array(colors)
            # flag |= rendering.Scene.UPDATE_COLORS_FLAG
        return True

    def update_img(self, 
                    key, 
                    img=None, 
                    img_bbox_3d=None, 
                    img_overlay=None):
        self.img_dict[key]['img'] = img
        self.img_dict[key]['bbox'] = img_bbox_3d
        self.img_dict[key]['overlay'] = img_overlay
        return True
    
    def update_bbox_3d(self,
                gt=[],
                pred=[]):
        with self._lock:
            self.bbox_gt_list = gt
            self.bbox_count_gt = len(gt)
            self.bbox_pred_list = pred
            self.bbox_count_pred = len(pred)
        return True

    def update_gui(self):
        if self.index != 0:
            self.clean_widget_3d()
        
        self.widget3d.scene.add_geometry("Stitched PC", self.pcd, self.pcd_mat)
        
        for idx, key in enumerate(self.cam_names):
            # TODO: add a button to select/deselct option of adding bbox, add as a if loop here.
            img = self.img_dict[key]['bbox'].copy()
            img = mmcv.imrescale(img, (self.img_rescaling_factor))
            # self.img_dict[key].update_image(o3d.geometry.Image(np.ascontiguousarray(img)))
            self.img_dict[key]['widget'].update_image(
                o3d.geometry.Image(np.ascontiguousarray(img)))
        
        for i, line_set in enumerate(self.bbox_gt_list):
            self.widget3d.scene.add_geometry(
                "Bbox_gt_"+str(i), line_set, self.bbox_mat)
        # for i, line_set in enumerate(self.bbox_pred_list):
        #     self.widget3d.scene.add_geometry(
        #         "Bbox_pred_"+str(i), line_set, self.bbox_mat)
        
        self.index += 1

    def clean_widget_3d(self):
        # TODO: this is a hack, need to some how update_geometry
        self.widget3d.scene.remove_geometry("Stitched PC")
        with self._lock:
            for i in range(self.bbox_count_gt):
            # for i in range(len(self.bbox_gt_list)):     
                self.widget3d.scene.remove_geometry("Bbox_gt_"+str(i))
            for i in range(self.bbox_count_pred):
            # for i in range(len(self.bbox_pred_list)):     
                self.widget3d.scene.remove_geometry("Bbox_pred_"+str(i))

    def _make_tcloud_array(self, np_array, copy=False):
        if copy or not np_array.data.c_contiguous:
            return o3d.core.Tensor(np_array)
        else:
            return o3d.core.Tensor.from_numpy(np_array)

    # def get_stitched_pcd(self):
    #     # TODO: this needs to be modified for multi-lidar setup
    #     # return self.pcd_dict['LIDAR_TOP']
    #     a = o3d.t.geometry.PointCloud(o3d.core.Device("CPU:0"))
    #     a.point["points"] = o3d.core.Tensor.from_numpy(
    #         np.ascontiguousarray(self.pcd_dict['LIDAR_TOP'][:, :3]))
    #     return a
