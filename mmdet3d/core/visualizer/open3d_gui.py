#!/usr/bin/env python
# Source: https://github.com/intel-isl/Open3D/blob/master/examples/python/gui/video.py
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

        self.is_done = False
        threading.Thread(target=self._update_thread).start()
    
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
        
        # self.metadata_panel.frame = gui.Rect(self.contentRect.x, self.widget3d.frame.get_bottom(),
        #                             self.widget3d.frame.get_right(),
        #                             contentRect.height)
        
    def init_data_str(self):
        data_infos = self.dataset.data_infos
        modality = self.dataset.modality

        if modality['use_lidar']:
            if data_infos[0]['lidar_path'] is not '':  # TODO: change logic for multi-lidar setup?
                # self.pcd_dict = {'LIDAR_TOP': o3d.geometry.PointCloud}
                # self.pcd_dict = {'LIDAR_TOP': np.zeros((100,4))}
                self.pcd = o3d.t.geometry.PointCloud(o3d.core.Device("CPU:0"))
                self.pcd.point["points"] = self._make_tcloud_array(np.random.rand(100,3)*10)

        if modality['use_camera']:
            if bool(data_infos[0]['cams']):
                self.img_dict = dict()
                for key, _ in data_infos[0]['cams'].items():
                    # TODO: this is a hack
                    self.img_dict[key] = gui.ImageWidget(o3d.geometry.Image(np.zeros((1,1,3)).astype(np.uint8)))    #gui.ImageWidget(np.array([]))
                
        if modality['use_radar']:  # TODO: this doesn't work yet.
            if bool(data_infos[0]['radars']):
                self.radar_dict = dict()
                for key, _ in data_infos[0]['radars'].items():
                    self.radar_dict[key] = o3d.geometry.PointCloud

        #TODO: its possible to initialize more structures based on metadata in self.dataset
        # eg HD Map, or based on class_names of annotated 3D bboxes, semantic segmentation etc 

    def init_user_interface(self):
        self.widget3d = gui.SceneWidget()
        self.widget3d.scene = rendering.Open3DScene(self.window.renderer)

        lit = rendering.Material()
        lit.shader = "defaultLit"
        
        # sample = self.get_stitched_pcd()
        # self.widget3d.scene.add_geometry("Stitched PC", sample, lit) # TODO: (',').join( list(data_infos[0]['cams'].keys()))        
        self.widget3d.scene.add_geometry("Stitched PC", self.pcd, lit)
        # bounds = self.widget3d.scene.bounding_box
        # self.widget3d.setup_camera(60.0, bounds, bounds.get_center()) #NOTE: set this carefully!
        self.widget3d.setup_camera(60.0, self.bounds, self.bounds.get_center()) 
        self.widget3d.scene.show_axes(True)
        self.window.add_child(self.widget3d)

        # em = self.window.theme.font_size
        margin = 0.25 * self.em
        
        self.img_panel = gui.VGrid(3, margin)
        # for key, value in self.img_dict.items():
        #     # img_tab = gui.Vert(margin, gui.Margins(margin))
        #     # img_tab.add_child(gui.Label(key))
        #     # img_tab.add_child(value)
        #     # self.img_panel.add_child(img_tab)
        #     self.img_panel.add_child(value)
        for index, _ in enumerate(self.panel_2_cam):
            self.img_panel.add_child(self.img_dict[self.panel_2_cam[index]])
        
        self.window.add_child(self.img_panel)

        #TODO: add for radar and HD Maps?

    def set_dataset_specific_prop(self):
        if self.dataset_type in ['NuScenesDataset']: #, 'LyftDataset']:
            self.panel_2_cam = {
                0: 'CAM_FRONT_LEFT',
                1: 'CAM_FRONT',
                2: 'CAM_FRONT_RIGHT',
                3: 'CAM_BACK_LEFT',
                4: 'CAM_BACK',
                5: 'CAM_BACK_RIGHT'
            }

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

    def _update_thread(self):
        # # This is NOT the UI thread, need to call post_to_main_thread() to update
        # # the scene or any part of the UI.
        # idx = 0
        # while not self.is_done:
        #     time.sleep(0.100)

        #     # Get the next frame, for instance, reading a frame from the camera.
        #     rgb_frame = self.rgb_images[idx]
        #     depth_frame = self.depth_images[idx]
        #     idx += 1
        #     if idx >= len(self.rgb_images):
        #         idx = 0

        #     # Update the images. This must be done on the UI thread.
        #     def update():
        #         self.rgb_widget.update_image(rgb_frame)
        #         self.depth_widget.update_image(depth_frame)
        #         self.widget3d.scene.set_background([1, 1, 1, 1], rgb_frame)

        #     if not self.is_done:
        #         gui.Application.instance.post_to_main_thread(
        #             self.window, update)

        data_infos = self.dataset.data_infos
        # dataset_type = self.dataset_type

        for idx, data_info in enumerate(track_iter_progress(data_infos)):
            # if dataset_type in ['KittiDataset', 'WaymoDataset']:
            #     data_path = data_info['point_cloud']['velodyne_path']
            # elif dataset_type in [
            #         'ScanNetDataset', 'SUNRGBDDataset', 'ScanNetSegDataset',
            #         'S3DISSegDataset'
            # ]:
            #     data_path = data_info['pts_path']
            # elif dataset_type in ['NuScenesDataset', 'LyftDataset']:
            #     data_path = data_info['lidar_path']
            # elif dataset_type in ['NuScenesMonoDataset']:
            #     data_path = data_info['file_name']
            # else:
            #     raise NotImplementedError(
            #         f'unsupported dataset type {dataset_type}')

            # file_name = osp.splitext(osp.basename(data_path))[0]

            example = self.dataset.prepare_train_data(idx)  # this already has loaded pc and img, img_meta
            gt_bboxes = self.dataset.get_ann_info(idx)['gt_bboxes_3d']
    
            # if vis_task in ['det', 'multi_modality-det']:
            #     # show 3D bboxes on 3D point clouds
            #     show_det_data(
            #         idx, dataset, args.output_dir, file_name, show=args.online)
            # if vis_task in ['multi_modality-det', 'mono-det']:
            #     # project 3D bboxes to 2D image
            #     show_proj_bbox_img(
            #         idx,
            #         dataset,
            #         args.output_dir,
            #         file_name,
            #         show=args.online,
            #         is_nus_mono=(dataset_type == 'NuScenesMonoDataset'))
            # elif vis_task in ['seg']:
            #     # show 3D segmentation mask on 3D point clouds
            #     show_seg_data(
            #         idx, dataset, args.output_dir, file_name, show=args.online)

            # Update the images. This must be done on the UI thread.
            def update():
                # self.pcd_dict['LIDAR_TOP'] = example['points']._data.numpy()
                # self.widget3d.scene.scene.update_geometry("Stitched PC", self.get_stitched_pcd(), o3d.visualization.rendering.Scene.UPDATE_POINTS_FLAG)
                # self.widget3d.scene.set_background([1, 1, 1, 1])
                # self.widget3d.force_redraw()
                self.widget3d.scene.remove_geometry("Stitched PC")
                lit = rendering.Material()
                lit.shader = "defaultLit"
                self.pcd.point["points"] = self._make_tcloud_array(example['points']._data.numpy()[:,:3])
                # self.widget3d.scene.add_geometry("Stitched PC", self.get_stitched_pcd(), lit)
                self.widget3d.scene.add_geometry("Stitched PC", self.pcd, lit)
                
                img_tensor = example['img']._data.numpy()
                if len(img_tensor.shape)>3: 
                    # for i in range(img_tensor.shape[0]):
                    for index, key in enumerate(self.img_dict):
                        img = img_tensor[index,:,:,:]
                        img = mmcv.imrescale(img.transpose(1,2,0), (0.25))    #TODO this needs to change if you put 3D bboxes to it
                        img = mmcv.bgr2rgb(img)
                        self.img_dict[key].update_image(o3d.geometry.Image(np.ascontiguousarray(img)))

            if not self.is_done:
                gui.Application.instance.post_to_main_thread(
                    self.window, update)

        self.is_done = True

def main():
    app = o3d.visualization.gui.Application.instance
    app.initialize()

    win = GUIWindow()

    app.run()


if __name__ == "__main__":
    main()
