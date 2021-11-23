# adopted from CenterPoint
import torch

from mmdet3d.core import bbox3d2result, merge_aug_bboxes_3d
from mmdet.models import DETECTORS
from .mvx_two_stage import MVXTwoStageDetector
from .. import builder
from mmdet3d.core import VoxelGenerator
from mmcv.runner import force_fp32

"""
The below code is credited from the paper
Wu, P., Chen, S., 
"MotionNet: Joint Perception and Motion Prediction for Autonomous Driving Based on Bird’s Eye View Maps", 
IEEE Conference on Computer Vision and Pattern Recognition (CVPR), DOI: 10.1109/​CVPR42600.2020.01140, June 2020, pp. 11382-11392.

URL:https://www.merl.com/research/license/MotionNet

Copyright (c) 2020 Mitsubishi Electric Research Laboratories (MERL). All rights reserved. The software, documentation and/or data in this file is provided on an "as is" basis, and MERL has no obligations to provide maintenance, support, updates, enhancements or modifications. MERL specifically disclaims any warranties, including, but not limited to, the implied warranties of merchantability and fitness for any particular purpose. In no event shall MERL be liable to any party for direct, indirect, special, incidental, or consequential damages, including lost profits, arising out of the use of this software and its documentation, even if MERL has been advised of the possibility of such damages. As more fully described in the license agreement that was required in order to download this software, documentation and/or data, permission to use, copy and modify this software without fee is granted, but only for educational, research and non-commercial purposes.
""" 

@DETECTORS.register_module()
class MotionNet(MVXTwoStageDetector):
    """MotionNet"""

    def __init__(self,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None,
                 motion_prediction_head=None):
        pts_voxel_layer_copy = pts_voxel_layer
        pts_voxel_layer = None
        super(MotionNet,
              self).__init__(pts_voxel_layer, pts_voxel_encoder,
                             pts_middle_encoder, pts_fusion_layer,
                             img_backbone, pts_backbone, img_neck, pts_neck,
                             pts_bbox_head, img_roi_head, img_rpn_head,
                             train_cfg, test_cfg, pretrained, init_cfg)

        if motion_prediction_head is not None:
            self.motion_pred = builder.build_head(motion_prediction_head)

        # override pts_voxel_layer
        self.pts_voxel_layer = VoxelGenerator(**pts_voxel_layer_copy)
    
    """
    def pts_voxel_layer(self, points):
        point_cloud_range = np.array(self.pts_voxel_layer["point_cloud_range"], dtype=np.float32)
        # [0, -40, -3, 70.4, 40, 1]
        voxel_size = np.array(self.pts_voxel_layer["voxel_size"], dtype=np.float32)
        grid_size = (point_cloud_range[3:] -
                     point_cloud_range[:3]) / voxel_size
        grid_size = np.round(grid_size).astype(np.int64)

        if not isinstance(voxel_size, np.ndarray):
            voxel_size = np.array(voxel_size, dtype=points.dtype)
        if not isinstance(coors_range, np.ndarray):
            coors_range = np.array(coors_range, dtype=points.dtype)
        voxelmap_shape = (coors_range[3:] - coors_range[:3]) / voxel_size
        voxelmap_shape = tuple(np.round(voxelmap_shape).astype(np.int32).tolist())
        if reverse_index:
            voxelmap_shape = voxelmap_shape[::-1]
        # don't create large array in jit(nopython=True) code.
        num_points_per_voxel = np.zeros(shape=(max_voxels, ), dtype=np.int32)
        coor_to_voxelidx = -np.ones(shape=voxelmap_shape, dtype=np.int32)
        voxels = np.zeros(
            shape=(max_voxels, max_points, points.shape[-1]), dtype=points.dtype)
        coors = np.zeros(shape=(max_voxels, 3), dtype=np.int32)
        if reverse_index:
            voxel_num = _points_to_voxel_reverse_kernel(
                points, voxel_size, coors_range, num_points_per_voxel,
                coor_to_voxelidx, voxels, coors, max_points, max_voxels)

        else:
            voxel_num = _points_to_voxel_kernel(points, voxel_size, coors_range,
                                                num_points_per_voxel,
                                                coor_to_voxelidx, voxels, coors,
                                                max_points, max_voxels)

        coors = coors[:voxel_num]
        voxels = voxels[:voxel_num]
        num_points_per_voxel = num_points_per_voxel[:voxel_num]

        return voxels, coors, num_points_per_voxel
    """
    
    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points):
        """Apply dynamic voxelization to points.

        Args:
            points (list[torch.Tensor]): Points of each sample.

        Returns:
            tuple[torch.Tensor]: Concatenated points, number of points
                per voxel, and coordinates.
        """
        voxels, coors, num_points = [], [], []
        for res in points:
            res_voxels, res_coors, res_num_points = self.pts_voxel_layer.generate(res.cpu().numpy())
            voxels.append(torch.from_numpy(res_voxels).to(res.device))
            coors.append(torch.from_numpy(res_coors).to(res.device))
            num_points.append(torch.from_numpy(res_num_points).to(res.device))
        voxels = torch.cat(voxels, dim=0)
        num_points = torch.cat(num_points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        return voxels, num_points, coors_batch


    def extract_pts_feat(self, pts, img_feats, img_metas):
        """Extract features of points."""
        # if not self.with_pts_bbox:
        #     return None
        voxels, num_points, coors = self.voxelize(pts)

        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors)
        batch_size = coors[-1, 0] + 1
        x = self.pts_middle_encoder(voxel_features, coors, batch_size)
        # x = self.pts_backbone(x)
        if self.with_pts_neck:
            x = self.pts_neck(x)
        return x

    def forward_pts_train(self,
                          pts_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          img_metas,
                          gt_bboxes_ignore=None):
        """Forward function for point cloud branch.

        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.

        Returns:
            dict: Losses of each branch.
        """
        outs = self.pts_bbox_head(pts_feats)
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
        losses = self.pts_bbox_head.loss(*loss_inputs)
        return losses

