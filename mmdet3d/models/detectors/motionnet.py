# adopted from CenterPoint
import torch

from mmdet3d.core import bbox3d2result, merge_aug_bboxes_3d
from mmdet.models import DETECTORS
from .mvx_two_stage import MVXTwoStageDetector
from .. import builder
from mmdet3d.core import VoxelGenerator
from mmcv.runner import force_fp32
from torch.nn import functional as F
from chamferdist import ChamferDistance
from mmdet3d.ops.render_pointcloud_in_image import map_pointcloud_to_image_torch, show_overlay

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
            self.motion_pred_head = builder.build_head(motion_prediction_head)

        # override pts_voxel_layer
        self.pts_voxel_layer = VoxelGenerator(**pts_voxel_layer_copy)

        self.chamfer_dist = ChamferDistance()

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
    
    def extract_pts_feat(self, voxels, num_points, coors, img_feats=None, img_metas=None):
        """Extract features of points."""
        # if not self.with_pts_bbox:
        #     return None
        # voxels, num_points, coors = self.voxelize(pts)

        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors)
        batch_size = coors[-1, 0] + 1
        x = self.pts_middle_encoder(voxel_features, coors, batch_size)
        # x = self.pts_backbone(x)
        if self.with_pts_neck:
            x1 = self.pts_neck(x)
        return x1
    
    def extract_feat(self, voxelized_pc, img, img_metas):
        """Extract features from images and points."""
        img_feats = self.extract_img_feat(img, img_metas)
        pts_feats = self.extract_pts_feat(voxels=voxelized_pc['voxels'], num_points=voxelized_pc['num_points'],
            coors=voxelized_pc['coors'], img_feats=img_feats, img_metas=img_metas)
        return (img_feats, pts_feats)

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
        voxels, coors, num_points, pt_in_voxel_mask = [], [], [], []
        for res in points:
            res_voxels, res_coors, res_num_points, res_pt_in_voxel_mask = self.pts_voxel_layer.generate(res.cpu().numpy())
            voxels.append(torch.from_numpy(res_voxels).to(res.device))
            coors.append(torch.from_numpy(res_coors).to(res.device))
            num_points.append(torch.from_numpy(res_num_points).to(res.device))
            pt_in_voxel_mask.append(torch.from_numpy(res_pt_in_voxel_mask).to(res.device))
        # voxels = torch.cat(voxels, dim=0)
        # num_points = torch.cat(num_points, dim=0)
        # pt_in_voxel_mask = torch.cat(pt_in_voxel_mask, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        # coors_batch = torch.cat(coors_batch, dim=0)
        return voxels, num_points, coors_batch, pt_in_voxel_mask

    def forward_train(self,
                      points=None,
                      points_next=None,
                      flow=None,
                      T_lidar2ego=None, 
                      T_lidar2cam=None,
                      img=None, 
                      cam_intrinsic=None,
                      cam_name=None,
                      delta_T=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      proposals=None,
                      gt_bboxes_ignore=None):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """
        voxels, num_points, coors, pt_in_voxel_mask = self.voxelize(points)
        voxelized_pc = dict(voxels=voxels, num_points=num_points, coors=coors, pt_in_voxel_mask=pt_in_voxel_mask)
        
        # voxels, num_points, coors = self.voxelize(points_next)
        # voxelized_pc_next = dict(voxels=voxels, num_points=num_points, coors=coors)
        
        # img_feats, pts_feats = self.extract_feat(
        #     voxelized_pc, img=img, img_metas=img_metas)
        img_feats, pts_feats = self.extract_feat(
            {key:torch.cat(value, dim=0) for (key,value) in voxelized_pc.items()}, img=img, img_metas=img_metas)
        
        losses = dict()
        if pts_feats is not None:
            losses_pts = self.forward_pts_train(pts_feats, points, points_next, voxelized_pc, #voxelized_pc_next,
                                                flow, T_lidar2ego, T_lidar2cam, img, cam_intrinsic,
                                                cam_name, delta_T,
                                                gt_bboxes_3d, gt_labels_3d, img_metas,
                                                gt_bboxes_ignore)
            losses.update(losses_pts)
        if img_feats is not None:
            losses_img = self.forward_img_train(
                img_feats,
                img_metas=img_metas,
                gt_bboxes=gt_bboxes,
                gt_labels=gt_labels,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposals=proposals)
            losses.update(losses_img)
        return losses

    def forward_pts_train(self,
                          pts_feats,
                          points,
                          points_next,
                          voxelized_pc,
                        #   voxelized_pc_next,
                          flow,
                          T_lidar2ego, 
                          T_lidar2cam,
                          img, 
                          cam_intrinsic,
                          cam_name,
                          delta_T,
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
        # initialize losses
        chamfer_loss = torch.Tensor([0]).to(pts_feats.device)
        # smooth_loss = torch.Tensor([0]).to(pts_feats.device)

        # get predicted motion field
        pred_motion = self.motion_pred_head(pts_feats)
        # import matplotlib.pyplot as plt
        # m = pred_motion[0,:,:,:]
        # m = pred_motion[0,:,:,:]
        # from tools.optical_flow.flowlib import flow_to_image
        # m = m.detach().cpu().numpy()
        # img = flow_to_image(m)
        # plt.imshow(img); plt.show()

        batch_size = pred_motion.size(0)
        coor = voxelized_pc['coors']

         
        # for batch_idx in range(batch_size):
        #     batch_mask = torch.where(coor[:,0]==batch_idx)[0]
        #     p = coor[batch_mask,2] * self.pts_voxel_layer.grid_size[0] + coor[batch_mask,2]
        #     if batch_idx==0:
        #         pind = p
        #     else:
        #         pind = torch.cat((pind, torch.add(p, torch.tensor([self.pts_voxel_layer.grid_size[0]* self.pts_voxel_layer.grid_size[1]]).to(p.device))))
        # pred_flat = pred_motion.view(-1,2) #pred_motion[batch_idx,:,:,:].view(-1,2)
        # delta_voxel = pred_flat[pind.long()] 
        # pred_voxel = voxelized_pc['voxels'].clone()
        
        # # apply predicted motion to pc at t to get a predicted pc at t+1
        # pred_voxel[:,:,:2] = torch.add(pred_voxel[:,:,:2], delta_voxel[:,None,:]) # but this doesn't solve problem! all zeros have motion added to them! 

        # # extract pc from voxelized_pc, calculate chamfer loss between predicted pc and actual pc at t+1
        # non_zero_mask = voxelized_pc['voxels'].view(-1,5).sum(dim=1) != 0
        # v = pred_voxel.view(-1,5)[non_zero_mask,:]

        # v_next = voxelized_pc_next['voxels'].view(-1,5)
        # v_next = v_next[v_next.sum(dim=1) != 0]

        # # # to visualize
        # # v_original = voxelized_pc['voxels'].clone().view(-1,5)
        # # v_original = v_original[v_original.sum(dim=1) != 0]
        # # import open3d as o3d
        # # pcd = o3d.geometry.PointCloud(); pcd.points = o3d.utility.Vector3dVector(v_original.cpu().numpy()[:,:3])
        # # pcd_next = o3d.geometry.PointCloud(); pcd_next.points = o3d.utility.Vector3dVector(v_next.cpu().numpy()[:,:3]); pcd_next.paint_uniform_color([1, 0.706, 0])
        # # pcd_pred = o3d.geometry.PointCloud(); pcd_pred.points = o3d.utility.Vector3dVector(v.detach().cpu().numpy()[:,:3]); pcd_pred.paint_uniform_color([1, 0, 0])
        # # o3d.visualization.draw_geometries([pcd, pcd_pred])

        # chamfer_loss = self.chamfer_dist(v[None,:,:], v_next[None,:,:], bidirectional=True)
        # smooth_loss = torch.mean(torch.abs(pred_motion[:, 1:] - pred_motion[:, :-1])) + \
        #               torch.mean(torch.abs(pred_motion[:, :, 1:] - pred_motion[:, :, :-1]))

        
        original_voxel = voxelized_pc['voxels'].clone()
            
        for batch_idx in range(batch_size):
            batch_mask = torch.where(coor[:,0]==batch_idx)[0]
            pred_voxel=voxelized_pc['voxels'].clone()
            
            # get original point cloud
            v_original = voxelized_pc_next['voxels'][batch_mask].view(-1,5)
            v_original=v_original[v_original.sum(dim=1) != 0]
            
            # apply predicted motion to voxelized pc at t to get a predicted pc at t+1
            pred_voxel[batch_mask, :, :2] = torch.add(original_voxel[batch_mask,:,:2], pred_motion[batch_idx, coor[batch_mask,3].long(), coor[batch_mask,2].long(), None, :])
            # extract pc from voxelized_pc
            non_zero_mask = original_voxel[batch_mask].view(-1,5).sum(dim=1) != 0
            v_pred = pred_voxel[batch_mask].view(-1,5)[non_zero_mask,:]
            # NOTE or check: v_pred.size()[0] == torch.sum(voxelized_pc['num_points'][batch_mask])

            batch_mask_next = torch.where(voxelized_pc_next['coors'][:,0]==batch_idx)[0]
            v_next = voxelized_pc_next['voxels'][batch_mask_next].view(-1,5)
            v_next=v_next[v_next.sum(dim=1) != 0]
            # NOTE or check: v_next.size()[0] == torch.sum(voxelized_pc_next['num_points'][batch_mask_next])
            
            # optical flow loss
            for camid in range(len(T_lidar2cam[0])):
                print('')
                pt_original, mask_original, color_original, _ = map_pointcloud_to_image_torch(v_original[:,:3], 
                                                                                img.squeeze()[camid],
                                                                                T_lidar2cam[0][camid],
                                                                                cam_intrinsic[0][camid])
                # show_overlay(pt_original.cpu().numpy(), mask_original.cpu().numpy(), 
                # img.squeeze()[camid].permute(1,2,0).cpu().numpy(), color_original.cpu().numpy(), title='original')
            # calculate chamfer loss between predicted pc and actual pc at t+1
            chamfer_loss += self.chamfer_dist(v_pred[None,:,:], v_next[None,:,:], bidirectional=True)

        # average for batch size
        chamfer_loss /= batch_size
        smooth_loss = torch.mean(torch.abs(2*pred_motion[:, 1:-1] -  pred_motion[:, :-2]- pred_motion[:, 2:])) + \
             torch.mean(torch.abs(2*pred_motion[:, :, 1:-1] -  pred_motion[:, :, :-2]- pred_motion[:, :, 2:]))
        

        losses = dict(
            chamfer_loss=chamfer_loss,
            smooth_loss =smooth_loss
            )
        # outs = self.pts_bbox_head(pts_feats)
        # loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
        # losses = self.pts_bbox_head.loss(*loss_inputs)
        return losses

