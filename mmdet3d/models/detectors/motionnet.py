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
from tools.optical_flow.flowlib import flow_to_image, dispOpticalFlow

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
        if ((torch.isnan(voxels).all()) or (torch.isnan(num_points).all()) or (torch.isnan(coors).all())):
            print('here, 33333333333333333333')
        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors)
        if ((torch.isnan(voxel_features)).all()):
            print('here, 4444444444444444')
        batch_size = coors[-1, 0] + 1
        x = self.pts_middle_encoder(voxel_features, coors, batch_size)

        if ((torch.isnan(x)).all()):
            print('here, 55555555555555')
        # x = self.pts_backbone(x)
        if self.with_pts_neck:
            x1 = self.pts_neck(x)
            if ((torch.isnan(x1)).all()):
                print('here, 666666666666666666')
        return x1
    
    def extract_feat(self, voxelized_pc, img, img_metas):
        """Extract features from images and points."""
        img_feats = self.extract_img_feat(img, img_metas)
        pts_feats = self.extract_pts_feat(voxels=voxelized_pc['voxels'], num_points=voxelized_pc['num_points'],
            coors=voxelized_pc['coors'], img_feats=img_feats, img_metas=img_metas)
        if ((torch.isnan(pts_feats)).all()):
            print('here, 11111111111111111111')
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
            if ((torch.isnan(torch.from_numpy(res_voxels).to(res.device))).all()):
                print("here, 8888888888888888")
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
                    #   flow_img=None,
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
        for res in points:
            if ((torch.isnan(res)).all()):
                print("here, 777777777777777777777777777777")
                return None
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
                                                cam_name, delta_T, #flow_img,
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
                        #   flow_img,
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
        optical_flow_loss = torch.Tensor([0]).to(pts_feats.device)

        # get predicted motion field
        pred_motion = self.motion_pred_head(pts_feats)
        if ((torch.isnan(pred_motion)).all()):
            print('here, 222222222222222222222222')
        # import matplotlib.pyplot as plt
        # m = pred_motion[0,:,:,:]
        # m = pred_motion[0,:,:,:]
        # from tools.optical_flow.flowlib import flow_to_image
        # m = m.detach().cpu().numpy()
        # img = flow_to_image(m)
        # plt.imshow(img); plt.show()

        batch_size = pred_motion.size(0)
        # coor = voxelized_pc['coors']
        # pt_in_voxel_mask_batch = voxelized_pc['pt_in_voxel_mask']

        # original_voxel = voxelized_pc['voxels'].clone()
        original_voxel = [v.clone() for v in voxelized_pc['voxels']]
            
        for batch_idx in range(batch_size):
            # batch_mask = torch.where(coor[:,0]==batch_idx)[0]
            # pred_voxel=voxelized_pc['voxels'].clone()
            
            # pt_in_voxel_mask = pt_in_voxel_mask_batch[batch_idx]
            # # # get original point cloud
            # # v_original = voxelized_pc['voxels'][batch_mask].view(-1,5)
            # # v_original=v_original[v_original.sum(dim=1) != 0]
            
            # # apply predicted motion to voxelized pc at t to get a predicted pc at t+1
            # pred_voxel[batch_mask, :, :2] = torch.add(original_voxel[batch_mask,:,:2], pred_motion[batch_idx, coor[batch_mask,3].long(), coor[batch_mask,2].long(), None, :])
            # # extract pc from voxelized_pc
            # non_zero_mask = original_voxel[batch_mask].view(-1,5).sum(dim=1) != 0
            # points_pred = pred_voxel[batch_mask].view(-1,5)[non_zero_mask,:]
            # # NOTE or check: points_pred.size()[0] == torch.sum(voxelized_pc['num_points'][batch_mask])

            # # batch_mask_next = torch.where(voxelized_pc_next['coors'][:,0]==batch_idx)[0]
            # # v_next = voxelized_pc_next['voxels'][batch_mask_next].view(-1,5)
            # # v_next=v_next[v_next.sum(dim=1) != 0]
            # # # NOTE or check: v_next.size()[0] == torch.sum(voxelized_pc_next['num_points'][batch_mask_next])
            
            coor = voxelized_pc['coors'][batch_idx]
            pt_in_voxel_mask = voxelized_pc['pt_in_voxel_mask'][batch_idx]
            
            # pred_voxel = [v.clone() for v in voxelized_pc['voxels']]
            pred_voxel = voxelized_pc['voxels'][batch_idx].clone()

            # apply predicted motion to voxelized pc at t to get a predicted pc at t+1
            pred_voxel[:,:,:2] = torch.add(original_voxel[batch_idx][:,:,:2], pred_motion[batch_idx, coor[:,3].long(), coor[:,2].long(), None, :])
            # extract pc from pred_voxel
            # non_zero_mask = original_voxel[batch_idx].view(-1,5).sum(dim=1) != 0
            non_zero_mask = (original_voxel[batch_idx].view(-1,5) != 0).sum(dim=1) >0
            points_pred = pred_voxel.view(-1,5)[non_zero_mask,:]

            delta = delta_T[batch_idx]
            lidar2ego = T_lidar2ego[batch_idx]
            # optical flow loss
            F_t = flow[batch_idx][pt_in_voxel_mask,:].float().T
                
            for camid in range(len(T_lidar2cam[0])):
                # set filtering dist to remove ego vehicle points for CAM_FRONT projection  
                min_dist = 8.0 if camid==0 else 1.0
                pt_original, mask_original, color_original, _ = map_pointcloud_to_image_torch(points[batch_idx][pt_in_voxel_mask,:3], 
                                                                                img[batch_idx][camid],
                                                                                T_lidar2cam[batch_idx][camid],
                                                                                cam_intrinsic[batch_idx][camid])
                # show_overlay(pt_original.cpu().numpy(), mask_original.cpu().numpy(), 
                # img[batch_idx][camid].permute(1,2,0).cpu().numpy(), color_original.cpu().numpy(), title='original')
                pt_pred, mask_pred, color_pred, _ = map_pointcloud_to_image_torch(points_pred[:,:3], 
                                                                                img[batch_idx][camid],
                                                                                T_lidar2cam[batch_idx][camid],
                                                                                cam_intrinsic[batch_idx][camid])
                # show_overlay(pt_pred.detach().cpu().numpy(), mask_pred.cpu().numpy(), 
                # img[batch_idx][camid].permute(1,2,0).cpu().numpy(), color_pred.detach().cpu().numpy(), title='pred')
                
                T_lidar2nextcam = T_lidar2cam[batch_idx][camid] @ torch.inverse(lidar2ego) @ delta @ lidar2ego
                pt_shifted, mask_shifted, color_shifted, _ = map_pointcloud_to_image_torch(points[batch_idx][pt_in_voxel_mask,:3], 
                                                                                img[batch_idx][camid],
                                                                                T_lidar2nextcam,
                                                                                cam_intrinsic[batch_idx][camid],
                                                                                min_dist=min_dist)
                # show_overlay(pt_shifted.detach().cpu().numpy(), mask_shifted.cpu().numpy(), 
                # img[batch_idx][camid].permute(1,2,0).cpu().numpy(), color_shifted.detach().cpu().numpy(), title='next')
                F_t_pred  = torch.zeros_like(pt_original)                
                F_t_pred[:, mask_original&mask_pred] = pt_original[:, mask_original&mask_pred] - pt_pred[:, mask_original&mask_pred]

                F_t_ego = torch.zeros_like(pt_original)
                F_t_obj = torch.zeros_like(pt_original)
                
                pt_original_round = torch.round(pt_original).long()

                # F_t = torch.zeros_like(pt_original)
                # F_t_img = flow_img[batch_idx][camid]
                # F_t[:, mask_original&mask_shifted] = F_t_img[pt_original_round[1, mask_original&mask_shifted], pt_original_round[0, mask_original&mask_shifted]].T
                
                F_t_ego[:, mask_original&mask_shifted] = pt_shifted[:, mask_original&mask_shifted] - pt_original[:, mask_original&mask_shifted]
                F_t_obj[:, mask_original&mask_shifted] = F_t[:, mask_original&mask_shifted] + F_t_ego[:, mask_original&mask_shifted]

                mask_common = mask_original&mask_pred&mask_shifted
                optical_flow_loss += torch.sum(torch.abs(F_t_pred[:,mask_common] + F_t_obj[:,mask_common]))

                # # Visualize optical flow
                # im = img[batch_idx][camid].permute(1,2,0)
                # flo_obj = torch.zeros_like(im[:,:,:2], dtype=torch.float)            
                # flo_obj[pt_original_round[1, mask_original&mask_shifted], pt_original_round[0, mask_original&mask_shifted]] =  F_t_obj[:, mask_original&mask_shifted].T
                # dispOpticalFlow(im.cpu().numpy().copy(), flo_obj.cpu().numpy(), Divisor=4)
                
                # flo_ego = torch.zeros_like(im[:,:,:2], dtype=torch.float)            
                # flo_ego[pt_original_round[1, mask_original&mask_shifted], pt_original_round[0, mask_original&mask_shifted]] =  F_t_ego[:, mask_original&mask_shifted].T
                # dispOpticalFlow(im.cpu().numpy().copy(), flo_ego.cpu().numpy(), Divisor=4)
                
                # flo = torch.zeros_like(im[:,:,:2], dtype=torch.float)            
                # flo[pt_original_round[1, mask_original&mask_shifted], pt_original_round[0, mask_original&mask_shifted]] =  F_t[:, mask_original&mask_shifted].T
                # dispOpticalFlow(im.cpu().numpy().copy(), flo.cpu().numpy(), Divisor=4)
                
                # # dispOpticalFlow_torch(img[batch_idx][camid].permute(1,2,0), F_t_obj, pt_original_round, mask_original&mask_shifted)
                # dispOpticalFlow(im.cpu().numpy().copy(), F_t_img.cpu().numpy(), Divisor=20)

            # average for all cameras
            optical_flow_loss /= len(T_lidar2cam[0])
            
            # calculate chamfer loss between predicted pc and actual pc at t+1
            chamfer_loss += self.chamfer_dist(points_pred[None,:,:], points_next[batch_idx][None,:,:], bidirectional=True)

        # average for batch size
        chamfer_loss /= batch_size
        optical_flow_loss /= batch_size

        # # change to follow the same scale
        # chamfer_loss /= 1e5
        # optical_flow_loss /= (1e5)
        
        smooth_loss = torch.mean(torch.abs(2*pred_motion[:, 1:-1] -  pred_motion[:, :-2]- pred_motion[:, 2:])) + \
             torch.mean(torch.abs(2*pred_motion[:, :, 1:-1] -  pred_motion[:, :, :-2]- pred_motion[:, :, 2:]))
        

        losses = dict(
            chamfer_loss=chamfer_loss,
            smooth_loss =smooth_loss,
            optical_flow_loss=optical_flow_loss
            )
        
        return losses

def dispOpticalFlow_torch(im, F, pt_original_round, mask, divisor=2):
    # im = img[batch_idx][camid].permute(1,2,0)
    flo = torch.zeros_like(im[:,:,:2], dtype=torch.float)            
    flo[pt_original_round[1, mask], pt_original_round[0, mask]] =  F[:, mask].T
    dispOpticalFlow(im.cpu().numpy().copy(), flo.cpu().numpy().copy(), Divisor=divisor)
                