import mmcv
import numpy as np
import trimesh
from os import path as osp

from .image_vis import (draw_camera_bbox3d_on_img, draw_depth_bbox3d_on_img,
                        draw_lidar_bbox3d_on_img)
import torch
try:
    import open3d as o3d
    from open3d import geometry
except ImportError:
    raise ImportError(
        'Please run "pip install open3d" to install open3d first.')

def _write_obj(points, out_filename):
    """Write points into ``obj`` format for meshlab visualization.

    Args:
        points (np.ndarray): Points in shape (N, dim).
        out_filename (str): Filename to be saved.
    """
    N = points.shape[0]
    fout = open(out_filename, 'w')
    for i in range(N):
        if points.shape[1] == 6:
            c = points[i, 3:].astype(int)
            fout.write(
                'v %f %f %f %d %d %d\n' %
                (points[i, 0], points[i, 1], points[i, 2], c[0], c[1], c[2]))

        else:
            fout.write('v %f %f %f\n' %
                       (points[i, 0], points[i, 1], points[i, 2]))
    fout.close()


def _write_oriented_bbox(scene_bbox, out_filename):
    """Export oriented (around Z axis) scene bbox to meshes.

    Args:
        scene_bbox(list[ndarray] or ndarray): xyz pos of center and
            3 lengths (dx,dy,dz) and heading angle around Z axis.
            Y forward, X right, Z upward. heading angle of positive X is 0,
            heading angle of positive Y is 90 degrees.
        out_filename(str): Filename.
    """

    def heading2rotmat(heading_angle):
        rotmat = np.zeros((3, 3))
        rotmat[2, 2] = 1
        cosval = np.cos(heading_angle)
        sinval = np.sin(heading_angle)
        rotmat[0:2, 0:2] = np.array([[cosval, -sinval], [sinval, cosval]])
        return rotmat

    def convert_oriented_box_to_trimesh_fmt(box):
        ctr = box[:3]
        lengths = box[3:6]
        trns = np.eye(4)
        trns[0:3, 3] = ctr
        trns[3, 3] = 1.0
        trns[0:3, 0:3] = heading2rotmat(box[6])
        box_trimesh_fmt = trimesh.creation.box(lengths, trns)
        return box_trimesh_fmt

    if len(scene_bbox) == 0:
        scene_bbox = np.zeros((1, 7))
    scene = trimesh.scene.Scene()
    for box in scene_bbox:
        scene.add_geometry(convert_oriented_box_to_trimesh_fmt(box))

    mesh_list = trimesh.util.concatenate(scene.dump())
    # save to obj file
    trimesh.io.export.export_mesh(mesh_list, out_filename, file_type='obj')

    return


def _draw_bboxes(bbox3d,
                 #vis,
                 #points_colors,
                 #pcd=None,
                 bbox_color=(0, 1.0, 0),
                 #points_in_box_color=(1, 0, 0),
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

    line_set_list = []

    # in_box_color = np.array(points_in_box_color)
    for i in range(len(bbox3d)):
        center = bbox3d[i, 0:3]
        dim = bbox3d[i, 3:6]
        yaw = np.zeros(3)
        yaw[rot_axis] = -bbox3d[i, 6]
        rot_mat = geometry.get_rotation_matrix_from_xyz(yaw)

        if center_mode == 'lidar_bottom':
            center[rot_axis] += dim[
                rot_axis] / 2  # bottom center to gravity center
        elif center_mode == 'camera_bottom':
            center[rot_axis] -= dim[
                rot_axis] / 2  # bottom center to gravity center
        box3d = geometry.OrientedBoundingBox(center, rot_mat, dim)

        line_set = geometry.LineSet.create_from_oriented_bounding_box(box3d)
        line_set.paint_uniform_color(bbox_color)
        line_set_list.append(line_set)
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
    return line_set_list


def show_result(points,
                gt_bboxes,
                pred_bboxes,
                out_dir,
                filename,
                show=True,
                write=False,
                snapshot=False):
    """Convert results into format that is directly readable for meshlab.

    Args:
        points (np.ndarray): Points.
        gt_bboxes (np.ndarray): Ground truth boxes.
        pred_bboxes (np.ndarray): Predicted boxes.
        out_dir (str): Path of output directory
        filename (str): Filename of the current frame.
        show (bool): Visualize the results online. Defaults to False.
        snapshot (bool): Whether to save the online results. Defaults to False.
    """
    result_path = osp.join(out_dir, filename)
    mmcv.mkdir_or_exist(result_path)

    gt_bboxes_o3d = pred_bboxes_o3d = None
    if gt_bboxes is not None:
        gt_bboxes_o3d =_draw_bboxes(
                            gt_bboxes,
                            #self.o3d_visualizer,
                            #self.points_colors,
                            #self.pcd,
                            bbox_color=(0, 1.0, 0),
                            #points_in_box_color,
                            )
    if pred_bboxes is not None:
        pred_bboxes_o3d =_draw_bboxes(
                            pred_bboxes,
                            #self.o3d_visualizer,
                            #self.points_colors,
                            #self.pcd,
                            bbox_color=(1.0, 0, 0),
                            #points_in_box_color,
                            )
    if show:
        from .open3d_vis import Visualizer

        vis = Visualizer(points)
        if pred_bboxes is not None:
            vis.add_bboxes(bbox3d=pred_bboxes)
        if gt_bboxes is not None:
            vis.add_bboxes(bbox3d=gt_bboxes, bbox_color=(0, 0, 1))
        show_path = osp.join(result_path,
                             f'{filename}_online.png') if snapshot else None
        vis.show(show_path)

    if write:
        if points is not None:
            _write_obj(points, osp.join(result_path, f'{filename}_points.obj'))

        if gt_bboxes is not None:
            # bottom center to gravity center
            gt_bboxes[..., 2] += gt_bboxes[..., 5] / 2
            # the positive direction for yaw in meshlab is clockwise
            gt_bboxes[:, 6] *= -1
            _write_oriented_bbox(gt_bboxes,
                                osp.join(result_path, f'{filename}_gt.obj'))

        if pred_bboxes is not None:
            # bottom center to gravity center
            pred_bboxes[..., 2] += pred_bboxes[..., 5] / 2
            # the positive direction for yaw in meshlab is clockwise
            pred_bboxes[:, 6] *= -1
            _write_oriented_bbox(pred_bboxes,
                                osp.join(result_path, f'{filename}_pred.obj'))

    return gt_bboxes_o3d, pred_bboxes_o3d

def show_seg_result(points,
                    gt_seg,
                    pred_seg,
                    out_dir,
                    filename,
                    palette,
                    ignore_index=None,
                    show=False,
                    snapshot=False):
    """Convert results into format that is directly readable for meshlab.

    Args:
        points (np.ndarray): Points.
        gt_seg (np.ndarray): Ground truth segmentation mask.
        pred_seg (np.ndarray): Predicted segmentation mask.
        out_dir (str): Path of output directory
        filename (str): Filename of the current frame.
        palette (np.ndarray): Mapping between class labels and colors.
        ignore_index (int, optional): The label index to be ignored, e.g. \
            unannotated points. Defaults to None.
        show (bool, optional): Visualize the results online. Defaults to False.
        snapshot (bool, optional): Whether to save the online results. \
            Defaults to False.
    """
    # we need 3D coordinates to visualize segmentation mask
    if gt_seg is not None or pred_seg is not None:
        assert points is not None, \
            '3D coordinates are required for segmentation visualization'

    # filter out ignored points
    if gt_seg is not None and ignore_index is not None:
        if points is not None:
            points = points[gt_seg != ignore_index]
        if pred_seg is not None:
            pred_seg = pred_seg[gt_seg != ignore_index]
        gt_seg = gt_seg[gt_seg != ignore_index]

    if gt_seg is not None:
        gt_seg_color = palette[gt_seg]
        gt_seg_color = np.concatenate([points[:, :3], gt_seg_color], axis=1)
    if pred_seg is not None:
        pred_seg_color = palette[pred_seg]
        pred_seg_color = np.concatenate([points[:, :3], pred_seg_color],
                                        axis=1)

    result_path = osp.join(out_dir, filename)
    mmcv.mkdir_or_exist(result_path)

    # online visualization of segmentation mask
    # we show three masks in a row, scene_points, gt_mask, pred_mask
    if show:
        from .open3d_vis import Visualizer
        mode = 'xyzrgb' if points.shape[1] == 6 else 'xyz'
        vis = Visualizer(points, mode=mode)
        if gt_seg is not None:
            vis.add_seg_mask(gt_seg_color)
        if pred_seg is not None:
            vis.add_seg_mask(pred_seg_color)
        show_path = osp.join(result_path,
                             f'{filename}_online.png') if snapshot else None
        vis.show(show_path)

    if points is not None:
        _write_obj(points, osp.join(result_path, f'{filename}_points.obj'))

    if gt_seg is not None:
        _write_obj(gt_seg_color, osp.join(result_path, f'{filename}_gt.obj'))

    if pred_seg is not None:
        _write_obj(pred_seg_color, osp.join(result_path,
                                            f'{filename}_pred.obj'))


def show_multi_modality_result(img,
                               gt_bboxes,
                               pred_bboxes,
                               proj_mat,
                               out_dir,
                               filename,
                               box_mode,
                               img_metas=None,
                               show=False,
                               write=False,
                               gt_bbox_color=(0, 255, 0),  # (61, 102, 255), is Blue
                               pred_bbox_color=(241, 101, 72)):
    """Convert multi-modality detection results into 2D results.

    Project the predicted 3D bbox to 2D image plane and visualize them.

    Args:
        img (np.ndarray): The numpy array of image in cv2 fashion.
        gt_bboxes (:obj:`BaseInstance3DBoxes`): Ground truth boxes.
        pred_bboxes (:obj:`BaseInstance3DBoxes`): Predicted boxes.
        proj_mat (numpy.array, shape=[4, 4]): The projection matrix
            according to the camera intrinsic parameters.
        out_dir (str): Path of output directory.
        filename (str): Filename of the current frame.
        box_mode (str): Coordinate system the boxes are in.
            Should be one of 'depth', 'lidar' and 'camera'.
        img_metas (dict): Used in projecting depth bbox.
        show (bool): Visualize the results online. Defaults to False.
        gt_bbox_color (str or tuple(int)): Color of bbox lines.
           The tuple of color should be in BGR order. Default: (255, 102, 61)
        pred_bbox_color (str or tuple(int)): Color of bbox lines.
           The tuple of color should be in BGR order. Default: (72, 101, 241)
    """
    if box_mode == 'depth':
        draw_bbox = draw_depth_bbox3d_on_img
    elif box_mode == 'lidar':
        draw_bbox = draw_lidar_bbox3d_on_img
    elif box_mode == 'camera':
        draw_bbox = draw_camera_bbox3d_on_img
    else:
        raise NotImplementedError(f'unsupported box mode {box_mode}')

    result_path = osp.join(out_dir, filename)
    mmcv.mkdir_or_exist(result_path)

    show_img = img.copy()
    if gt_bboxes is not None:
        show_img = draw_bbox(
            gt_bboxes, show_img, proj_mat, img_metas, color=gt_bbox_color)
    if pred_bboxes is not None:
        show_img = draw_bbox(
            pred_bboxes,
            show_img,
            proj_mat,
            img_metas,
            color=pred_bbox_color)
    if show:
        mmcv.imshow(show_img, win_name='project_bbox3d_img', wait_time=0)

    if write:
        if img is not None:
            mmcv.imwrite(img, osp.join(result_path, f'{filename}_img.png'))

        if gt_bboxes is not None:
            gt_img = draw_bbox(
                gt_bboxes, img, proj_mat, img_metas, color=gt_bbox_color)
            mmcv.imwrite(gt_img, osp.join(result_path, f'{filename}_gt.png'))

        if pred_bboxes is not None:
            pred_img = draw_bbox(
                pred_bboxes, img, proj_mat, img_metas, color=pred_bbox_color)
            mmcv.imwrite(pred_img, osp.join(result_path, f'{filename}_pred.png'))

    return show_img