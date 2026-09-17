import cv2
import numpy as np
import torch
import os
import sys
from scipy.spatial.transform import Rotation

from datasets.gradslam_datasets.geometryutils import relative_transformation

def as_intrinsics_matrix(intrinsics):
    """
    Get matrix representation of intrinsics.

    """
    K = np.eye(3)
    K[0, 0] = intrinsics[0]
    K[1, 1] = intrinsics[1]
    K[0, 2] = intrinsics[2]
    K[1, 2] = intrinsics[3]
    return K

def pose_matrix_from_quaternion(pvec):
    """ convert pose vector to 4x4 matrix """
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_quat(pvec[3:]).as_matrix()
    pose[:3, 3] = pvec[:3]
    return pose


class ROSHelper:
    def __init__(self, desired_width, desired_height, ori_width, ori_height, 
                 png_depth_scale, fx, fy, cx, cy, max_depth, device, dtype=torch.float32):
        # intrinsics are with the original width and height.
        self.desired_width = desired_width
        self.desired_height = desired_height
        self.png_depth_scale = png_depth_scale
        self.device = device
        self.dtype = dtype
        self.h_ratio = desired_height / ori_height
        self.w_ratio = desired_width / ori_width
        if self.h_ratio != 1 or self.w_ratio != 1:
            print("resize ratio: ", self.h_ratio, self.w_ratio)
            self.fx = fx * self.w_ratio
            self.fy = fy * self.h_ratio
            self.cx = cx * self.w_ratio
            self.cy = cy * self.h_ratio
        else:
            self.fx = fx
            self.fy = fy
            self.cx = cx
            self.cy = cy
        self.step_count = 0
        self.max_depth = max_depth
        self.set_intrinsics()
        print("initialized ROSHelper.")
        print("max depth: ", self.max_depth)

    def process_observation(self, pos, quat, color_img, depth_img, first=False):
        # pos and rot are dict of x,y,z
        self.step_count += 1
        # depth_scaled = (event.depth_frame*1000).astype(np.int64)
        # rgb = event.frame.astype(float)

        # color = helper.color_img(rgb)
        # depth = helper.depth_img(depth_scaled)
        color = self.color_img(color_img)
        depth = self.depth_img(depth_img)
        intrinsics = self.intrinsics()
        # r = Rotation.from_euler('xyz', list(rot.values()), degrees=True)
        # Get the quaternion representation
        # quaternion = r.as_quat()
        pose_vec = pos.tolist() + quat.tolist()
        if first:
            pose = self.get_pose0(pose_vec)
        else:
            pose = self.process_pose(pose_vec)
        return color, depth, intrinsics, pose

    def process_depth(self, depth: np.ndarray):
        r"""Preprocesses the depth image by resizing, adding channel dimension, and scaling values to meters. Optionally
        converts depth from channels last :math:`(H, W, 1)` to channels first :math:`(1, H, W)` representation.

        Args:
            depth (np.ndarray): Raw depth image

        Returns:
            np.ndarray: Preprocessed depth

        Shape:
            - depth: :math:`(H_\text{old}, W_\text{old})`
            - Output: :math:`(H, W, 1)` if `self.channels_first == False`, else :math:`(1, H, W)`.
        """
        depth = cv2.resize(
            depth.astype(float),
            (self.desired_width, self.desired_height),
            interpolation=cv2.INTER_NEAREST,
        )
        depth = np.expand_dims(depth, -1)
        # depth = depth / self.png_depth_scale
        # clip depth values to max_depth
        # depth[depth > self.max_depth] = self.max_depth

        return depth / self.png_depth_scale

    def process_color(self, color: np.ndarray):
        r"""Preprocesses the color image by resizing to :math:`(H, W, C)`, (optionally) normalizing values to
        :math:`[0, 1]`, and (optionally) using channels first :math:`(C, H, W)` representation.

        Args:
            color (np.ndarray): Raw input rgb image

        Retruns:
            np.ndarray: Preprocessed rgb image

        Shape:
            - Input: :math:`(H_\text{old}, W_\text{old}, C)`
            - Output: :math:`(H, W, C)` if `self.channels_first == False`, else :math:`(C, H, W)`.
        """
        color = cv2.resize(
            color,
            (self.desired_width, self.desired_height),
            interpolation=cv2.INTER_LINEAR,
        )
        return color

    def color_img(self, color):
        print("color type should be uint8: ", color.dtype)
        if self.h_ratio != 1 or self.w_ratio != 1:
            color = self.process_color(color)
        # ret = self.process_color(color)
        ret = torch.from_numpy(color)
        return ret.to(self.device).type(self.dtype)

    def depth_img(self, depth):
        print("depth type should be float32: ", depth.dtype)
        if self.h_ratio != 1 or self.w_ratio != 1:
            depth = self.process_depth(depth)
        else:
            depth = np.expand_dims(depth, -1)
            depth = depth / self.png_depth_scale
        ret = torch.from_numpy(depth)
        return ret.to(self.device).type(self.dtype)
    
    def set_intrinsics(self):
        K = as_intrinsics_matrix([self.fx, self.fy, self.cx, self.cy])
        K = torch.from_numpy(K)
        intrinsics = torch.eye(4).to(K)
        intrinsics[:3, :3] = K
        self.intrinsics_matrix = intrinsics.to(self.device).type(self.dtype)

    def intrinsics(self):
        return self.intrinsics_matrix

    def get_pose0(self, data):
        # assume input data (x, y, z, q1, q2, q3, q4)
        c2w = pose_matrix_from_quaternion(data)
        c2w = torch.from_numpy(c2w).float()
        self.pose0 = c2w
        print("set pose 0 at: ", c2w)
        # reset c2w to indentity matrix
        c2w_first = torch.from_numpy(np.eye(4)).float()
        return c2w_first.to(self.device).type(self.dtype)
    
    def set_pose0(self, data):
        c2w = pose_matrix_from_quaternion(data)
        c2w = torch.from_numpy(c2w).float()
        self.pose0 = c2w

    def process_pose(self, data):
        # assume input data (x, y, z, q1, q2, q3, q4)
        c2w = pose_matrix_from_quaternion(data)
        c2w = torch.from_numpy(c2w).float()
        
        r"""Preprocesses the poses by setting first pose in a sequence to identity and computing the relative
        homogenous transformation for all other poses.

        Args:
            poses (torch.Tensor): Pose matrices to be preprocessed

        Returns:
            Output (torch.Tensor): Preprocessed poses

        Shape:
            - poses: :math:`(L, 4, 4)` where :math:`L` denotes sequence length.
            - Output: :math:`(L, 4, 4)` where :math:`L` denotes sequence length.
        """
        # print("shape of pose0: ", self.pose0.shape)
        # print("shape of c2w: ", c2w.shape)
        transformed_poses = relative_transformation(
            self.pose0.unsqueeze(0),
            c2w.unsqueeze(0),
            orthogonal_rotations=False,
        )
        # print("shape of transformed_poses: ", transformed_poses.shape)
        ret = transformed_poses.squeeze(0)
        # print("++++++++++++++++++++++++++++++++++++++")
        # print("transformed poses: ", ret)
        return ret.to(self.device).type(self.dtype)

