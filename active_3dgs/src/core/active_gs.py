#!/usr/bin/env python3

import rospy

import argparse
import os
import shutil
import sys
import time
import numpy as np
from importlib.machinery import SourceFileLoader
from matplotlib import pyplot as plt
# import imageio
import numpy as np
from scipy.spatial.transform import Rotation
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from utils.common_utils import seed_everything, save_params_ckpt, save_params
from utils.eval_helpers import report_loss, report_progress
from utils.keyframe_selection import keyframe_selection_overlap
from utils.recon_helpers import setup_camera
from utils.slam_helpers import (
    transformed_params2rendervar, transformed_params2depthplussilhouette,
    transformed_params2rendervar_multi, transformed_params2rendervar_multi_split,
    transform_to_frame, l1_loss_v1, matrix_to_quaternion
)
from utils.slam_external import calc_ssim, build_rotation, prune_gaussians, densify
import json

# pip installed.
from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from diff_gaussian_rasterization_multi import GaussianRasterizer as RendererMulti

from nav_msgs.msg import Odometry, Path
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
import message_filters
from sensor_msgs.msg import Image, PointCloud2, PointField
from sensor_msgs import point_cloud2
from core.ros_helper import ROSHelper
from cv_bridge import CvBridge
from std_msgs.msg import Header
from std_srvs.srv import Trigger, TriggerResponse
from std_msgs.msg import String as RosString
from copy import deepcopy
import tf2_ros
from geometry_msgs.msg import TransformStamped, PoseStamped, Pose
from active_3dgs_msgs.msg import GaussianPoints, PointWithSize
# For testing husky move_base
from move_base_msgs.msg import MoveBaseActionGoal, MoveBaseActionResult
from mpl.planner_ros import LocalPlanner

from core.gaussian_gradients import get_gaussian_frontiers, evaluate_trajectories_relevancy
from core.topological_graph import TopoTree
from core.submap import Submap, SubmapManager

from core.global_planner import GlobalPlanner

from threading import Lock

import gc

from active_3dgs_msgs.srv import SetTask

PCA_DIM = 24

bridge = CvBridge()

class ActiveGS:

    def __init__(self):
        self.config_path = rospy.get_param('~config_path', 'config.py')

        # Use old Splatam way to load params
        # parser = argparse.ArgumentParser()
        # parser.add_argument("experiment", type=str, help="Path to experiment file")
        # args = parser.parse_args()


        experiment = SourceFileLoader(
            os.path.basename(self.config_path), self.config_path
        ).load_module()

        # Set Experiment Seed
        seed_everything(seed=experiment.config['seed'])
        
        # Create Results Directory and Copy Config
        results_dir = os.path.join(
            experiment.config["workdir"], experiment.config["run_name"]
        )
        # if not experiment.config['load_checkpoint']:
        #     os.makedirs(results_dir, exist_ok=True)
        #     shutil.copy(self.config_path, os.path.join(results_dir, "config.py"))

        print("Loaded Config:")
        print("-----------------------------------------")
        print(f"{experiment.config}")
        print("-----------------------------------------")


        self.config = experiment.config

        self.load_wp_from_json = rospy.get_param('~load_wp_from_json', False)
        wp_json_path = rospy.get_param('~wp_json_path', 'data/waypoints.json')

        self.map_rate = rospy.get_param('~map_rate', 1.0)
        self.first_observation = True
        self.odom_pos_ = np.zeros(3)
        self.odom_rpy_ = np.zeros(3)
        self.odom_quat_ = np.zeros(4)
        self.odom_stamp_ = None
        self.planning_param_set = False
        self.prev_time = rospy.Time.now()
        self.sim_ = rospy.get_param('~gs_sim', False)
        self.cam2body_ = np.asarray(self.config['cam_config'].get('cam2body'), dtype=float)
        if self.cam2body_.shape != (4, 4) or not np.isfinite(self.cam2body_).all():
            raise ValueError("cam_config.cam2body must be a finite 4x4 camera-to-body transform")
        if not np.allclose(self.cam2body_[3], [0, 0, 0, 1], atol=1e-8, rtol=0):
            raise ValueError("cam_config.cam2body must have last row [0, 0, 0, 1]")
        rotation = self.cam2body_[:3, :3]
        if (not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5, rtol=0)
                or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5, rtol=0)):
            raise ValueError("cam_config.cam2body rotation must be orthonormal with determinant +1")
        self.cam2body_t_ = self.cam2body_[:3, 3].copy()
        self.pc_fields_ = self.make_fields()
        self.prev_goal = None
        self.visited_frontiers = []
        self.mutex = Lock()
        self.ftr_cluster_size_ = rospy.get_param('~ftr_cluster_size', 0.5)
        self.nms_radius_ = rospy.get_param('~nms_radius', 1.5)
        self.filter_grad_mean_ = rospy.get_param('~filter_grad_mean', False)
        self.ftr_counter_ = 0

        # ros topics
        self.use_odom_msg = rospy.get_param('~use_odom_msg', False)
        self.odom_topic = rospy.get_param('~odom_topic', '/dragonfly67/quadrotor_ukf/control_odom')
        self.rgb_topic = rospy.get_param('~rgb_topic', '/camera/color/image_raw/')
        self.aligned_depth_topic = rospy.get_param('~aligned_depth_topic', '/camera/aligned_depth_to_color/image_raw')
        self.workdir = rospy.get_param('~workdir', results_dir)
        # sync odom, rgb and depth
        rospy.loginfo("syncing rgb, aligned depth and odom")
        self.rgb_sub = message_filters.Subscriber(self.rgb_topic, Image)
        self.aligned_depth_sub = message_filters.Subscriber(self.aligned_depth_topic, Image)
        # self.odom_sub = message_filters.Subscriber(self.odom_topic, Odometry)
        if self.use_odom_msg:
            self.odom_sub = message_filters.Subscriber(self.odom_topic, Odometry)
        else:
            self.odom_sub = message_filters.Subscriber(self.odom_topic, PoseStamped)
        self.map_pub = rospy.Publisher('/map', PointCloud2, queue_size=10)
        self.relevancy_pub = rospy.Publisher('/relevancy', PointCloud2, queue_size=10)
        self.tf_pub = tf2_ros.StaticTransformBroadcaster()
        self.gaussian_pub = rospy.Publisher('/gaussian_pts', GaussianPoints, queue_size=10)
        self.ftr_goal_pub = rospy.Publisher('/ftr_goal', PoseStamped, queue_size=10)

        # Optimized odom callback
        self.opt_pose_path_sub = rospy.Subscriber('/opt_pose', Path, self.opt_pose_path_cb)
        self.latest_pose_path = None

        # For testing move base
        self.husky_goal_pub = rospy.Publisher('/move_base/goal', MoveBaseActionGoal, queue_size=10)
        self.move_base_result_sub = rospy.Subscriber('/move_base/result', MoveBaseActionResult, self.move_base_result_cb)

        # Service to store Gaussians
        self.save_gs_map = rospy.Service('save_gs_map', Trigger, self.save_map_cb)

        # Init planner
        self.local_planner = LocalPlanner(evaluate_trajectories_relevancy)
        self.topo_tree = TopoTree()
        # self.plan_rate = rospy.get_param('~plan_rate', 3.0)
        self.height_filter = rospy.get_param('~height_filter', 0.3)
        gs_map_type = rospy.get_param('~gs_map_type', 'full')
        if gs_map_type == 'full':
            self.map_type_ = 0
        elif gs_map_type == 'filtered':
            self.map_type_ = 1
        elif gs_map_type == 'debug':
            self.map_type_ = 2
        elif gs_map_type == 'none':
            self.map_type_ = 3
        else:
            self.map_type_ = 0
        self.subsample_ = rospy.get_param('~subsample', 10)
        self.ftr_goal_tol_ = rospy.get_param('~ftr_goal_tol', 1.0)
        self.fail_pos_tol_ = rospy.get_param('~fail_pos_tol', 0.1)
        self.fail_yaw_tol_ = rospy.get_param('~fail_yaw_tol', 0.1)
        self.path_to_ftr = None
        self.params_copy = None
        self.process_bag = rospy.get_param("~offline", False)

        # Print params from config
        rospy.loginfo("odom_topic: " + self.odom_topic)
        rospy.loginfo("rgb_topic: " + self.rgb_topic)
        rospy.loginfo("aligned_depth_topic: " + self.aligned_depth_topic)
        rospy.loginfo("workdir: " + self.workdir)
        # rospy.loginfo("plan_rate: " + str(self.plan_rate))
        rospy.loginfo("height_filter: " + str(self.height_filter))
        rospy.loginfo("gs_sim: " + str(self.sim_))
        rospy.loginfo("cam2body_t: " + str(self.cam2body_t_))
        rospy.loginfo("gs_map_type: " + gs_map_type)
        rospy.loginfo("subsample: " + str(self.subsample_))

        self.init_rgbd_slam()        
        
        # global planner
        self.global_grid_size = rospy.get_param('~global_grid_size', 100)
        self.global_grid_resolution = rospy.get_param('~global_grid_resolution', 5)
        self.region_relevancy_threshold = rospy.get_param('~region_relevancy_threshold', 0.5)
        self.object_relevancy_threshold = rospy.get_param('~object_relevancy_threshold', 0.7)
        self.global_planner = GlobalPlanner(grid_length=self.global_grid_size,
                                            resolution=self.global_grid_resolution,
                                            region_rel_th=self.region_relevancy_threshold,
                                            obj_rel_th=self.object_relevancy_threshold)
        self.first_cam2world = None
        
        # language features
        self.ipca_path = rospy.get_param('~ipca_path', 'data/ipca.pth')
        self.use_language_features = rospy.get_param('~use_language_features', False)
        if self.use_language_features:
            if not os.path.isfile(self.ipca_path):
                raise ValueError("Set ipca_path to an existing PCA checkpoint to enable language features")
            from core.language_features import LanguageFeaturesManager
            self.lfm = LanguageFeaturesManager(feature_mode='pca', clip_dim=512, pca_dim=PCA_DIM, 
                                               data_path=self.ipca_path, region_rel_th=self.region_relevancy_threshold,
                                               use_vlm=rospy.get_param('~use_vlm', False))
            self.set_task_service = rospy.Service('set_task', SetTask, self.set_task_cb)
            self.termination_pub = rospy.Publisher('/vlm', RosString, queue_size=10)
        self.task_mutex = Lock()

        # Initialize submap
        self.submap_manager = SubmapManager(submap_distance=2.0, submap_size=10.0, cam2body=self.cam2body_)
        self.map_idx = None
        self.prev_map_idx = None
        # ApproximateTimeSynchronizer to allow for 0.01s time difference
        ts = message_filters.ApproximateTimeSynchronizer([self.rgb_sub, self.aligned_depth_sub, self.odom_sub], 10, 0.015)
        ts.registerCallback(self.rgb_aligned_depth_odom_callback)
        
        # timer based on plan_rate
        # if not self.process_bag:
        #     self.plan_timer = rospy.Timer(rospy.Duration(1.0/self.plan_rate), self.plan_cb)
        
        rospy.loginfo("Active GS Node Initialized")
        # torch.cuda.memory._record_memory_history()

        if self.load_wp_from_json:
            self.path_to_ftr = self.load_wp(wp_json_path)

        self.vlm_ctr = 0
        
        self.ring_pub = rospy.Publisher('ring_viz', Marker, queue_size=10)


    def initialize_optimizer(self, params, lrs_dict, tracking):
        lrs = lrs_dict
        param_groups = [{'params': [v], 'name': k, 'lr': lrs[k]} for k, v in params.items()]
        if tracking:
            return torch.optim.Adam(param_groups)
        else:
            return torch.optim.Adam(param_groups, lr=0.0, eps=1e-15)

    def initialize_camera_pose(self, params, curr_time_idx, forward_prop):
        with torch.no_grad():
            if curr_time_idx > 1 and forward_prop:
                # Initialize the camera pose for the current frame based on a constant velocity model
                # Rotation
                prev_rot1 = F.normalize(params['cam_unnorm_rots'][..., curr_time_idx-1].detach())
                prev_rot2 = F.normalize(params['cam_unnorm_rots'][..., curr_time_idx-2].detach())
                new_rot = F.normalize(prev_rot1 + (prev_rot1 - prev_rot2))
                params['cam_unnorm_rots'][..., curr_time_idx] = new_rot.detach()
                # Translation
                prev_tran1 = params['cam_trans'][..., curr_time_idx-1].detach()
                prev_tran2 = params['cam_trans'][..., curr_time_idx-2].detach()
                new_tran = prev_tran1 + (prev_tran1 - prev_tran2)
                params['cam_trans'][..., curr_time_idx] = new_tran.detach()
            else:
                # Initialize the camera pose for the current frame
                params['cam_unnorm_rots'][..., curr_time_idx] = params['cam_unnorm_rots'][..., curr_time_idx-1].detach()
                params['cam_trans'][..., curr_time_idx] = params['cam_trans'][..., curr_time_idx-1].detach()
        
        return params


    def ros_initialize_first_timestep(self, color, depth, intrinsics, pose, num_frames, scene_radius_depth_ratio, 
                                mean_sq_dist_method, gaussian_distribution=None):
        # Get RGB-D Data & Camera Parameters
        # color, depth, intrinsics, pose = dataset[0]

        # Process RGB-D Data
        color_orig = deepcopy(color)
        color = color.permute(2, 0, 1) / 255 # (H, W, C) -> (C, H, W)
        depth = depth.permute(2, 0, 1) # (H, W, C) -> (C, H, W)
        
        if self.use_language_features:
            feat_img, feat_orig_img = self.lfm.get_features(color)
        else:
            feat_img = torch.zeros((PCA_DIM, color.shape[1], color.shape[2])).cuda()

        # Process Camera Parameters
        intrinsics = intrinsics[:3, :3]
        # w2c = torch.linalg.inv(pose)
        w2c = pose # Identity.

        # Setup Camera
        cam = setup_camera(color.shape[2], color.shape[1], intrinsics.cpu().numpy(), w2c.detach().cpu().numpy())

        densify_intrinsics = intrinsics

        # Get Initial Point Cloud (PyTorch CUDA Tensor)
        mask = (depth > 0) & (depth <= self.cam_config['max_depth']) # Mask out invalid depth values
        mask = mask.reshape(-1)
        init_pt_cld, mean3_sq_dist = self.get_pointcloud(color, depth, feat_img, densify_intrinsics, w2c, 
                                                    mask=mask, compute_mean_sq_dist=True, 
                                                    mean_sq_dist_method=mean_sq_dist_method)

        # Initialize Parameters
        params, variables = self.initialize_params(init_pt_cld, num_frames, mean3_sq_dist, gaussian_distribution)

        # Initialize an estimate of scene radius for Gaussian-Splatting Densification
        variables['scene_radius'] = torch.max(depth)/scene_radius_depth_ratio

        return params, variables, intrinsics, w2c, cam



    def initialize_params(self, init_pt_cld, num_frames, mean3_sq_dist, gaussian_distribution):
        num_pts = init_pt_cld.shape[0]
        # height_filter = 0.0
        ground_labels = torch.zeros((num_pts)).cuda().float()
        ground_labels[init_pt_cld[:, 1] > self.height_filter] = 1
        means3D = init_pt_cld[:, :3] # [num_gaussians, 3]
        unnorm_rots = np.tile([1, 0, 0, 0], (num_pts, 1)) # [num_gaussians, 4]
        logit_opacities = torch.zeros((num_pts, 1), dtype=torch.float, device="cuda")
        if gaussian_distribution == "isotropic":
            log_scales = torch.tile(torch.log(torch.sqrt(mean3_sq_dist))[..., None], (1, 1))
        elif gaussian_distribution == "anisotropic":
            log_scales = torch.tile(torch.log(torch.sqrt(mean3_sq_dist))[..., None], (1, 3))
        else:
            raise ValueError(f"Unknown gaussian_distribution {gaussian_distribution}")
        params = {
            'means3D': means3D,
            'rgb_colors': init_pt_cld[:, 3:6],
            'unnorm_rotations': unnorm_rots,
            'logit_opacities': logit_opacities,
            'log_scales': log_scales,
            # 'features': torch.zeros((num_pts, PCA_DIM), dtype=torch.float, device="cuda"),
            'features': init_pt_cld[:, 6:6+PCA_DIM],
            'grad_means3D': torch.zeros((num_pts)).cuda().float(),
            'ground_labels': ground_labels,
            'submap_id': torch.zeros((num_pts)).cuda().float()
        }

        # Initialize a single gaussian trajectory to model the camera poses relative to the first frame
        cam_rots = np.tile([1, 0, 0, 0], (1, 1))
        cam_rots = np.tile(cam_rots[:, :, None], (1, 1, num_frames))
        params['cam_unnorm_rots'] = cam_rots
        params['cam_trans'] = np.zeros((1, 3, num_frames))

        for k, v in params.items():
            # Check if value is already a torch tensor
            if not isinstance(v, torch.Tensor):
                params[k] = torch.nn.Parameter(torch.tensor(v).cuda().float().contiguous().requires_grad_(True))
            else:
                if k in ["update_means3D", "noupdate_means3D", "nochange_means3D"]:
                    params[k] = torch.nn.Parameter(v.cuda().float().contiguous().requires_grad_(False))
                else:
                    params[k] = torch.nn.Parameter(v.cuda().float().contiguous().requires_grad_(True))

        variables = {'max_2D_radius': torch.zeros(params['means3D'].shape[0]).cuda().float(),
                    'means2D_gradient_accum': torch.zeros(params['means3D'].shape[0]).cuda().float(),
                    'denom': torch.zeros(params['means3D'].shape[0]).cuda().float(),
                    'timestep': torch.zeros(params['means3D'].shape[0]).cuda().float()}

        return params, variables


    def initialize_new_params(self, new_pt_cld, mean3_sq_dist, gaussian_distribution, submap_id):
        num_pts = new_pt_cld.shape[0]
        # label point cloud by height
        # height_filter = 0.0
        ground_labels = torch.zeros((num_pts)).cuda().float()
        ground_labels[new_pt_cld[:, 1] > self.height_filter] = 1
        submap_ids = torch.full_like((ground_labels), submap_id).cuda().float()
        means3D = new_pt_cld[:, :3] # [num_gaussians, 3]
        unnorm_rots = np.tile([1, 0, 0, 0], (num_pts, 1)) # [num_gaussians, 4]
        logit_opacities = torch.zeros((num_pts, 1), dtype=torch.float, device="cuda")
        if gaussian_distribution == "isotropic":
            log_scales = torch.tile(torch.log(torch.sqrt(mean3_sq_dist))[..., None], (1, 1))
        elif gaussian_distribution == "anisotropic":
            log_scales = torch.tile(torch.log(torch.sqrt(mean3_sq_dist))[..., None], (1, 3))
        else:
            raise ValueError(f"Unknown gaussian_distribution {gaussian_distribution}")
        params = {
            'means3D': means3D,
            'rgb_colors': new_pt_cld[:, 3:6],
            'unnorm_rotations': unnorm_rots,
            'logit_opacities': logit_opacities,
            'log_scales': log_scales,
            # 'features': torch.zeros((num_pts, PCA_DIM), dtype=torch.float, device="cuda"),
            'features': new_pt_cld[:, 6:6+PCA_DIM],
            'grad_means3D': torch.zeros((num_pts)).cuda().float(),
            'ground_labels': ground_labels,
            'submap_id': submap_ids
        }
        for k, v in params.items():
            # Check if value is already a torch tensor
            if not isinstance(v, torch.Tensor):
                params[k] = torch.nn.Parameter(torch.tensor(v).cuda().float().contiguous().requires_grad_(True))
            else:
                if k in ["update_means3D", "noupdate_means3D", "nochange_means3D"]:
                    params[k] = torch.nn.Parameter(v.cuda().float().contiguous().requires_grad_(False))
                else:
                    params[k] = torch.nn.Parameter(v.cuda().float().contiguous().requires_grad_(True))

        return params
    

    def get_pointcloud(self, color, depth, feat_img, intrinsics, w2c, transform_pts=True, 
                    mask=None, compute_mean_sq_dist=False, mean_sq_dist_method="projective"):
        width, height = color.shape[2], color.shape[1]
        CX = intrinsics[0][2]
        CY = intrinsics[1][2]
        FX = intrinsics[0][0]
        FY = intrinsics[1][1]

        # Compute indices of pixels
        x_grid, y_grid = torch.meshgrid(torch.arange(width).cuda().float(), 
                                        torch.arange(height).cuda().float(),
                                        indexing='xy')
        xx = (x_grid - CX)/FX
        yy = (y_grid - CY)/FY
        xx = xx.reshape(-1)
        yy = yy.reshape(-1)
        depth_z = depth[0].reshape(-1)

        # Initialize point cloud
        pts_cam = torch.stack((xx * depth_z, yy * depth_z, depth_z), dim=-1)
        if transform_pts:
            pix_ones = torch.ones(height * width, 1).cuda().float()
            pts4 = torch.cat((pts_cam, pix_ones), dim=1)
            c2w = torch.inverse(w2c)
            pts = (c2w @ pts4.T).T[:, :3]
        else:
            pts = pts_cam

        # Compute mean squared distance for initializing the scale of the Gaussians
        if compute_mean_sq_dist:
            if mean_sq_dist_method == "projective":
                # Projective Geometry (this is fast, farther -> larger radius)
                scale_gaussian = depth_z / ((FX + FY)/2)
                mean3_sq_dist = scale_gaussian**2
            else:
                raise ValueError(f"Unknown mean_sq_dist_method {mean_sq_dist_method}")
        
        # Colorize point cloud
        cols = torch.permute(color, (1, 2, 0)).reshape(-1, 3) # (C, H, W) -> (H, W, C) -> (H * W, C)
        feat_cols = torch.permute(feat_img, (1, 2, 0)).reshape(-1, PCA_DIM) # (C, H, W) -> (H, W, C) -> (H * W, C)
        point_cld = torch.cat((pts, cols), -1)
        point_cld = torch.cat((point_cld, feat_cols), -1)

        # Select points based on mask
        if mask is not None:
            point_cld = point_cld[mask]
            if compute_mean_sq_dist:
                mean3_sq_dist = mean3_sq_dist[mask]

        if compute_mean_sq_dist:
            return point_cld, mean3_sq_dist
        else:
            return point_cld
        

    def render_depth(self, params, curr_data, sil_thres, iter_time_idx):
        # Get current frame Gaussians
        transformed_gaussians = transform_to_frame(params, iter_time_idx,
                                                    gaussians_grad=False,
                                                    camera_grad=False)

        # Initialize Render Variables
        depth_sil_rendervar = transformed_params2depthplussilhouette(params, curr_data['w2c'],
                                                                    transformed_gaussians)

        # Depth & Silhouette Rendering
        depth_sil, _, _, = Renderer(raster_settings=curr_data['cam'])(**depth_sil_rendervar)
        depth = depth_sil[0, :, :].unsqueeze(0) # (1, H, W)
        # silhouette = depth_sil[1, :, :]
        # presence_sil_mask = (silhouette > sil_thres)
        # depth_sq = depth_sil[2, :, :].unsqueeze(0)
        # uncertainty = depth_sq - depth**2
        # uncertainty = uncertainty.detach()
        # print("Depth Shape: ", depth.shape)
        return depth.squeeze(0)
    

    def add_new_gaussians(self, params, variables, curr_data, sil_thres, 
                      time_idx, mean_sq_dist_method, gaussian_distribution):
        # Silhouette Rendering
        transformed_gaussians = transform_to_frame(params, time_idx, gaussians_grad=False, camera_grad=False)
        depth_sil_rendervar = transformed_params2depthplussilhouette(params, curr_data['w2c'],
                                                                    transformed_gaussians)
        depth_sil, _, _, = Renderer(raster_settings=curr_data['cam'])(**depth_sil_rendervar)
        silhouette = depth_sil[1, :, :]
        non_presence_sil_mask = (silhouette < sil_thres)
        # Check for new foreground objects by using GT depth
        gt_depth = curr_data['depth'][0, :, :]
        render_depth = depth_sil[0, :, :]
        depth_error = torch.abs(gt_depth - render_depth) * (gt_depth > 0)
        non_presence_depth_mask = (render_depth > gt_depth) * (depth_error > 50*depth_error.median())
        # Determine non-presence mask
        non_presence_mask = non_presence_sil_mask | non_presence_depth_mask
        # Flatten mask
        non_presence_mask = non_presence_mask.reshape(-1)

        # Get the new frame Gaussians based on the Silhouette
        if torch.sum(non_presence_mask) > 0:
            # Get the new pointcloud in the world frame
            curr_cam_rot = torch.nn.functional.normalize(params['cam_unnorm_rots'][..., time_idx].detach())
            curr_cam_tran = params['cam_trans'][..., time_idx].detach()
            curr_w2c = torch.eye(4).cuda().float()
            curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
            curr_w2c[:3, 3] = curr_cam_tran
            valid_depth_mask = (curr_data['depth'][0, :, :] > 0) & (curr_data['depth'][0, :, :] <= self.cam_config['max_depth'])
            non_presence_mask = non_presence_mask & valid_depth_mask.reshape(-1)
            new_pt_cld, mean3_sq_dist = self.get_pointcloud(curr_data['im'], curr_data['depth'],
                                        curr_data['feat_img'], curr_data['intrinsics'], 
                                        curr_w2c, mask=non_presence_mask, compute_mean_sq_dist=True,
                                        mean_sq_dist_method=mean_sq_dist_method)
            new_params = self.initialize_new_params(new_pt_cld, mean3_sq_dist, gaussian_distribution, curr_data['submap_id'])
            if 'grad_means3D' not in params:
                params['grad_means3D'] = torch.zeros(params['means3D'].shape[0]).cuda().float()
            if 'ground_labels' not in params:
                params['ground_labels'] = torch.zeros(params['means3D'].shape[0]).cuda().float()
            for k, v in new_params.items():
                if k in ["update_means3D", "noupdate_means3D", "nochange_means3D"]:
                    params[k] = torch.nn.Parameter(torch.cat((params[k], v), dim=0).requires_grad_(False))
                else:
                    params[k] = torch.nn.Parameter(torch.cat((params[k], v), dim=0).requires_grad_(True))
            num_pts = params['means3D'].shape[0]
            variables['means2D_gradient_accum'] = torch.zeros(num_pts, device="cuda").float()
            variables['denom'] = torch.zeros(num_pts, device="cuda").float()
            variables['max_2D_radius'] = torch.zeros(num_pts, device="cuda").float()
            new_timestep = time_idx*torch.ones(new_pt_cld.shape[0],device="cuda").float()
            variables['timestep'] = torch.cat((variables['timestep'],new_timestep),dim=0)

        return params, variables
    
    
    def get_loss_feat_split(self, params, curr_data, variables, iter_time_idx, loss_weights, use_sil_for_loss,
             sil_thres, use_l1, ignore_outlier_depth_loss, tracking=False, 
             mapping=False, do_ba=False, plot_dir=None, visualize_tracking_loss=False, tracking_iteration=None):

        # Fix Gaussian parameters
        transformed_gaussians = transform_to_frame(params, iter_time_idx,
                                                gaussians_grad=False,
                                                camera_grad=False)
        
        train_features = True
        loss = 0
        
        rendervar1, rendervar2 = transformed_params2rendervar_multi_split(params, transformed_gaussians)
        
        # Feature Rendering 1
        feat_pca1, _, _, = RendererMulti(raster_settings=curr_data['cam'])(**rendervar1)
        feat_pca2, _, _, = RendererMulti(raster_settings=curr_data['cam'])(**rendervar2)
        feat_pca = torch.cat((feat_pca1, feat_pca2), dim=0)
        # Feature loss
        nan_mask = ~torch.isnan(feat_pca)
        loss_feat = F.l1_loss(feat_pca[nan_mask], curr_data['feat_img'][nan_mask])
        loss += loss_feat
                
        return loss
    
    
    def get_loss_feat(self, params, curr_data, variables, iter_time_idx, loss_weights, use_sil_for_loss,
             sil_thres, use_l1, ignore_outlier_depth_loss, tracking=False, 
             mapping=False, do_ba=False, plot_dir=None, visualize_tracking_loss=False, tracking_iteration=None):

        # Fix Gaussian parameters
        transformed_gaussians = transform_to_frame(params, iter_time_idx,
                                                gaussians_grad=False,
                                                camera_grad=False)
        
        train_features = True
        loss = 0
        if train_features:
            # Feature Rendering
            rendervar_multi = transformed_params2rendervar_multi(params, transformed_gaussians)
            feat_pca, _, _, = RendererMulti(raster_settings=curr_data['cam'])(**rendervar_multi)        
            # Feature loss
            nan_mask = ~torch.isnan(feat_pca)
            loss_feat = F.l1_loss(feat_pca[nan_mask], curr_data['feat_img'][nan_mask])
            loss += loss_feat
                
        return loss
    
    
    def get_loss(self, params, curr_data, variables, iter_time_idx, loss_weights, use_sil_for_loss,
                sil_thres, use_l1, ignore_outlier_depth_loss, tracking=False, 
                mapping=False, do_ba=False, plot_dir=None, visualize_tracking_loss=False, tracking_iteration=None):
        # Initialize Loss Dictionary
        losses = {}

        if tracking:
            # Get current frame Gaussians, where only the camera pose gets gradient
            transformed_gaussians = transform_to_frame(params, iter_time_idx, 
                                                gaussians_grad=False,
                                                camera_grad=True)
        elif mapping:
            if do_ba:
                # Get current frame Gaussians, where both camera pose and Gaussians get gradient
                transformed_gaussians = transform_to_frame(params, iter_time_idx,
                                                    gaussians_grad=True,
                                                    camera_grad=True)
            else:
                # Get current frame Gaussians, where only the Gaussians get gradient
                transformed_gaussians = transform_to_frame(params, iter_time_idx,
                                                    gaussians_grad=True,
                                                    camera_grad=False)
        else:
            # Get current frame Gaussians, where only the Gaussians get gradient
            transformed_gaussians = transform_to_frame(params, iter_time_idx,
                                                gaussians_grad=True,
                                                camera_grad=False)

        # Initialize Render Variables
        rendervar = transformed_params2rendervar(params, transformed_gaussians)
        depth_sil_rendervar = transformed_params2depthplussilhouette(params, curr_data['w2c'],
                                                                    transformed_gaussians)

        # RGB Rendering
        rendervar['means2D'].retain_grad()
        im, radius, _, = Renderer(raster_settings=curr_data['cam'])(**rendervar)
        variables['means2D'] = rendervar['means2D']  # Gradient only accum from colour render for densification

        # Depth & Silhouette Rendering
        depth_sil, _, _, = Renderer(raster_settings=curr_data['cam'])(**depth_sil_rendervar)
        depth = depth_sil[0, :, :].unsqueeze(0)
        silhouette = depth_sil[1, :, :]
        presence_sil_mask = (silhouette > sil_thres)
        depth_sq = depth_sil[2, :, :].unsqueeze(0)
        uncertainty = depth_sq - depth**2
        uncertainty = uncertainty.detach()

        # Mask with valid depth values (accounts for outlier depth values)
        nan_mask = (~torch.isnan(depth)) & (~torch.isnan(uncertainty))
        if ignore_outlier_depth_loss:
            depth_error = torch.abs(curr_data['depth'] - depth) * (curr_data['depth'] > 0)
            mask = (depth_error < 10*depth_error.median())
            mask = mask & (curr_data['depth'] > 0)
        else:
            mask = (curr_data['depth'] > 0)
        mask = mask & nan_mask
        # Mask with presence silhouette mask (accounts for empty space)
        if tracking and use_sil_for_loss:
            mask = mask & presence_sil_mask

        # Depth loss
        if use_l1:
            mask = mask.detach()
            if tracking:
                losses['depth'] = torch.abs(curr_data['depth'] - depth)[mask].sum()
            else:
                losses['depth'] = torch.abs(curr_data['depth'] - depth)[mask].mean()
        
        # RGB Loss
        if tracking and (use_sil_for_loss or ignore_outlier_depth_loss):
            color_mask = torch.tile(mask, (3, 1, 1))
            color_mask = color_mask.detach()
            losses['im'] = torch.abs(curr_data['im'] - im)[color_mask].sum()
        elif tracking:
            losses['im'] = torch.abs(curr_data['im'] - im).sum()
        else:
            losses['im'] = 0.8 * l1_loss_v1(im, curr_data['im']) + 0.2 * (1.0 - calc_ssim(im, curr_data['im']))

        # Visualize the Diff Images
        if tracking and visualize_tracking_loss:
            fig, ax = plt.subplots(2, 4, figsize=(12, 6))
            weighted_render_im = im * color_mask
            weighted_im = curr_data['im'] * color_mask
            weighted_render_depth = depth * mask
            weighted_depth = curr_data['depth'] * mask
            diff_rgb = torch.abs(weighted_render_im - weighted_im).mean(dim=0).detach().cpu()
            diff_depth = torch.abs(weighted_render_depth - weighted_depth).mean(dim=0).detach().cpu()
            viz_img = torch.clip(weighted_im.permute(1, 2, 0).detach().cpu(), 0, 1)
            ax[0, 0].imshow(viz_img)
            ax[0, 0].set_title("Weighted GT RGB")
            viz_render_img = torch.clip(weighted_render_im.permute(1, 2, 0).detach().cpu(), 0, 1)
            ax[1, 0].imshow(viz_render_img)
            ax[1, 0].set_title("Weighted Rendered RGB")
            ax[0, 1].imshow(weighted_depth[0].detach().cpu(), cmap="jet", vmin=0, vmax=6)
            ax[0, 1].set_title("Weighted GT Depth")
            ax[1, 1].imshow(weighted_render_depth[0].detach().cpu(), cmap="jet", vmin=0, vmax=6)
            ax[1, 1].set_title("Weighted Rendered Depth")
            ax[0, 2].imshow(diff_rgb, cmap="jet", vmin=0, vmax=0.8)
            ax[0, 2].set_title(f"Diff RGB, Loss: {torch.round(losses['im'])}")
            ax[1, 2].imshow(diff_depth, cmap="jet", vmin=0, vmax=0.8)
            ax[1, 2].set_title(f"Diff Depth, Loss: {torch.round(losses['depth'])}")
            ax[0, 3].imshow(presence_sil_mask.detach().cpu(), cmap="gray")
            ax[0, 3].set_title("Silhouette Mask")
            ax[1, 3].imshow(mask[0].detach().cpu(), cmap="gray")
            ax[1, 3].set_title("Loss Mask")
            # Turn off axis
            for i in range(2):
                for j in range(4):
                    ax[i, j].axis('off')
            # Set Title
            fig.suptitle(f"Tracking Iteration: {tracking_iteration}", fontsize=16)
            # Figure Tight Layout
            fig.tight_layout()
            os.makedirs(plot_dir, exist_ok=True)
            plt.savefig(os.path.join(plot_dir, f"tmp.png"), bbox_inches='tight')
            plt.close()
            plot_img = cv2.imread(os.path.join(plot_dir, f"tmp.png"))
            cv2.imshow('Diff Images', plot_img)
            cv2.waitKey(1)
            ## Save Tracking Loss Viz
            # save_plot_dir = os.path.join(plot_dir, f"tracking_%04d" % iter_time_idx)
            # os.makedirs(save_plot_dir, exist_ok=True)
            # plt.savefig(os.path.join(save_plot_dir, f"%04d.png" % tracking_iteration), bbox_inches='tight')
            # plt.close()

        weighted_losses = {k: v * loss_weights[k] for k, v in losses.items()}
        loss = sum(weighted_losses.values())

        seen = radius > 0
        variables['max_2D_radius'][seen] = torch.max(radius[seen], variables['max_2D_radius'][seen])
        variables['seen'] = seen
        weighted_losses['loss'] = loss

        return loss, variables, weighted_losses

    def init_rgbd_slam(self):
        # Print Config
        if "use_depth_loss_thres" not in self.config['tracking']:
            self.config['tracking']['use_depth_loss_thres'] = False
            self.config['tracking']['depth_loss_thres'] = 100000
        if "visualize_tracking_loss" not in self.config['tracking']:
            self.config['tracking']['visualize_tracking_loss'] = False
        if "gaussian_distribution" not in self.config:
            self.config['gaussian_distribution'] = "isotropic"

        # Create Output Directories
        # self.output_dir = os.path.join(self.config["workdir"], self.config["run_name"])
        self.output_dir = os.path.join(self.workdir, self.config["run_name"])
        # eval_dir = os.path.join(output_dir, "eval")
        # os.makedirs(eval_dir, exist_ok=True)
        
        # Get Device
        self.device = torch.device(self.config["primary_device"])

        # Init agent_pose
        # self.agent_pos = dict(x=0, y=0, z=0)
        
        # Load Dataset
        # print("Loading Dataset ...")
        
        # dataset_config renamed to cam_config
        self.cam_config = self.config["cam_config"]
        self.cam_config["tracking_image_height"] = self.cam_config["image_height"]
        self.cam_config["tracking_image_width"] = self.cam_config["image_width"]
          
        # event = sim_c.step(dict(action="Initialize", renderDepthImage=True, renderSemanticSegmentation=False, gridSize=0.25, fieldOfView=90))
        # event = sim_c.step(action="GetReachablePositions")
        # reachable_positions = event.metadata["actionReturn"]
        
        # num_frames = config["n_steps"]
        self.num_frames = 10000

        # ai2thor_helper = AI2ThorHelper(desired_width=300, desired_height=300, png_depth_scale=1000, 
        #                                 fx=150.0, fy=150.0, cx=150.0, cy=150.0, max_depth=dataset_config['max_depth'], device=device)
        # position=reachable_positions[0]
        # rotation=dict(x=0, y=90, z=0)
        self.ros_helper = ROSHelper(desired_width=self.cam_config["desired_width"],
                                    desired_height=self.cam_config["desired_height"],
                                    ori_width=self.cam_config["image_width"],
                                    ori_height=self.cam_config["image_height"],
                                    png_depth_scale=self.cam_config["png_depth_scale"],
                                    fx=self.cam_config["fx"], fy=self.cam_config["fy"],
                                    cx=self.cam_config["cx"], cy=self.cam_config["cy"],
                                    max_depth=self.cam_config["max_depth"], device=self.device)

        self.tracking_dataset = None
        self.tracking_cam = None
        self.tracking_intrinsics = None

        # Initialize list to keep track of Keyframes
        self.keyframe_list = []
        self.keyframe_time_indices = []
        
        # Init Variables to keep track of ground truth poses and runtimes
        self.gt_w2c_all_frames = []
        self.tracking_iter_time_sum = 0
        self.tracking_iter_time_count = 0
        self.mapping_iter_time_sum = 0
        self.mapping_iter_time_count = 0
        self.tracking_frame_time_sum = 0
        self.tracking_frame_time_count = 0
        self.mapping_frame_time_sum = 0
        self.mapping_frame_time_count = 0

        rospy.loginfo("Init Finished!")
        

    def rgb_aligned_depth_odom_callback(self, rgb, aligned_depth, odom):
        # limit the frequency of processing by skipping 
        # callback_start_time = rospy.Time.now()

        if (rospy.Time.now() - self.prev_time).to_sec() < 1.0/self.map_rate:
            print("time elapsed since last depth rgb callback is: ", (rospy.Time.now() - self.prev_time).to_sec())
            print("skipping current depth image to get desired rate of ", self.map_rate)
            return
        else:
            self.prev_time = rospy.Time.now()
        
        ################################################
        # Splatam treats the first frame as Identity Rotation and 0 translation
        # But in simulation, we may have non-zero pose for the first frame
        # What being done here is to inherently keep the first pose,
        # and then expose identity to the splatam
        # Then whenever a new odom cames, we need to find the TF to the first pose
        # and then apply this TF to the splatam
        ################################################
        try:
            # Convert ROS image message to OpenCV image
            # Preserve depth values; ROSHelper applies cam_config["png_depth_scale"].
            color_img = bridge.imgmsg_to_cv2(rgb, "bgr8")
            color_img = color_img[:, :, [2, 1, 0]]
            depth_img = bridge.imgmsg_to_cv2(aligned_depth, desired_encoding="passthrough")
        except Exception as e:
            rospy.logerr(e)
            return

        # extract position and rpy from odom
        self.odom_stamp_ = [odom.header.stamp.secs, odom.header.stamp.nsecs]
        if self.use_odom_msg:
            odom = odom.pose
        self.odom_pos_ = np.array([odom.pose.position.x, odom.pose.position.y, odom.pose.position.z])
        self.odom_rpy_ = Rotation.from_quat([odom.pose.orientation.x, 
                                            odom.pose.orientation.y, 
                                            odom.pose.orientation.z, 
                                            odom.pose.orientation.w]).as_euler('xyz')
        self.odom_quat_ = np.array([odom.pose.orientation.x,
                                    odom.pose.orientation.y,
                                    odom.pose.orientation.z,
                                    odom.pose.orientation.w])

        # apply cam2body to odom_pos. cam_pos = body2world * cam2body
        body2world = np.eye(4)
        body2world[:3, :3] = Rotation.from_euler('xyz', self.odom_rpy_).as_matrix()
        body2world[:3, 3] = self.odom_pos_
        cam2world = np.dot(body2world, self.cam2body_)
        # apply cam2body_t 
        # cam2world[:3, 3] += self.cam2body_t_
        self.cam_pos_ = cam2world[:3, 3]
        self.cam_quat_ = Rotation.from_matrix(cam2world[:3, :3]).as_quat()
        # print("---------------------------------!!!!!!---------------")
        # print("Odom: ", self.odom_pos_)
        # print("Cam: ", self.cam_pos_)
        # print("odom_quat: ", self.odom_quat_)
        # print("odom rot:", self.odom_rpy_)

        # TODO: now we need to pass the pose to Submap class to create submap first,
        # 1. If it is first observation, we need to process it differently and save params to submap
        # 2. If it is not the first observation, we need to optimize the current frame and update the submap

        # Globally, we need to keep track of current submap index, if it changes, we need to unload the map from memory
        # and save it, and load the new submap to GPU memory.
        pose_vec = np.concatenate([self.cam_pos_, self.cam_quat_])
        
        # 1. If it is global first pose. set it in the map manager
        if self.first_observation:
            # Call to publish static tf. Splatam always treat the first pose as identity
            self.publish_map_tf(cam2world)
            self.first_cam2world = cam2world
            
            self.submap_manager.set_global_first_pose(pose_vec)

            # Initialize ros_helper, set first pose
            ori_color, ori_depth, intrinsics, pose = self.ros_helper.process_observation(self.cam_pos_, self.cam_quat_, color_img, depth_img, first=True)
            self.first_gs_pose = pose # Should be identity 

            self.params, self.variables, self.intrinsics, self.first_frame_w2c, self.cam = self.ros_initialize_first_timestep(ori_color, ori_depth, intrinsics, pose, self.num_frames,
                                                                                                self.config['scene_radius_depth_ratio'],
                                                                                                self.config['mean_sq_dist_method'],
                                                                                                gaussian_distribution=self.config['gaussian_distribution'])
            self.time_idx = 0

        print("param keys: ", self.params.keys())

        # Update submap anchor pose with latest_pose_path
        s_time = rospy.Time.now()
        if self.latest_pose_path is not None:
            self.submap_manager.update_anchor_pose(self.latest_pose_path)
        # TODO: Call the submap manager to update the local map. The automatic load and unload should be handled by the submap manager
        self.params, self.variables, self.map_idx, submap_first_obs = self.submap_manager.update_local_map(self.odom_stamp_, pose_vec, self.params, self.variables)
        if self.use_language_features:
            self.task_mutex.acquire()
            self.lfm.cluster_submaps(self.submap_manager, self.first_cam2world)
            self.task_mutex.release()
        print("Update local map time: ", (rospy.Time.now() - s_time).to_sec())

        # submap, first_observation_submap, self.map_idx = self.submap_manager.get_submap(pose_vec)
        # if self.prev_map_idx != None and self.prev_map_idx != self.map_idx:
            # Unload the previous map
            # self.submap_manager.unload_map(self.prev_map_idx)
        # self.prev_map_idx = self.map_idx

        # Also train on the first interation.
        # the main loop 
        if self.time_idx < self.num_frames:
            if self.first_observation:
                color = ori_color.permute(2, 0, 1) / 255
                depth = ori_depth.permute(2, 0, 1)
                if self.use_language_features:
                    feat_img, feat_orig_img = self.lfm.get_features(color)
                else:
                    feat_img = torch.zeros((PCA_DIM, color.shape[1], color.shape[2])).cuda()
                self.gt_w2c_all_frames.append(self.first_gs_pose)
                curr_gt_w2c = self.gt_w2c_all_frames
                curr_data = {'cam': self.cam, 'im': color, 'depth': depth, 'id': self.time_idx, 'intrinsics': self.intrinsics, 
                                'w2c': self.first_frame_w2c, 'iter_gt_w2c_list': curr_gt_w2c,
                                'feat_img': feat_img, 'submap_id': self.map_idx}
                tracking_curr_data = curr_data
                self.first_observation = False

            else:
                self.time_idx += 1
                # get the current observation, note the pose is the pose in the gs map frame (could be different from the odom pose)
                ori_color, ori_depth, _, gt_pose = self.ros_helper.process_observation(self.cam_pos_, self.cam_quat_, color_img, depth_img, first=False)

                # Process poses
                gt_w2c = torch.linalg.inv(gt_pose)
                # Process RGB-D Data
                color = ori_color.permute(2, 0, 1) / 255
                depth = ori_depth.permute(2, 0, 1)                
                if self.use_language_features:
                    feat_img, feat_orig_img = self.lfm.get_features(color)
                    if self.vlm_ctr > 10:
                        self.check_termination(ori_color, feat_orig_img, depth)
                        self.vlm_ctr = 0
                    self.vlm_ctr += 1
                else:
                    feat_img = torch.zeros((PCA_DIM, color.shape[1], color.shape[2])).cuda()
                self.gt_w2c_all_frames.append(gt_w2c)
                curr_gt_w2c = self.gt_w2c_all_frames
                # Optimize only current time step for tracking
                iter_time_idx = self.time_idx
                # Initialize Mapping Data for selected frame
                curr_data = {'cam': self.cam, 'im': color, 'depth': depth, 'id': iter_time_idx, 'intrinsics': self.intrinsics, 
                            'w2c': self.first_frame_w2c, 'iter_gt_w2c_list': curr_gt_w2c,
                            'feat_img': feat_img, 'submap_id': self.map_idx}
                
                # Initialize Data for Tracking
                tracking_curr_data = curr_data

            # Optimization Iterations
            num_iters_mapping = self.config['mapping']['num_iters']
            
            # Initialize the camera pose for the current frame
            if self.time_idx > 0:
                self.params = self.initialize_camera_pose(self.params, self.time_idx, forward_prop=self.config['tracking']['forward_prop'])
                # self.params = self.initialize_camera_pose(self.params, self.time_idx, forward_prop=self.config['tracking']['forward_prop'])

            #################################################################################
            # Tracking (No tracking for now!!!)
            tracking_start_time = rospy.Time.now()
            if self.time_idx > 0 and not self.config['tracking']['use_gt_poses']:
                # Reset Optimizer & Learning Rates for tracking
                optimizer = self.initialize_optimizer(self.params, self.config['tracking']['lrs'], tracking=True)
                # Keep Track of Best Candidate Rotation & Translation
                candidate_cam_unnorm_rot = self.params['cam_unnorm_rots'][..., self.time_idx].detach().clone()
                candidate_cam_tran = self.params['cam_trans'][..., self.time_idx].detach().clone()
                current_min_loss = float(1e20)
                # Tracking Optimization
                iter = 0
                do_continue_slam = False
                num_iters_tracking = self.config['tracking']['num_iters']
                progress_bar = tqdm(range(num_iters_tracking), desc=f"Tracking Time Step: {self.time_idx}")
                while True:
                    iter_start_time = rospy.Time.now()
                    # Loss for current frame
                    loss, self.variables, losses = self.get_loss(self.params, tracking_curr_data, self.variables, iter_time_idx, self.config['tracking']['loss_weights'],
                                                    self.config['tracking']['use_sil_for_loss'], self.config['tracking']['sil_thres'],
                                                    self.config['tracking']['use_l1'], self.config['tracking']['ignore_outlier_depth_loss'], tracking=True, 
                                                    plot_dir=eval_dir, visualize_tracking_loss=self.config['tracking']['visualize_tracking_loss'],
                                                    tracking_iteration=iter)
                    # Backprop
                    loss.backward()
                    # Optimizer Update
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    with torch.no_grad():
                        # Save the best candidate rotation & translation
                        if loss < current_min_loss:
                            current_min_loss = loss
                            candidate_cam_unnorm_rot = self.params['cam_unnorm_rots'][..., self.time_idx].detach().clone()
                            candidate_cam_tran = self.params['cam_trans'][..., self.time_idx].detach().clone()
                        # Report Progress TODO: remove report in real-time operation?
                        if self.config['report_iter_progress']:
                            report_progress(self.params, tracking_curr_data, iter+1, progress_bar, iter_time_idx, sil_thres=self.config['tracking']['sil_thres'], tracking=True)
                        else:
                            progress_bar.update(1)
                    # Update the runtime numbers
                    iter_end_time = rospy.Time.now()
                    self.tracking_iter_time_sum += (iter_end_time - iter_start_time).to_sec()
                    tracking_iter_time_count += 1
                    # Check if we should stop tracking
                    iter += 1
                    if iter == num_iters_tracking:
                        if losses['depth'] < self.config['tracking']['depth_loss_thres'] and self.config['tracking']['use_depth_loss_thres']:
                            break
                        elif self.config['tracking']['use_depth_loss_thres'] and not do_continue_slam:
                            do_continue_slam = True
                            progress_bar = tqdm(range(num_iters_tracking), desc=f"Tracking Time Step: {self.time_idx}")
                            num_iters_tracking = 2*num_iters_tracking
                        else:
                            break

                progress_bar.close()
                # Copy over the best candidate rotation & translation
                with torch.no_grad():
                    self.params['cam_unnorm_rots'][..., self.time_idx] = candidate_cam_unnorm_rot
                    self.params['cam_trans'][..., self.time_idx] = candidate_cam_tran
            ############################# NO TRACKING Case ###########################
            elif self.time_idx >= 0 and self.config['tracking']['use_gt_poses']:
                with torch.no_grad():
                    # Get the ground truth pose relative to frame 0
                    rel_w2c = curr_gt_w2c[-1]
                    # print("shape of rel_w2c: ", rel_w2c.shape)
                    rel_w2c_rot = rel_w2c[:3, :3].unsqueeze(0).detach()
                    rel_w2c_rot_quat = matrix_to_quaternion(rel_w2c_rot)
                    rel_w2c_tran = rel_w2c[:3, 3].detach()
                    # Update the camera parameters
                    # TODO: we need to use sampled poses here as cam pose. 
                    self.params['cam_unnorm_rots'][..., self.time_idx] = rel_w2c_rot_quat
                    self.params['cam_trans'][..., self.time_idx] = rel_w2c_tran
            # Update the runtime numbers
            tracking_end_time = rospy.Time.now()
            self.tracking_frame_time_sum += (tracking_end_time - tracking_start_time).to_sec()
            self.tracking_frame_time_count += 1

            ## TODO: remove this?
            # if self.time_idx == 0 or (self.time_idx+1) % self.config['report_global_progress_every'] == 0:
            #     try:
            #         # Report Final Tracking Progress
            #         progress_bar = tqdm(range(1), desc=f"Tracking Result Time Step: {self.time_idx}")
            #         with torch.no_grad():
            #             report_progress(self.params, tracking_curr_data, 1, progress_bar, iter_time_idx, sil_thres=self.config['tracking']['sil_thres'], tracking=True)
            #         progress_bar.close()
            #     except:
            #         ckpt_output_dir = os.path.join(self.config["workdir"], self.config["run_name"])
            #         save_params_ckpt(self.params, ckpt_output_dir, self.time_idx)
            #         print('Failed to evaluate trajectory.')

            #################################################################################
            # Densification & KeyFrame-based Mapping
            # Correct keyframe cam_poses using latest_optimized_pose_path
            self.update_keyframe_list()

            if self.time_idx == 0 or (self.time_idx+1) % self.config['map_every'] == 0:
                # Densification
                if self.config['mapping']['add_new_gaussians'] and self.time_idx > 0:
                    # Setup Data for Densification
                    densify_curr_data = curr_data
                    # Add new Gaussians to the scene based on the Silhouette
                    self.params, self.variables = self.add_new_gaussians(self.params, self.variables, densify_curr_data, 
                                                        self.config['mapping']['sil_thres'], self.time_idx,
                                                        self.config['mean_sq_dist_method'], self.config['gaussian_distribution'])
                    # post_num_pts = params['means3D'].shape[0]
                
                with torch.no_grad():
                    # Get the current estimated rotation & translation
                    curr_cam_rot = F.normalize(self.params['cam_unnorm_rots'][..., self.time_idx].detach())
                    curr_cam_tran = self.params['cam_trans'][..., self.time_idx].detach()
                    curr_w2c = torch.eye(4).cuda().float()
                    curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
                    curr_w2c[:3, 3] = curr_cam_tran
                    # Select Keyframes for Mapping
                    num_keyframes = self.config['mapping_window_size']-2
                    selected_keyframes = keyframe_selection_overlap(depth, curr_w2c, self.intrinsics, self.keyframe_list[:-1], num_keyframes)
                    selected_time_idx = [self.keyframe_list[frame_idx]['id'] for frame_idx in selected_keyframes]
                    if len(self.keyframe_list) > 0:
                        # Add last keyframe to the selected keyframes
                        selected_time_idx.append(self.keyframe_list[-1]['id'])
                        selected_keyframes.append(len(self.keyframe_list)-1)
                    # Add current frame to the selected keyframes
                    selected_time_idx.append(self.time_idx)
                    selected_keyframes.append(-1)
                    # Print the selected keyframes
                    print(f"\nSelected Keyframes at Frame {self.time_idx}: {selected_time_idx}")

                # Save params for gradient computation
                self.params['old_means3D'] = self.params['means3D'].clone().detach()
                self.params['update_means3D'] = torch.zeros(self.params['means3D'].shape[0]).cuda().float()
                self.params['noupdate_means3D'] = torch.zeros(self.params['means3D'].shape[0]).cuda().float()
                self.params['nochange_means3D'] = torch.zeros(self.params['means3D'].shape[0]).cuda().float()
                self.params['relevancy'] = torch.zeros(self.params['means3D'].shape[0]).cuda().float()
                # print("keys in params: ", self.params.keys())
                # Reset Optimizer & Learning Rates for Full Map Optimization
                optimizer = self.initialize_optimizer(self.params, self.config['mapping']['lrs'], tracking=False) 

                # Mapping --------------------------------------------------------------------from_sec----------
                mapping_start_time = rospy.Time.now()
                if num_iters_mapping > 0:
                    pass
                    # progress_bar = tqdm(range(num_iters_mapping), desc=f"Mapping Time Step: {self.time_idx}")
                # start = torch.cuda.Event(enable_timing=True)
                # end = torch.cuda.Event(enable_timing=True)
                # start.record()
                for iter in range(num_iters_mapping):
                    # if (rospy.Time.now() - callback_start_time).to_sec() > 1.0/self.map_rate:
                    #     break
                    iter_start_time = rospy.Time.now()
                    # Randomly select a frame until current time step amongst keyframes
                    rand_idx = np.random.randint(0, len(selected_keyframes))
                    selected_rand_keyframe_idx = selected_keyframes[rand_idx]
                    if selected_rand_keyframe_idx == -1:
                        # Use Current Frame Data
                        iter_time_idx = self.time_idx
                        iter_color = color
                        iter_depth = depth
                        iter_feat_img = feat_img
                    else:
                        # Use Keyframe Data
                        iter_time_idx = self.keyframe_list[selected_rand_keyframe_idx]['id']
                        iter_color = self.keyframe_list[selected_rand_keyframe_idx]['color']
                        iter_depth = self.keyframe_list[selected_rand_keyframe_idx]['depth']
                        iter_feat_img = self.keyframe_list[selected_rand_keyframe_idx]['feat_img']
                    iter_gt_w2c = self.gt_w2c_all_frames[:iter_time_idx+1]
                    iter_data = {'cam': self.cam, 'im': iter_color, 'depth': iter_depth, 'feat_img': iter_feat_img, 'id': iter_time_idx, 
                                'intrinsics': self.intrinsics, 'w2c': self.first_frame_w2c, 'iter_gt_w2c_list': iter_gt_w2c}
                    # Loss for current frame
                    loss, self.variables, losses = self.get_loss(self.params, iter_data, self.variables, iter_time_idx, self.config['mapping']['loss_weights'],
                                                    self.config['mapping']['use_sil_for_loss'], self.config['mapping']['sil_thres'],
                                                    self.config['mapping']['use_l1'], self.config['mapping']['ignore_outlier_depth_loss'], mapping=True)
                    if iter > num_iters_mapping - 10:
                        if self.use_language_features:
                            loss_feat = self.get_loss_feat_split(self.params, iter_data, self.variables, iter_time_idx, self.config['mapping']['loss_weights'],
                                                        self.config['mapping']['use_sil_for_loss'], self.config['mapping']['sil_thres'],
                                                        self.config['mapping']['use_l1'], self.config['mapping']['ignore_outlier_depth_loss'], mapping=True)                        
                            loss_feat.backward()
                        
                    # Backprop
                    loss.backward()
                    with torch.no_grad():
                        # Prune Gaussians
                        if self.config['mapping']['prune_gaussians']:
                            self.params, self.variables = prune_gaussians(self.params, self.variables, optimizer, iter, self.config['mapping']['pruning_dict'])
                        # Gaussian-Splatting's Gradient-based Densification
                        if self.config['mapping']['use_gaussian_splatting_densification']:
                            self.params, self.variables = densify(self.params, self.variables, optimizer, iter, self.config['mapping']['densify_dict'])
                        # Optimizer Update
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                        # Update gradients
                        if iter == num_iters_mapping-1:
                            # mean
                            old_means3D = self.params['old_means3D']
                            new_means3D = self.params['means3D'].clone()
                            update_means3D = torch.linalg.norm(new_means3D - old_means3D, dim=1)
                            self.params['grad_means3D'][update_means3D > 0] = update_means3D[update_means3D > 0]
                            update_means3D_pos = update_means3D[update_means3D > 0]
                            # compute update
                            means3D_std = torch.std(update_means3D_pos)
                            self.params['update_means3D'] = update_means3D > torch.mean(update_means3D_pos) + 0.5*means3D_std
                            self.params['noupdate_means3D'] = update_means3D <= torch.mean(update_means3D_pos) - 0.5*means3D_std
                            self.params['nochange_means3D'] = update_means3D == 0
                            if self.use_language_features:
                                self.task_mutex.acquire()
                                # compute relevancy
                                self.params['relevancy'] = self.lfm.compute_relevancy_ptcloud(self.params['features'].clone())
                                self.task_mutex.release()
                        # Report Progress
                        # if self.config['report_iter_progress']:
                        #     report_progress(self.params, iter_data, iter+1, progress_bar, iter_time_idx, sil_thres=self.config['mapping']['sil_thres'], 
                        #                         mapping=True, online_time_idx=self.time_idx)
                        # else:
                        #     progress_bar.update(1)
                            
                    # Update the runtime numbers
                    iter_end_time = rospy.Time.now()
                    self.mapping_iter_time_sum += (iter_end_time - iter_start_time).to_sec()
                    self.mapping_iter_time_count += 1
                # if num_iters_mapping > 0:
                #     progress_bar.close()
                
                # end.record()
                # torch.cuda.synchronize()
                # print("Time taken for mapping: ", start.elapsed_time(end)/1000.0)
                    
                # Update the runtime numbers
                mapping_end_time = rospy.Time.now()
                self.mapping_frame_time_sum += (mapping_end_time - mapping_start_time).to_sec()
                self.mapping_frame_time_count += 1

                # if self.time_idx == 0 or (self.time_idx+1) % self.config['report_global_progress_every'] == 0:
                #     try:
                #         # Report Mapping Progress
                #         progress_bar = tqdm(range(1), desc=f"Mapping Result Time Step: {self.time_idx}")
                #         with torch.no_grad():
                #             report_progress(self.params, curr_data, 1, progress_bar, self.time_idx, sil_thres=self.config['mapping']['sil_thres'], 
                #                                 mapping=True, online_time_idx=self.time_idx)
                #         progress_bar.close()
                #     except:
                #         ckpt_output_dir = os.path.join(self.config["workdir"], self.config["run_name"])
                #         save_params_ckpt(self.params, ckpt_output_dir, self.time_idx)
                #         print('Failed to evaluate trajectory.')
            
            # Add frame to keyframe list
            if ((self.time_idx == 0) or ((self.time_idx+1) % self.config['keyframe_every'] == 0) or \
                        (self.time_idx == self.num_frames-2)) and (not torch.isinf(curr_gt_w2c[-1]).any()) and (not torch.isnan(curr_gt_w2c[-1]).any()):
                with torch.no_grad():
                    # Get the current estimated rotation & translation
                    curr_cam_rot = F.normalize(self.params['cam_unnorm_rots'][..., self.time_idx].detach())
                    curr_cam_tran = self.params['cam_trans'][..., self.time_idx].detach()
                    curr_w2c = torch.eye(4).cuda().float()
                    curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
                    curr_w2c[:3, 3] = curr_cam_tran
                    # Initialize Keyframe Info
                    curr_keyframe = {'id': self.time_idx, 'est_w2c': curr_w2c, 'color': color, 'depth': depth,
                                     'feat_img': feat_img, "time_stamp": self.odom_stamp_}
                    # Add to keyframe list
                    self.keyframe_list.append(curr_keyframe)
                    self.keyframe_time_indices.append(self.time_idx)
            
            # Checkpoint every iteration
            # if self.time_idx % self.config["checkpoint_interval"] == 0 and self.config['save_checkpoints']:
            #     ckpt_output_dir = os.path.join(self.config["workdir"], self.config["run_name"])
            #     save_params_ckpt(self.params, ckpt_output_dir, self.time_idx)
            #     np.save(os.path.join(ckpt_output_dir, f"keyframe_time_indices{self.time_idx}.npy"), np.array(self.keyframe_time_indices))
            
            if not self.process_bag:
                # Keep a copy of latest self.params['means3D']. rgb_colors and log_scales
                if self.map_type_ != 3:
                    self.submap_manager.publish_all_submap_anchor_pose()
                    # Publish only local map
                    self.publish_map(downsample=self.subsample_)
                    self.publish_relevancy_map(downsample=self.subsample_)
                    self.publish_ring(self.cam_pos_)
                    # submap_cloud = self.submap_manager.create_submap_cloud(self.map_idx, downsample=self.subsample_, map_type=self.map_type_)
                    # self.map_pub.publish(submap_cloud)
                    # self.submap_manager.print_cuda_usage()
                    # self.submap_manager.print_all_submap_size()
                    self.submap_manager.log_submap_usage()
                start_time = rospy.Time.now()
                # TODO: update Gaussian Frontiers
                #self.topo_tree.odom_callback(odom)
                self.submap_manager.gen_planning_graph()
                self.global_planner.update_map(self.submap_manager.submaps)
                local_means3d = self.params['means3D'].detach()
                local_means3d = torch.cat([local_means3d, torch.ones(local_means3d.shape[0], 1).cuda()], dim=1)
                local_means3d = torch.from_numpy(self.first_cam2world).float().cuda() @ local_means3d.t()
                local_means3d = local_means3d.t()[:, :3]
                # curr_pose = self.first_cam2world @ np.concatenate([self.cam_pos_, np.array([1])])
                # curr_pose = curr_pose[:3]
                curr_pose = self.cam_pos_
                json_data = self.global_planner.global_planner_update(local_means3d, self.params['relevancy'].detach(), self.submap_manager, curr_pose, self.first_cam2world)
                # frontier_centers, frontier_utilities = get_gaussian_frontiers(self.params, self.ftr_cluster_size_, self.nms_radius_, self.filter_grad_mean_)
                rospy.loginfo("global_planner_update time: {} s".format( (rospy.Time.now() - start_time).to_sec()))
                start_time = rospy.Time.now()
                if not self.load_wp_from_json:
                    self.path_to_ftr = self.topo_tree.spin(json_data) # frontier_centers, frontier_utilities, self.first_cam2world, self.ftr_goal_tol_, self.fail_pos_tol_, self.fail_yaw_tol_)
                    rospy.loginfo("Planning time: {} s".format( (rospy.Time.now() - start_time).to_sec()))
                    if self.path_to_ftr is None or len(self.path_to_ftr) == 0:
                        rospy.logerr("No path to frontier found")
                    else:
                        self.pub_ftr_goal(self.path_to_ftr[-1])
                # Set planner map
                s_time = rospy.Time.now()
                self.params_copy = {key: tensor.detach().clone() for key, tensor in self.params.items() if key in ['means3D', 'log_scales', 'ground_labels']}
                self.local_planner.set_map(self.params_copy['means3D'], self.params_copy['log_scales'], self.params_copy['ground_labels'])
                rospy.loginfo("copy time: {} s".format( (rospy.Time.now() - s_time).to_sec()))

                # call planner!!!
                self.plan_cb(None)

            # ##########################
            # # temporary block to plan to highest relevancy gaussian
            # # get highest relevancy gaussian
            # relevancy = self.params['relevancy']
            # print('relevancy ptcloud', relevancy.min(), relevancy.max())
            # max_idx = torch.argmax(relevancy)
            # max_3d = self.params['means3D'][max_idx]
            # max_3d = max_3d.detach().cpu().numpy()
            # max_3d_world = self.first_cam2world[:3, :3] @ max_3d.reshape(3,1)
            # max_3d_world = max_3d_world.squeeze()
            # self.path_to_ftr = max_3d_world[:2].reshape(1,2)
            # self.pub_ftr_goal(self.path_to_ftr.squeeze())
            # ##########################
                        
            # self.local_planner.plan_to_ftr(self.params, self.intrinsics, path)
            del curr_data
            # del self.params
            # del self.variables
            gc.collect()
            torch.cuda.empty_cache()
            # torch gc


        # Compute Average Runtimes
        if self.tracking_iter_time_count == 0:
            self.tracking_iter_time_count = 1
            self.tracking_frame_time_count = 1
        if self.mapping_iter_time_count == 0:
            self.mapping_iter_time_count = 1
            self.mapping_frame_time_count = 1
        tracking_iter_time_avg = self.tracking_iter_time_sum / self.tracking_iter_time_count
        tracking_frame_time_avg = self.tracking_frame_time_sum / self.tracking_frame_time_count
        mapping_iter_time_avg = self.mapping_iter_time_sum / self.mapping_iter_time_count
        mapping_frame_time_avg = self.mapping_frame_time_sum / self.mapping_frame_time_count
        print(f"\nAverage Tracking/Iteration Time: {tracking_iter_time_avg*1000} ms")
        print(f"Average Tracking/Frame Time: {tracking_frame_time_avg} s")
        # print(f"Total mapping iter: {self.mapping_iter_time_count}")
        print(f"Average Mapping/Iteration Time: {mapping_iter_time_avg*1000} ms")
        print(f"Average Mapping/Frame Time: {mapping_frame_time_avg} s")
        
        # Evaluate Final Parameters
        ## TODO: think about how to do this with simulator
        # with torch.no_grad():
        #     evaluate_all_reachable_pose(first_pose, reachable_positions, ai2thor_helper, sim_c, params, num_frames, eval_dir, sil_thres=config['mapping']['sil_thres'],
        #             mapping_iters=config['mapping']['num_iters'], add_new_gaussians=config['mapping']['add_new_gaussians'],
        #             eval_every=config['eval_every'])

    def plan_cb(self, req):
        if self.params_copy is not None and self.path_to_ftr is not None and len(self.path_to_ftr) > 0:
            self.local_planner.plan_to_ftr(self.intrinsics, self.path_to_ftr, img_w=self.cam_config["desired_width"], img_h=self.cam_config["desired_height"])
            rospy.loginfo("plan to frontier")
        else:
            rospy.logerr("No path to frontier, skip planning callback")

    def opt_pose_path_cb(self, msg):
        # save the latest msg as local variable
        self.latest_pose_path = msg.poses


    def update_keyframe_list(self, match_threshold=0.05):
        # iterate through keyframe list. 
        # For each keyframe, using its timestamp
        # find the latest pose on latest_pose_path
        # update the keyframe with the latest pose
        if self.latest_pose_path is None:
            rospy.logerr("No opt pose path received yet")
            return
        pose_idx = 0
        for keyframe in self.keyframe_list:
            keyframe_time = keyframe["time_stamp"]
            pose = self.latest_pose_path[pose_idx]
            pose_time = [pose.header.stamp.secs, pose.header.stamp.nsecs]
            if keyframe_time[0] < pose_time[0] or (keyframe_time[0] == pose_time[0] and keyframe_time[1] < pose_time[1]):
                pose_idx += 1
            elif (keyframe_time[0] == pose_time[0] and keyframe_time[1] == pose_time[1]) or \
                 ((pose_time[0]-keyframe_time[0]) * 1e9 + (pose_time[1]-keyframe_time[1]) < match_threshold*1e9):
                # update
                new_odom_pos = np.array([pose.position.x, pose.position.y, pose.position.z])
                new_odom_rpy = Rotation.from_quat([pose.orientation.x, 
                                                    pose.orientation.y, 
                                                    pose.orientation.z, 
                                                    pose.orientation.w]).as_euler('xyz')
                body2world = np.eye(4)
                body2world[:3, :3] = Rotation.from_euler('xyz', new_odom_rpy).as_matrix()
                body2world[:3, 3] = new_odom_pos
                cam2world = np.dot(body2world, self.cam2body_)
                cam_pos = cam2world[:3, 3]
                cam_quat = Rotation.from_matrix(cam2world[:3, :3]).as_quat()
                mew_pose_vec = np.concatenate([cam_pos, cam_quat])
                # get relative pose
                relative_pose = self.ros_helper.process_pose(mew_pose_vec)
                rel_w2c = torch.linalg.inv(relative_pose)
                keyframe['est_w2c'][0:3, 3] = rel_w2c[0:3, 3]
                keyframe['est_w2c'][0:3, 0:3] = rel_w2c[0:3, 0:3]
                # move to next keyframe
                continue
            else:
                rospy.logerr("Keyframe list and pose path does not match")
                continue


    def save_map_cb(self, req):
        try:
            live_params = deepcopy(self.params)
            live_variables = {}
            for key, value in self.variables.items():
                live_variables[key] = value.clone().detach()
            all_params, num_submaps = self.submap_manager.get_all_submap_params(live_params)
            # Also generate individual submap params for saving.
            print("Generating submap params!")
            submap_params = self.submap_manager.flush_and_gen_submap_params(live_params, live_variables)

            # Scratch optimization buffers describe only the live subset.
            # They are not part of the persisted Gaussian representation.
            for key in self.submap_manager.unsave_params:
                all_params.pop(key, None)

            intrinsics = deepcopy(self.intrinsics)
            first_frame_w2c = deepcopy(self.first_frame_w2c)
            gt_w2c_all_frames = deepcopy(self.gt_w2c_all_frames)
            keyframe_time_indices = deepcopy(self.keyframe_time_indices)

            # Add Camera Parameters to Save them
            # Match the concatenation order used by get_all_submap_params:
            # live Gaussians first, then unloaded submaps in index order.
            timesteps = [live_variables['timestep'].detach().cpu()]
            for idx, submap in enumerate(self.submap_manager.submaps):
                if idx not in self.submap_manager.local_submaps and submap.params is not None:
                    timesteps.append(submap.variables['timestep'].detach().cpu())
            all_params['timestep'] = torch.cat(timesteps)
            all_params['intrinsics'] = intrinsics.detach().cpu().numpy()
            all_params['w2c'] = first_frame_w2c.detach().cpu().numpy()
            all_params['org_width'] = self.cam_config["image_width"]
            all_params['org_height'] = self.cam_config["image_height"]
            all_params['gt_w2c_all_frames'] = []
            for gt_w2c_tensor in gt_w2c_all_frames:
                all_params['gt_w2c_all_frames'].append(gt_w2c_tensor.detach().cpu().numpy())
            all_params['gt_w2c_all_frames'] = np.stack(all_params['gt_w2c_all_frames'], axis=0)
            all_params['keyframe_time_indices'] = np.array(keyframe_time_indices)
            all_params["submap_params"] = submap_params
            
            # add submap params
            # all_params['submap_cluster_params'] = []
            # for submap in self.submap_manager.submaps:
            #     submap_params = {}
            #     submap_params['cluster_means'] = submap.cluster_means
            #     submap_params['cluster_feats'] = submap.cluster_feats
            #     submap_params['cluster_pts'] = submap.cluster_pts
            #     submap_params['cluster_labels'] = submap.cluster_labels
            #     submap_params['tree'] = submap.tree
            #     all_params['submap_cluster_params'].append(submap_params)

            # Save Parameters
            save_params(all_params, self.output_dir)
            return TriggerResponse(success=True, message="Map saved successfully to {}".format(self.output_dir))
        except Exception as e:
            rospy.logerr(f"Failed to save map: {e}")
            return TriggerResponse(success=False, message=f"Failed to save info: {e}")


    def publish_map_tf(self, first_cam2world):
        # Publish the map tf
        t = TransformStamped()
        t.header.stamp = rospy.Time.now()
        t.header.frame_id = "world"
        t.child_frame_id = "gs_map"
        t.transform.translation.x = first_cam2world[0, 3]
        t.transform.translation.y = first_cam2world[1, 3]
        t.transform.translation.z = first_cam2world[2, 3]
        quat = Rotation.from_matrix(first_cam2world[:3, :3]).as_quat()
        t.transform.rotation.x = quat[0]
        t.transform.rotation.y = quat[1]
        t.transform.rotation.z = quat[2]
        t.transform.rotation.w = quat[3]
        self.tf_pub.sendTransform(t)
        rospy.loginfo("Published map static tf!")


    def make_fields(self):
        fields = []
        field = PointField()
        field.name = 'x'
        field.count = 1
        field.offset = 0
        field.datatype = PointField.FLOAT32
        fields.append(field)

        field = PointField()
        field.name = 'y'
        field.count = 1
        field.offset = 4
        field.datatype = PointField.FLOAT32
        fields.append(field)

        field = PointField()
        field.name = 'z'
        field.count = 1
        field.offset = 8
        field.datatype = PointField.FLOAT32
        fields.append(field)

        field = PointField()
        field.name = 'rgb'
        field.count = 1
        field.offset = 12
        field.datatype = PointField.UINT32
        fields.append(field)

        field = PointField()
        field.name = 'size'
        field.count = 1
        field.offset = 16
        field.datatype = PointField.FLOAT32
        fields.append(field)
        return fields


    def publish_map(self, downsample=20):
        s_time = rospy.Time.now()
        # Publish the GS points as PointCloud2
        
        original_gs_points = self.params['means3D'].detach().cpu().numpy()
        gs_ground_labels = self.params['ground_labels'].detach().cpu().numpy()
        if self.map_type_ == 1:
            original_gs_points = original_gs_points[gs_ground_labels == 0]
        elif self.map_type_ == 2:
            original_gs_points = original_gs_points[gs_ground_labels == 1]
        # print("shape of latest_means3D: ", self.latest_means3D.shape)
        gs_points = original_gs_points[0:original_gs_points.shape[0]:downsample]
        
        # convert to ros point cloud
        # print("shape of gs_points: ", gs_points.shape)
        gs_colors = self.params['rgb_colors'].detach().cpu().numpy()
        gs_colors = gs_colors[0:gs_colors.shape[0]:downsample]

        # print("shape of gs_colors: ", gs_colors.shape)
        original_gs_sizes = self.params['log_scales'].detach().cpu().numpy()
        gs_sizes = original_gs_sizes[0:original_gs_sizes.shape[0]:downsample]

        # take exponential of the log scales
        gs_sizes = np.exp(gs_sizes)
        # print("shape of gs_sizes: ", gs_sizes.shape)

        # create the point cloud
        rgb_data = np.array([
                (int(b * 255) << 16) | (int(g * 255) << 8) | int(r * 255)
                for b, g, r in gs_colors], dtype=np.uint32)                

        point_data = [[None] *5] * len(gs_points)
        for i in range(len(gs_points)):
            point_data[i] = [gs_points[i][0], gs_points[i][1], gs_points[i][2], int(rgb_data[i]), gs_sizes[i]]
        # point_data = np.concatenate((gs_points, rgb_data[:, np.newaxis], gs_sizes), axis=1).tolist()
        # change rgb_data in point_data to int
        # for i in range(len(point_data)):
        #     point_data[i][3] = int(point_data[i][3])

        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "gs_map"
        cloud = point_cloud2.create_cloud(header, self.pc_fields_, point_data)
        self.map_pub.publish(cloud)
        e_time = rospy.Time.now()
        rospy.loginfo(f"Published GS Map! Time: {(e_time-s_time).to_sec()} s")

        # Also publish GS Points as GaussianPoints
        # s_time = rospy.Time.now()
        # gs_msg = GaussianPoints()
        # gs_true_sizes = np.exp(original_gs_sizes)
        # for i in range(len(original_gs_points)):
        #     # Publish Gaussian Points
        #     pt = PointWithSize()
        #     pt.x = original_gs_points[i][0]
        #     pt.y = original_gs_points[i][1]
        #     pt.z = original_gs_points[i][2]
        #     pt.size = gs_true_sizes[i]
        #     gs_msg.points.append(pt)
        # self.gaussian_pub.publish(gs_msg)
        # e_time = rospy.Time.now()
        # rospy.loginfo(f"Published GS Points! Time: {(e_time-s_time).to_sec()} s")
        

    def publish_ring(self, curr_pose):
        marker = Marker()
        marker.header.frame_id = "world"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "sphere"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = curr_pose[0]
        marker.pose.position.y = curr_pose[1]
        marker.pose.position.z = 0
        marker.pose.orientation.w = 1.0
        marker.scale.x = 20.0  # Diameter of 20m (radius of 10m)
        marker.scale.y = 20.0
        marker.scale.z = 0.1
        marker.color.g = 1.0
        marker.color.a = 0.2        
        
        self.ring_pub.publish(marker)


    def publish_relevancy_map(self, downsample=20):
        s_time = rospy.Time.now()
        gs_ground_labels = self.params['ground_labels'].detach().cpu().numpy()
        relevancy = self.params['relevancy'].detach().cpu().numpy()
        relevancy = relevancy[gs_ground_labels == 0]
        relevancy = relevancy[::downsample]
        relevancy_points = self.params['means3D'].detach().cpu().numpy()
        relevancy_points = relevancy_points[gs_ground_labels == 0]
        relevancy_points = relevancy_points[::downsample]
        relevancy_colors = plt.cm.coolwarm(relevancy)[:, :3] * 255
        # create the point cloud
        rgb_data = np.array([
                (int(b) << 16) | (int(g) << 8) | int(r)
                for b, g, r in relevancy_colors], dtype=np.uint32)                
        
        original_gs_sizes = self.params['log_scales'].detach().cpu().numpy()
        gs_sizes = original_gs_sizes[gs_ground_labels == 0]
        gs_sizes = gs_sizes[::downsample]

        # take exponential of the log scales
        gs_sizes = np.exp(gs_sizes)
        
        point_data = np.concatenate((relevancy_points, rgb_data[:, np.newaxis], gs_sizes), axis=1).tolist()
        # change rgb_data in point_data to int
        for i in range(len(point_data)):
            point_data[i][3] = int(point_data[i][3])

        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "gs_map"
        cloud = point_cloud2.create_cloud(header, self.pc_fields_, point_data)
        self.relevancy_pub.publish(cloud)
        e_time = rospy.Time.now()
        rospy.loginfo(f"Published Relevancy Map! Time: {(e_time-s_time).to_sec()} s")
    
    def pub_ftr_goal(self, goal_pos):
        # Compose goal msg
        goal_msg = PoseStamped()
        goal_msg.header.stamp = rospy.Time.now()
        goal_msg.header.frame_id = "world"
        goal_msg.pose.position.x = goal_pos[0]
        goal_msg.pose.position.y = goal_pos[1]
        goal_msg.pose.position.z = 0
        goal_msg.pose.orientation.x = 0
        goal_msg.pose.orientation.y = 0
        goal_msg.pose.orientation.z = 0
        goal_msg.pose.orientation.w = 1.0
        self.ftr_goal_pub.publish(goal_msg)

    def move_base_result_cb(self, msg):
        if msg.status.status == 3:
            rospy.loginfo("Goal reached!")
            self.move_base_goal_reached = True
            
            self.visited_frontiers.append(self.prev_goal)
            
    def set_task_cb(self, req):
        self.task_mutex.acquire()
        rospy.loginfo(f"Setting task...")
        # get prompt string
        prompt = req.task
        if ',' in prompt:
            prompts = prompt.split(',')
            prompt_task = prompts[0]
            prompt_background = prompts[1]
            if len(prompts) == 3:
                nms_radius = float(prompts[2])
                self.global_planner.nms_radius = nms_radius
                rospy.loginfo("Set nms radius to: {}".format(nms_radius))
        else:
            prompt_task = prompt
            prompt_background = 'background'
        self.lfm.set_prompt(prompt)
        self.lfm.recompute_submap_relevancy(self.submap_manager, self.first_cam2world)
        # reset topo tree odom history
        self.topo_tree.odom_history = None
        rospy.loginfo(f"Successfully set task to: {prompt_task}, {prompt_background}")
        self.task_mutex.release()
        return True, "Task set to '{}'".format(prompt)
    

    def load_wp(self, wp_json_path):
        with open(wp_json_path, 'r') as f:
            wp = json.load(f)
        tmp_path = []
        for pos in wp:
            tmp_path.append([pos['x'], pos['y']])
        return np.array(tmp_path)
    
    def check_termination(self, color, feat, depth):
        vlm_msg = self.lfm.check_termination(color, feat, depth)
        if vlm_msg is not None:
            msg = RosString()
            msg.data = vlm_msg
            self.termination_pub.publish(msg)
        


####################################################################################################
####################################################################################################
