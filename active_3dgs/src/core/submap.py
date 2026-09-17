import numpy as np
from scipy.spatial.transform import Rotation as R
import rospy
import torch
from sensor_msgs.msg import Image, PointCloud2, PointField
from sensor_msgs import point_cloud2
from std_msgs.msg import Header, String
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA

# For logging
from datetime import datetime
import sys
import os
from scipy.spatial.transform import Rotation

import multiprocessing as mp
from core.clustering import cluster_submap_worker, cluster_submap
import networkx as nx
import math
from networkx.readwrite import json_graph
import json


class Submap:
    def __init__(self, time_stamp, anchor_pose, submap_size):
        # Note that anchor poses are already camera poses, not body poses
        self.gpu = True  # Whether the submap is on GPU
        self.anchor_pose = anchor_pose # size 7 vector
        self.anchor_h = np.eye(4)
        self.anchor_h[:3, :3] = Rotation.from_quat(self.anchor_pose[3:]).as_matrix()
        self.anchor_h[:3, 3] = self.anchor_pose[:3]
        self.anchor_h_inv = np.linalg.inv(self.anchor_h)
        self.anchor_point = anchor_pose[:3] # Extract the 3D anchor point (translation part)
        self.params = None  # Any additional parameters # Dict
        self.variables = None # variables # Dict
        self.first_frame_pose = anchor_pose # The 6DoF first frame pose (4x4 matrix)
        self.size = submap_size # The size of the submap (threshold distance)
        self.updated = False
        self.cluster_means = []
        self.cluster_feats = []
        self.cluster_pts = []
        self.cluster_labels = []
        self.regions = []
        self.waypoints = []
        self.tree = None
        self.time_stamp = time_stamp

    def get_params(self):
        return self.params
    
    def get_variables(self):
        return self.variables
    
    def update_params(self, params):
        self.params = params
    
    def update_variables(self, variables):
        self.variables = variables

    def get_anchor_pose(self):
        return self.anchor_pose

    def get_anchor_h(self):
        return self.anchor_h

class SubmapManager:
    def __init__(self, submap_distance, submap_size, cam2body, local_map_radius=5):
        self.submap_distance = submap_distance
        self.submap_size = submap_size  # Size of each submap (distance threshold)
        self.submaps = []  # List to store submaps
        self.global_first_pose_set = False  # Whether the global first pose is set
        self.global_first_pose = None  # The first frame pose in the global frame
        self.cam2body = cam2body  # Transformation matrix from camera to body frame

        self.local_map_radius = local_map_radius  # size of local radius to load submaps
        self.pc_fields_ = self._make_pc_fields()

        self.local_submaps = []  # store list of submap id (idx) that is being loaded in GPU
        
        self.pool = mp.Pool(processes=2)
        self.submap_objs = {} # objects of each submap
        self.use_relative_pose = True
        self.logging = False
        self.graph = nx.Graph()
        self.previous_position = None

        self.special_params = ['cam_unnorm_rots', 'cam_trans']
        self.unsave_params = ['old_means3D', 'update_means3D', 'noupdate_means3D', 'old_log_scales', 'update_log_scales', 'noupdate_log_scales', 'nochange_means3D']
        self.special_vars = ['scene_radius']

        # Publishers
        self.anchor_pose_pub = rospy.Publisher("submap_anchors", Marker, queue_size=10)
        self.nx_graph_pub = rospy.Publisher("nx_graph", String, queue_size=10)

        # Use a txt file to store statistics
        if self.logging:
            self.log_fname = os.path.join(rospy.get_param("~workdir", os.path.expanduser("~/.ros/active_3dgs")), "submap_stats.txt")
            # create dir if doesn't exit
            dir = os.path.dirname(self.log_fname)
            if not os.path.exists(dir):
                os.makedirs(dir)
            self.debug_stats_file = open(self.log_fname, "a")
            # add datetime to the file
            self.debug_stats_file.write(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]\n")
            self.debug_stats_file.close()
            self.log_start_time = rospy.Time.now()
            self.log_dict = {"Time elapsed (s)": 0, "Total # submaps": 0, "Total submaps on GPU": 0, "CUDA allocated mem (MB)": 0, "CUDA reserved mem (MB)": 0, "Total submap usage (MB)": 0}
            
    def create_submap(self, stamp, anchor_pose):
        # Create a new submap dictionary with a 4x4 transformation matrix as anchor pose
        # submap = {
        #     'gpu': False,  # Whether the submap is on GPU
        #     'anchor_pose': anchor_pose,  # size 7 vector
        #     'anchor_point': anchor_pose[:3],  # Extract the 3D anchor point (translation part)
        #     'params': None,  # Any additional parameters
        #     'variables': None,  # variables
        #     'first_frame_pose': anchor_pose,  # The 6DoF first frame pose (4x4 matrix)
        #     'size': self.submap_size  # The size of the submap (threshold distance)
        # }
        submap = Submap(stamp, anchor_pose, self.submap_size)
        self.submaps.append(submap)
        return len(self.submaps) - 1  # Return the index of the new submap
        
    # def distance(self, position, anchor_point):
    #     # Compute Euclidean distance between the 3D positions
    #     return np.linalg.norm(position - anchor_point)

    # def _get_map_idx(self, new_pose):
    #     # Extract XYZ position from the new pose matrix
    #     new_position = new_pose[:3]
    #     # Check if the new position is within the range of an existing submap
    #     for idx, submap in enumerate(self.submaps):
    #         anchor_point = submap.anchor_point
    #         if self.distance(new_position, anchor_point) <= self.submap_distance:
    #             return idx
    #     return None

    # def get_submaps(self):
    #     # Returns all submaps
    #     return self.submaps
    
    # def get_submap(self, pose):
    #     # input is 3+4 pose vector
    #     # 1. Decide if the new frame should be added to an existing submap or create a new submap
    #     # TODO: we can use any tree structure to speed up the search

    #     # First check if the global first pose is set
    #     if not self.global_first_pose_set:
    #         self.set_global_first_pose(pose)
    #         self.global_first_pose_set = True

    #     first_observation = False
    #     map_idx = self._get_map_idx(pose)
    #     if map_idx == None:
    #         map_idx = self.create_submap(pose)
    #         first_observation = True
    #     return self.submaps[map_idx], first_observation, map_idx
    #     # 2. If the new frame is added to an existing submap, update the submap parameters
    #     # 3. If a new submap is created, add the new frame to the submap


    def update_local_map(self, stamp, pose, params, variables):
        # 1. find local submap ids. If needed, also create a new one
        curr_id, within_ids, new_submap = self.get_local_submap_id(stamp, pose)

        # 2. compare if ids has changed
        # in local_submaps but not in within_ids
        unload_ids = np.setdiff1d(self.local_submaps, within_ids)
        # in within_ids but not in local_submaps
        load_ids = np.setdiff1d(within_ids, self.local_submaps)
        
        # 3. if changed
        # (a) load changed submaps in local region to GPU
        if len(load_ids) > 0:
            for load_id in load_ids:
                self.load_map(load_id)
        
        # (b) unload submaps that are not in local region, this involves finding the params with the submap_id, and detach them from GPU
        if len(unload_ids) > 0:
            for unload_id in unload_ids:
                # find the submap_id in params
                # for key, val in params.items():
                #     print("size of key: ", key, val.size())
                unload_mask = params['submap_id'] == unload_id
                # we use nochange_means3D to check if we need to update the submap.
                if (self.submaps[unload_id].params is None
                        or self.submaps[unload_id].variables is None
                        or torch.any(params['nochange_means3D'][unload_mask] == 0)):
                    if self.use_relative_pose:
                        print("Use relative pose!!!")
                        # means3D needs to be transformed to anchor frame
                        # means = anchor_h_inv @ means3D
                        means_homo = torch.cat((params['means3D'][unload_mask], torch.ones(params['means3D'][unload_mask].size(0), 1).cuda()), dim=1)
                        print("shape of means_homo: ", means_homo.shape)
                        unload_means = torch.matmul(torch.from_numpy(self.submaps[unload_id].anchor_h_inv @ self.world_from_gs).float().cuda(), means_homo.T)
                        unload_means = unload_means[:3, :].T.detach().cpu()
                        unload_params = {
                            key: val[unload_mask].detach().cpu()
                            for key, val in params.items()
                            if key not in self.special_params and key not in self.unsave_params and key != 'means3D'
                        }
                        unload_params['means3D'] = unload_means
                    else:
                        # extract the params and variables
                        unload_params = {
                            key: val[unload_mask].detach().cpu() 
                            for key, val in params.items() 
                            if key not in self.special_params and key not in self.unsave_params
                        }
                    # for key, val in variables.items():
                    #     print("size of var key: ", key, val.size())
                    unload_variables = {
                        key: val[unload_mask].detach().cpu() 
                        for key, val in variables.items() 
                        if key not in self.special_vars
                    }


                    # update submap
                    self.update_submap_params(unload_id, unload_params)
                    self.update_submap_variables(unload_id, unload_variables)
                    self.submaps[unload_id].gpu = False
                    self.submaps[unload_id].updated = True
                else:
                    # if nochange_means3D == 1, we can unload the submap
                    self.unload_map(unload_id)
                    self.submaps[unload_id].updated = False

                # delete the unloaded params and variables
                # params = {key: val[~unload_mask] for key, val in params.items()}
                # variables = {key: val[~unload_mask] for key, val in variables.items()}
                for key, val in params.items():
                    if key not in self.special_params:
                        params[key] = val[~unload_mask]
                    else:
                        params[key] = val
                for key, val in variables.items():
                    if key not in self.special_vars:
                        variables[key] = val[~unload_mask]
                    else:
                        variables[key] = val
    
        # (c) update the list of local submaps
        self.local_submaps = within_ids

        # 4. TODO: transform all changed to global frame using anchor pose and global pose
        #    Note, now all GSs are saved in global frame. so just add them to params and variables
        for load_id in load_ids:
            if self.submaps[load_id].params is None:
                continue
            for k, v in self.submaps[load_id].params.items():
                # if k = means3D and use_relative_pose, we need to transform it to global frame
                if k in ["means3D"]:
                    if self.use_relative_pose:
                        print("[Load] Use relative pose!!!")
                        # Make v homogeneous
                        v = torch.cat((v, torch.ones(v.size(0), 1).cuda()), dim=1)  # N x 4)
                        v = torch.matmul(torch.from_numpy(self.gs_from_world @ self.submaps[load_id].anchor_h).float().cuda(),v.T)
                        v = v[:3, :].T  # (N x 3)
                    # add to params
                    params[k] = torch.nn.Parameter(torch.cat((params[k], v), dim=0))
                elif k in ["update_means3D", "noupdate_means3D", "nochange_means3D"]: # should not happen now
                    params[k] = torch.nn.Parameter(torch.cat((params[k], v), dim=0), requires_grad=False)
                else:
                    params[k] = torch.nn.Parameter(torch.cat((params[k], v), dim=0))
                    print("Is tensor on GPU? ", params[k].is_cuda)
            for k, v in self.submaps[load_id].variables.items():
                variables[k] = torch.cat((variables[k].detach(), v.detach()), dim=0)
        
        # 5. return the latest param and variables, and current_id (for init new gaussians)        
        return params, variables, curr_id, new_submap


    def get_local_submap_id(self, stamp, pose):
        # Returns all submaps that are within the distance threshold of the current pose
        # First check if the global first pose is set
        if not self.global_first_pose_set:
            self.set_global_first_pose(pose)
            self.global_first_pose_set = True

        # 2. Find closest submap, and the distance to the anchor point
        curr_submap_id = None
        new_submap = False
        # first_observation = False
        if len(self.submaps) == 0:
            # Create a new submap
            map_idx = self.create_submap(stamp, pose)
            curr_submap_id = map_idx
            new_submap = True
            # first_observation = True
            return curr_submap_id, np.array([curr_submap_id]), new_submap

        # Else, first find the closest submap
        submap_distances = np.array([np.linalg.norm(pose[:3] - submap.anchor_point) for submap in self.submaps])

        closest_id = np.argmin(submap_distances)
        closest_distance = submap_distances[closest_id]
        # Check if the closest submap is within the local map size
        if closest_distance <= self.submap_distance:
            curr_submap_id = closest_id
            new_submap = False
            # first_observation = False
        else:
            # Create a new submap
            map_idx = self.create_submap(stamp, pose)
            curr_submap_id = map_idx
            new_submap = True
            # first_observation = True

        # 3. Find all submaps in local region
        local_submap_ids = []
        within_ids = np.where(submap_distances <= self.local_map_radius)[0]
        # Note: may not include the current submap
        if curr_submap_id not in within_ids:
            within_ids = np.append(within_ids, curr_submap_id)
        
        return curr_submap_id, within_ids, new_submap
    

    def get_submaps(self, ids):
        # ids can be either a single id or a list of ids
        if isinstance(ids, int):
            return self.submaps[ids]
        else:
            return [self.submaps[id] for id in ids]


    def load_local_submaps(self):
        # Load all submaps in the local region to GPU
        # transform
        for submap_id in self.local_submaps:
            self.load_map(submap_id)

    def get_all_submap_params(self, live_params):
        # First, put all val in live_params to CPU
        for key, val in live_params.items():
            if val.is_cuda:
                live_params[key] = val.detach().cpu()
        # concat thing into live_params
        for i in range(len(self.submaps)):
            # Only add submaps that are not in local_submaps
            if i not in self.local_submaps:
                submap = self.submaps[i]
                if submap.params is not None:
                    for key, val in submap.params.items():
                        # make copy of val to cpu and cat 
                        if val.is_cuda:
                            val_save = val.clone().detach().cpu()
                        else:
                            val_save = val.clone().detach()
                        # If means3D, transform to global frame
                        if key == "means3D" and self.use_relative_pose:
                            means_homo = torch.cat((val_save, torch.ones(val_save.size(0), 1)), dim=1)
                            means_world = torch.matmul(torch.from_numpy(self.gs_from_world @ submap.anchor_h).float(), means_homo.T)
                            val_save = means_world[:3, :].T
                        if key in live_params:
                            live_params[key] = torch.cat((live_params[key], val_save), dim=0)
                        else:
                            live_params[key] = val_save
        return live_params, len(self.submaps)
        
    def flush_and_gen_submap_params(self, live_params, live_variables):
        # Now, unload live_params to all submaps and also save submaps individually
        # save params, id, anchor pose. 
        # Get the unique id from params["submap_id"]
        unique_ids = torch.unique(live_params["submap_id"]).tolist()
        for s_id in unique_ids:
            submap_mask = live_params["submap_id"] == s_id
            # Submap changed, unload
            if self.use_relative_pose:
                # means3D needs to be transformed to anchor frame
                # means = anchor_h_inv @ means3D
                means_homo = torch.cat((live_params['means3D'][submap_mask].cuda(), torch.ones(live_params['means3D'][submap_mask].size(0), 1).cuda()), dim=1)
                unload_means = torch.matmul(torch.from_numpy(self.submaps[int(s_id)].anchor_h_inv @ self.world_from_gs).float().cuda(), means_homo.T)
                unload_means = unload_means[:3, :].T
                unload_params = {
                    key: val[submap_mask]
                    for key, val in live_params.items()
                    if key not in self.special_params and key not in self.unsave_params and key != 'means3D'
                }
                unload_params['means3D'] = unload_means
            else:
                # extract the params and variables
                unload_params = {
                    key: val[submap_mask]
                    for key, val in live_params.items() 
                    if key not in self.special_params and key not in self.unsave_params
                }
            # unload_variables = {
            #     key: val[submap_mask] 
            #     for key, val in live_variables.items() 
            #     if key not in self.special_vars
            # }

            # update submap
            self.update_submap_params(int(s_id), unload_params)
            # self.update_submap_variables(int(s_id), unload_variables)

        # Now, go through all submaps, generate a dictionary to save them:
        all_submap_params = {}
        all_submap_params["relative_pose"] = self.use_relative_pose
        for i in range(len(self.submaps)):
            submap = self.submaps[i]
            if submap.params is not None:
                all_submap_params[i] = {}
                # additional submap info we want to save
                all_submap_params[i]['anchor_pose'] = submap.anchor_pose
                all_submap_params[i]['anchor_h'] = submap.anchor_h
                all_submap_params[i]['size'] = submap.size
                all_submap_params[i]['params'] = {}
                for key, val in submap.params.items():
                    if val.is_cuda:
                        all_submap_params[i]['params'][key] = val.detach().cpu().numpy()
                    else:
                        all_submap_params[i]['params'][key] = val.numpy()
        return all_submap_params



    def set_global_first_pose(self, pose):
        self.global_first_pose = pose
        self.world_from_gs = np.eye(4)
        self.world_from_gs[:3, :3] = Rotation.from_quat(pose[3:]).as_matrix()
        self.world_from_gs[:3, 3] = pose[:3]
        self.gs_from_world = np.linalg.inv(self.world_from_gs)

    # Customize what to unload from GPU
    def _detach_tensors_on_gpu(self, submap):
        if submap.params is None or submap.variables is None:
            return
        # check if the submap is on GPU
        if submap.params is not None:
            for key, val in submap.params.items():
                if torch.is_tensor(val) and val.is_cuda:
                    submap.params[key] = val.detach().cpu()
                    print("Detaching {} from GPU".format(key))
                    print("After detach, check if on gpu: ", submap.params[key].is_cuda)
        if submap.variables is not None:
            for key, val in submap.variables.items():
                if torch.is_tensor(val) and val.is_cuda:
                    submap.variables[key] = val.detach().cpu()
        if torch.is_tensor(submap.params) and submap.params.is_cuda:
            # Detach the tensor from GPU and move it to CPU
            submap.params = submap.params.detach().cpu()
        if torch.is_tensor(submap.variables) and submap.variables.is_cuda:
            # Detach the tensor from GPU and move it to CPU
            submap.variables = submap.variables.detach().cpu()
        submap.gpu = False
        # Iterate through the dictionary
        # for key, value in data_dict.items():
        #     # Check if the value is a torch tensor and if it's on GPU
        #     if torch.is_tensor(value) and value.is_cuda:
        #         # Detach the tensor from GPU and move it to CPU
        #         data_dict[key] = value.detach().cpu()
        # return data_dict

    def _load_tensors_on_gpu(self, submap):
        if submap.params is None or submap.variables is None:
            return
        # if not torch.is_tensor(submap.params) or not torch.is_tensor(submap.variables):
        #     print("Submap tensors are not initialized!")
        #     return
        if submap.params is not None:
            for key, val in submap.params.items():
                if torch.is_tensor(val) and not val.is_cuda:
                    submap.params[key] = val.cuda()
                    print("Loading {} to GPU".format(key))
                    print("After load, check if on gpu: ", submap.params[key].is_cuda)
        if submap.variables is not None:
            for key, val in submap.variables.items():
                if torch.is_tensor(val) and not val.is_cuda:
                    submap.variables[key] = val.cuda()
        # check if the submap is on GPU
        if torch.is_tensor(submap.params) and not submap.params.is_cuda:
            # Load the tensor to GPU
            submap.params = submap.params.cuda()
        else:
            print("Submap params is already on GPU!")
        if torch.is_tensor(submap.params) and not submap.variables.is_cuda:
            submap.variables = submap.variables.cuda()
        else:
            print("Submap variables is already on GPU!")
        submap.gpu = True

    def unload_map(self, idx):
        # Unload a specific submap from GPU memory
        map = self.submaps[idx]
        if map.gpu:
            # detach map from GPU memory
            self._detach_tensors_on_gpu(map)
        else:
            print("Submap is not on GPU!")
        # TODO: whether we need to save map to disk

    def load_map(self, idx):
        # Load map to GPU memory
        map = self.submaps[idx]
        if not map.gpu:
            # load map to GPU memory
            self._load_tensors_on_gpu(map)
        else:
            print("Submap is already on GPU!")

    def update_submap_params(self, idx, params):
        # Update the parameters of a specific submap
        self.submaps[idx].update_params(params)
        # self.submaps[idx].gpu = True

    def update_submap_variables(self, idx, variables):
        # Update the variables of a specific submap
        self.submaps[idx].update_variables(variables)
        # self.submaps[idx].gpu = True

    def get_submap_to_global_tf_matrix(self, idx):
        # Get the transformation matrix from submap to global frame
        submap = self.submaps[idx]
        anchor_pose = submap.anchor_pose
        anchor_pos = anchor_pose[:3]
        anchor_quat = anchor_pose[3:]
        # Create a 4x4 transformation matrix from the anchor pose
        cam2world = np.eye(4)
        cam2world[:3, :3] = R.from_quat(anchor_quat).as_matrix()
        cam2world[:3, 3] = anchor_pos
        return cam2world
    

    def publish_all_submap_anchor_pose(self):
        # Create a marker for anchor poses
        marker = Marker()
        marker.header.frame_id = "world"  # Set frame, change if necessary
        marker.header.stamp = rospy.Time.now()
        marker.ns = "submap_anchors"
        marker.id = 0
        marker.type = Marker.POINTS
        marker.action = Marker.ADD
        marker.scale.x = 0.5  # Size of points in meters
        marker.scale.y = 0.5

        # Add anchor poses as points to the marker
        for submap in self.submaps:
            anchor_pose = submap.anchor_point
            point = Point()
            point.x = anchor_pose[0]
            point.y = anchor_pose[1]
            point.z = anchor_pose[2]
            marker.points.append(point)

            # Set color based on submap.GPU
            if submap.gpu:
                color = ColorRGBA(0.0, 1.0, 0.0, 1.0)  # Green with full opacity
            else:
                color = ColorRGBA(1.0, 0.0, 0.0, 1.0)  # Red with full opacity
            marker.colors.append(color)

        # Publish the marker
        self.anchor_pose_pub.publish(marker)

    def create_submap_cloud(self, idx, downsample=10, map_type=1):
        # Publish the submap point cloud
        # First, get cam2world
        s_time = rospy.Time.now()
        cam2world = self.get_submap_to_global_tf_matrix(idx)
        # Then, get the submap point cloud
        submap = self.submaps[idx]
        # Check if submap is on GPU
        if submap.gpu:
            # create a copy of the submap on CPU
            original_gs_points = submap.params['means3D'].detach().cpu().numpy()
            gs_ground_labels = submap.params['ground_labels'].detach().cpu().numpy()
            gs_colors = submap.params['rgb_colors'].detach().cpu().numpy()
            original_gs_sizes = submap.params['log_scales'].detach().cpu().numpy()


        else:
            original_gs_points = submap.params['means3D'].detach().numpy()
            gs_ground_labels = submap.params['ground_labels'].detach().numpy()
            gs_colors = submap.params['rgb_colors'].detach().numpy()
            original_gs_sizes = submap.params['log_scales'].detach().numpy()

        if map_type == 1:
            original_gs_points = original_gs_points[gs_ground_labels == 0]
        elif map_type == 2:
            original_gs_points = original_gs_points[gs_ground_labels == 1]
        # print("shape of latest_means3D: ", self.latest_means3D.shape)
        gs_points = original_gs_points[0:original_gs_points.shape[0]:downsample]
        # convert to ros point cloud
        # print("shape of gs_points: ", gs_points.shape)
        gs_colors = gs_colors[0:gs_colors.shape[0]:downsample]
        # print("shape of gs_colors: ", gs_colors.shape)
        gs_sizes = original_gs_sizes[0:original_gs_sizes.shape[0]:downsample]
        # take exponential of the log scales
        gs_sizes = np.exp(gs_sizes)
        # print("shape of gs_sizes: ", gs_sizes.shape)
        # create the point cloud
        rgb_data = np.array([
                (int(r * 255) << 16) | (int(g * 255) << 8) | int(b * 255)
                for b, g, r in gs_colors], dtype=np.uint32)                
        point_data = [[None] *5] * len(gs_points)

        # Transform gs_points to global frame
        gs_points = np.dot(cam2world[:3, :3], gs_points.T).T + cam2world[:3, 3].reshape(1, 3)
        for i in range(len(gs_points)):
            point_data[i] = [gs_points[i][0], gs_points[i][1], gs_points[i][2], int(rgb_data[i]), gs_sizes[i]]
        # point_data = np.concatenate((gs_points, rgb_data[:, np.newaxis], gs_sizes), axis=1).tolist()
        # change rgb_data in point_data to int
        # for i in range(len(point_data)):
        #     point_data[i][3] = int(point_data[i][3])

        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "world"
        cloud = point_cloud2.create_cloud(header, self.pc_fields_, point_data)
        # self.map_pub.publish(cloud)
        e_time = rospy.Time.now()
        rospy.loginfo(f"Prepare submap cloud time: {(e_time-s_time).to_sec()} s")
        return cloud

    def _make_pc_fields(self):
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
    
    def print_cuda_usage(self):
        # Total CUDA memory used by the program in MB
        print("Total allocated CUDA memory:", torch.cuda.memory_allocated() / 1024 ** 2, "MB")
        print("Total reserved CUDA memory:", torch.cuda.memory_reserved() / 1024 ** 2, "MB")

    def print_all_submap_size(self):
        # sum of all submap sizes
        total_size = 0
        for idx in range(len(self.submaps)):
            total_size += self.compute_submap_size(idx)
            print("Total size: ", total_size)
        print("Total size of all submaps:", total_size / 1024 ** 2, "MB")

    def compute_submap_size(self, idx):
        submap = self.submaps[idx]
        # Check all variables if they are tensor, if yes, sum up
        total_size = 0
        if submap.params is not None:
            for key, val in submap.params.items():
                if torch.is_tensor(val):
                    total_size += val.element_size() * val.nelement()
        if submap.variables is not None:
            for key, val in submap.variables.items():
                if torch.is_tensor(val):
                    total_size += val.element_size() * val.nelement()
        if torch.is_tensor(submap.anchor_pose):
            total_size += submap.anchor_pose.element_size() * submap.anchor_pose.nelement()
        if torch.is_tensor(submap.anchor_point):
            total_size += submap.anchor_point.element_size() * submap.anchor_point.nelement()
        if torch.is_tensor(submap.first_frame_pose):
            total_size += submap.first_frame_pose.element_size() * submap.first_frame_pose.nelement()
        return total_size # in bytes

    def log_submap_usage(self):
        if not self.logging:
            return
        log_time = rospy.Time.now()
        log_time_elapsed = (log_time - self.log_start_time).to_sec()
        self.log_dict["Time elapsed (s)"] = log_time_elapsed
        # 1. log the total number of submaps
        self.log_dict["Total # submaps"] = len(self.submaps)
        # 2. log the total number of submaps on GPU
        self.log_dict["Total submaps on GPU"] = sum([submap.gpu for submap in self.submaps])
        # 3. log the total CUDA allocated memory
        self.log_dict["CUDA allocated mem (MB)"] = torch.cuda.memory_allocated() / 1024 ** 2
        # 4. log the total CUDA reserved memory
        self.log_dict["CUDA reserved mem (MB)"] = torch.cuda.memory_reserved() / 1024 ** 2
        # 5. log the total submap usage
        total_submap_usage = 0
        for idx in range(len(self.submaps)):
            total_submap_usage += self.compute_submap_size(idx)
        self.log_dict["Total submap usage (MB)"] = total_submap_usage / 1024 ** 2
        # write to file
        self.debug_stats_file = open(self.log_fname, "a")
        self.debug_stats_file.write(";\t".join(f"{key}: {value}" for key, value in self.log_dict.items()) + ";\n")
        self.debug_stats_file.close()


    def add_submap_node(self, submap_pose: np.ndarray, submap_id: int, utility: float):
        # pose = 7 vec
        pos = submap_pose[:3]
        yaw = R.from_quat(submap_pose[3:]).as_euler("xyz")[2]
        # rospy.loginfo(f"Odom pos: {self.odom_pos}, Odom yaw: {self.odom_yaw}")
        
        # Create a unique ID for this position
        current_node_id = submap_id
        node_id = 's'+str(current_node_id)
        if self.previous_position is not None:
            # distance = math.sqrt((pos[0] - self.previous_position[0])**2 +
            #                      (pos[1] - self.previous_position[1])**2)
            # if distance >= self.distance_threshold or distance == 0.:
            # Add edge between previous node and current node if threshold is met
            distances = [(n, np.linalg.norm(pos[:2] - np.array(d['pos']))) for (n,d) in self.graph.nodes(data=True) if d['predicted'] == False]
            _min = min(distances, key=lambda x: x[1])
            # if _min[1] > self.odom_threshold:            
            self.graph.add_node(node_id, pos=(pos[0], pos[1]), predicted=False, utility=0., frontier=False, relevancy=0.0)
            self.graph.add_edge(_min[0], node_id)
            self.previous_position = (pos[0], pos[1])
            self.odom_id = current_node_id
            #print("Adding an odom node")
        else:
            self.graph.add_node(node_id, pos=(pos[0], pos[1]), predicted=False, utility=0., frontier=False, relevancy=0.0)
            self.previous_position = (pos[0], pos[1])

    def gen_planning_graph(self):
        # clean the graph
        self.graph.clear()
        self.previous_position = None
        for i in range(len(self.submaps)):
            submap = self.submaps[i]
            anchor_pose = submap.anchor_pose
            self.add_submap_node(anchor_pose, int(i), utility=0.)
        # publish a nx json
        # data = json_graph.adjacency_data(self.graph)
        # json_data = json.dumps(data)
        # self.nx_graph_pub.publish(json_data)


    def cluster_submap_mp(self):
        for idx, submap in enumerate(self.submaps):
            if submap.updated:
                means3d = submap.params['means3D'].detach().clone()
                features = submap.params['features'].detach().clone()
                clustered_means, clustered_features = cluster_submap(means3d, features)
                objs = {}
                objs['means'] = clustered_means
                objs['features'] = clustered_features
                self.submap_objs[idx] = objs
                self.submaps[idx].updated = False

    def update_anchor_pose(self, pose_path):
        # the pose_path is the path messag with pose and corresponding timestamp
        # Iterate through submaps, find their timestamp and correponding pose in pose_path 
        # and update the anchor pose
        pose_idx = 0
        self.match_threshold = 0.05 # in seconds
        for submap in self.submaps:
            submap_time = submap.time_stamp
            pose = pose_path[pose_idx]
            pose_time = [pose.header.stamp.secs, pose.header.stamp.nsecs]
            if submap_time[0] > pose_time[0] or (submap_time[0] == pose_time[0] and submap_time[1] > pose_time[1]):
                # Find the next pose
                pose_idx += 1
            elif (submap_time[0] == pose_time[0] and submap_time[1] == pose_time[1]) or \
                 ((pose_time[0]-submap_time[0]) * 1e9 + (pose_time[1]-submap_time[1]) < self.match_threshold*1e9):
                # Find match
                new_odom_pos = np.array([pose.position.x, pose.position.y, pose.position.z])
                new_odom_rpy = Rotation.from_quat([pose.orientation.x, 
                                                    pose.orientation.y, 
                                                    pose.orientation.z, 
                                                    pose.orientation.w]).as_euler('xyz')
                # apply cam2body to odom_pos. cam_pos = body2world * cam2body
                body2world = np.eye(4)
                body2world[:3, :3] = Rotation.from_euler('xyz', new_odom_rpy).as_matrix()
                body2world[:3, 3] = new_odom_pos
                cam2world = np.dot(body2world, self.cam2body)
                cam_pos = cam2world[:3, 3]
                cam_quat = Rotation.from_matrix(cam2world[:3, :3]).as_quat()
                mew_pose_vec = np.concatenate([cam_pos, cam_quat])
                submap.anchor_pose = mew_pose_vec
                submap.anchor_h = cam2world
                submap.anchor_h_inv = np.linalg.inv(cam2world)
                continue
            else:
                # doesn't find the corresponding pose
                print("Cannot find the corresponding pose for submap id: ", submap.id)
                continue
