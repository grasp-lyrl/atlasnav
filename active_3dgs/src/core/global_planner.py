import numpy as np
import torch
import matplotlib.pyplot as plt
import json

import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

import networkx as nx
import math
from networkx.readwrite import json_graph

class GlobalPlanner:
    def __init__(self, grid_length=100, resolution=5, region_rel_th=0.5, obj_rel_th=0.7):
        self.grid_length = grid_length
        self.resolution = resolution
        self.grid_size = int(grid_length / resolution)
        self.occupancy_grid = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        self.visited_grid = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        self.grid_origin = np.array([self.grid_size // 2, self.grid_size // 2])
        
        self.region_relevancy_threshold = region_rel_th
        self.object_relevancy_threshold = obj_rel_th
                
        self.global_plan_pub = rospy.Publisher('global_planner', String, queue_size=10)
        self.global_plan_viz_pub = rospy.Publisher('global_planner_viz', Image, queue_size=10)
        self.cv_bridge = CvBridge()
        self.nms_radius = 3.0
        
    def world_to_grid(self, x, y):
        x = x.reshape(-1, 1)
        y = y.reshape(-1, 1)
        grid_coords = np.hstack((x, y)) / self.resolution + self.grid_origin.reshape(1,2)
        
        return np.clip(grid_coords.astype(int), 0, self.grid_size - 1)

    def grid_to_world(self, grid_x, grid_y):
        world_coords = (np.array([grid_x, grid_y]) - self.grid_origin) * self.resolution
        
        return world_coords

    def update_map(self, submaps):
        submap_pos = np.array([submap.anchor_point[:2] for submap in submaps])
        if submap_pos.size == 0:
            return
        submap_pos = submap_pos.squeeze().reshape(-1, 2)
        submap_grid_coords = self.world_to_grid(submap_pos[:, 0], submap_pos[:, 1])
        
        self.occupancy_grid[submap_grid_coords[:, 0], submap_grid_coords[:, 1]] = True

    def get_nearest_unvisited(self, curr_pose):
        unvisited = ~np.logical_or(self.occupancy_grid, self.visited_grid)
        unvisited_idxs = np.argwhere(unvisited)

        if unvisited_idxs.size > 0:            
            current_grid_pos = self.world_to_grid(curr_pose[0], curr_pose[1])
            dists = np.linalg.norm(unvisited_idxs - current_grid_pos, axis=1)
            nearest_idx = unvisited_idxs[np.argmin(dists)]
            
            self.nearest_idx = nearest_idx
            
            return self.grid_to_world(nearest_idx[0], nearest_idx[1])

        return None
    
    def get_submap_regions(self, submaps):
        submap_regions = {}
        for idx, submap in enumerate(submaps):
            if len(submap.regions) > 0:
                submap_regions[idx] = submap.regions
            
        return submap_regions
    
    def get_local_regions(self, means3d, relevancy, voxel_size=0.1):
        # voxel indices
        voxel_indices = torch.floor(means3d / voxel_size).long()
        unique_voxels, inverse_indices = torch.unique(voxel_indices, dim=0, return_inverse=True)
        
        num_voxels = unique_voxels.size(0)
        voxel_rel_sums = torch.zeros(num_voxels, device=relevancy.device)
        voxel_counts = torch.zeros(num_voxels, device=relevancy.device)
        
        # aggregate relevancy per voxel
        voxel_rel_sums.index_add_(0, inverse_indices, relevancy)
        voxel_counts.index_add_(0, inverse_indices, torch.ones_like(relevancy))
        
        # avoid division by zero
        voxel_counts = torch.clamp(voxel_counts, min=1)
        
        # compute average relevancy per voxel
        voxel_relevancy = voxel_rel_sums / voxel_counts
        
        # compute voxel centers
        voxel_centers = unique_voxels.float() * voxel_size + voxel_size / 2
        
        valid_idxs = voxel_relevancy > self.object_relevancy_threshold
        valid_voxels = voxel_centers[valid_idxs]
        valid_rel = voxel_relevancy[valid_idxs]
        
        # sort by relevancy
        sort_idxs = torch.argsort(valid_rel, descending=True)
        valid_voxels = valid_voxels[sort_idxs]
        valid_rel = valid_rel[sort_idxs]
        
        # NMS
        keep = torch.ones(valid_voxels.size(0), dtype=torch.bool, device=valid_voxels.device)
        
        for i in range(valid_voxels.size(0)):
            if keep[i]:
                # compute distances to all other boxes
                distances = torch.norm(valid_voxels[i+1:] - valid_voxels[i].unsqueeze(0), dim=1)
                # suppress boxes with distance less than nms_radius
                keep[i+1:][distances < self.nms_radius] = False
        
        # apply suppression
        valid_voxels = valid_voxels[keep]
        valid_rel = valid_rel[keep]
        
        # return as (pos,rel) tuples
        valid_regions = [(pos.cpu().numpy(), rel.item()) for pos, rel in zip(valid_voxels, valid_rel)]
        
        return valid_regions
    
    def create_msg(self, submap_regions, local_regions, new_region):
        # create json msg
        msg = {}
        # submap regions
        msg['submap_regions'] = {}
        for id, regions in submap_regions:
            msg['submap_regions']['submap_id'] = int(id)
            msg['submap_regions']['submap_id']['regions'] = []
            for pos, rel in regions:
                region = {}
                region['pos'] = pos.tolist()
                region['rel'] = rel.tolist()
                msg['submap_regions']['submap_id']['regions'].append(region)
        # local regions
        msg['local_regions'] = []
        for pos, rel in local_regions:
            region = {}
            region['pos'] = pos.tolist()
            region['rel'] = rel.tolist()
            msg['local_regions'].append(region)
        # new region
        msg['new_region'] = new_region.tolist()
                
        str_msg = json.dumps(msg)
        
        return str_msg
    
    def add_regions_to_graph(self, submap_regions, local_regions, new_region, submap_manager, first_cam2world):
        graph_size = len(submap_manager.submaps)
        # submap regions
        for id, regions in submap_regions.items():
            for pos, rel in regions:
                # check if rel is an array
                if isinstance(rel, np.ndarray):
                    rel = float(rel)
                pos_h = np.hstack((pos, 1))
                pos_w = np.dot(first_cam2world, pos_h)
                pos = pos_w[:2]
                submap_manager.graph.add_node('r'+str(graph_size), pos=(str(pos[0]), str(pos[1])), predicted=False, utility=0., frontier=False, submap_id='s'+str(id), relevancy=str(rel))
                graph_size += 1
        # local regions
        for pos, rel in local_regions:
            # check if rel is an array
            if isinstance(rel, np.ndarray):
                rel = float(rel)
            submap_manager.graph.add_node('l'+str(graph_size), pos=(str(pos[0]), str(pos[1])), predicted=False, utility=0., frontier=False, local_region=True, relevancy=str(rel))
            graph_size += 1
        # new region
        #submap_manager.graph.add_node('p'+str(graph_size), pos=(str(new_region[0]), str(new_region[1])), predicted=True, utility=0., frontier=False)
        
         # publish a nx json
        data = json_graph.adjacency_data(submap_manager.graph)
        # log to txt
        # with open('graph.json', 'w') as f:
        #     json.dump(data, f)
        # json_data = json.dumps(data)
        return data
        # submap_manager.nx_graph_pub.publish(json_data)
        
    def publish_msg(self, msg):
        # publish msg
        self.global_plan_pub.publish(String(data=msg))
    
    def global_planner_update(self, local_means3d, local_relevancy, submap_manager, curr_pose, first_cam2world):
        # get submap regions
        submap_regions = self.get_submap_regions(submap_manager.submaps)
        
        # get local regions
        local_regions = self.get_local_regions(local_means3d, local_relevancy)
        
        # get new region
        new_region = self.get_nearest_unvisited(curr_pose)
        
        # add regions to graph
        json_data = self.add_regions_to_graph(submap_regions, local_regions, new_region, submap_manager, first_cam2world)
        
        # create str msg
        # msg = self.create_msg(submap_regions, local_regions, new_region)
        
        # publish msg
        # self.publish_msg(msg)
        
        # publish viz
        #self.publish_global_map()
        return json_data

    def viz_global_map(self):
        # plot both
        plt.figure(figsize=(10, 10))
        plt.imshow(self.occupancy_grid, cmap='gray')
        plt.imshow(self.visited_grid, cmap='jet', alpha=0.5)
        plt.show()
        
    def publish_global_map(self):
        fig, ax = plt.subplots()
        ax.imshow(self.occupancy_grid, cmap='gray')
        ax.imshow(self.visited_grid, cmap='jet', alpha=0.5)
        # mark nearest_idx green
        ax.scatter(self.nearest_idx[1], self.nearest_idx[0], c='g', s=100)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.canvas.draw()
        
        # convert to cv2 image
        img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # publish image
        img_msg = self.cv_bridge.cv2_to_imgmsg(img, encoding='bgr8')
        self.global_plan_viz_pub.publish(img_msg)
        
