import numpy as np
import torch

from utils.slam_helpers import (
    transform_to_frame, transform_to_frame_batch
)

def in_image(x, y, width, height):
    return (x >= 0) & (x < width) & (y >= 0) & (y < height)

def get_utility_gradients(params, curr_data, iter_time_idx, img_w, img_h):
    # Get current frame Gaussians
    transformed_gaussians = transform_to_frame(params, iter_time_idx, 
                                             gaussians_grad=False,
                                             camera_grad=False)
    
    means3d = transformed_gaussians['means3D']
    intrinsics = curr_data['intrinsics']
    
    # project means on to the image plane
    fx, fy, cx, cy = intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]
    x = means3d[:, 0] / means3d[:, 2] * fx + cx
    y = means3d[:, 1] / means3d[:, 2] * fy + cy
    
    # check which points are valid
    valid = in_image(x, y, img_w, img_h)
    num_valid = torch.sum(valid).cpu().numpy()
    
    if params.get('update_means3D') is not None:
        update_idx = params['update_means3D']
        noupdate_idx = params['noupdate_means3D']
        x_update = x[update_idx]
        y_update = y[update_idx]
        valid_update = in_image(x_update, y_update, img_w, img_h)
        num_valid_update = torch.sum(valid_update).cpu().numpy() # maximize this
        grad_means3D = torch.sum(params['grad_means3D'])
        
        x_noupdate = x[noupdate_idx]
        y_noupdate = y[noupdate_idx]
        valid_noupdate = in_image(x_noupdate, y_noupdate, img_w, img_h)
        num_valid_noupdate = torch.sum(valid_noupdate).cpu().numpy() # minimize this
        
        num_valid = num_valid_noupdate - num_valid_update
    
    return num_valid

def get_utility_gradients_batch(means3d, params, intrinsics, img_w, img_h):
        
    # project means on to the image plane
    fx, fy, cx, cy = intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]
    
    x = means3d[:, :, 0] / means3d[:, :, 2] * fx + cx
    y = means3d[:, :, 1] / means3d[:, :, 2] * fy + cy
    
    # check which points are valid
    valid = in_image(x, y, img_w, img_h)
    num_valid = torch.sum(valid, dim=1)
    
    if params.get('update_means3D') is not None:
        # project means on to the image plane
        fx, fy, cx, cy = intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]
        x = means3d[:, :, 0] / means3d[:, :, 2] * fx + cx
        y = means3d[:, :, 1] / means3d[:, :, 2] * fy + cy
        
        update_idx = params['update_means3D']
        noupdate_idx = params['noupdate_means3D']
        
        x_update = x[:, update_idx]
        y_update = y[:, update_idx]
        valid_update = in_image(x_update, y_update, img_w, img_h)
        num_valid_update = torch.sum(valid_update, dim=1) # maximize this
        grad_means3D = torch.sum(params['grad_means3D'])
        
        x_noupdate = x[:, noupdate_idx]
        y_noupdate = y[:, noupdate_idx]
        valid_noupdate = in_image(x_noupdate, y_noupdate, img_w, img_h)
        num_valid_noupdate = torch.sum(valid_noupdate, dim=1) # minimize this

        num_valid = num_valid_noupdate - num_valid_update
        
    return num_valid

def get_utility_relevancy_batch(means3d, params, intrinsics, img_w, img_h):
        
    # project means on to the image plane
    fx, fy, cx, cy = intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]
    
    x = means3d[:, :, 0] / means3d[:, :, 2] * fx + cx
    y = means3d[:, :, 1] / means3d[:, :, 2] * fy + cy
    
    if params.get('relevancy') is not None:
        # project means on to the image plane
        fx, fy, cx, cy = intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]
        x = means3d[:, :, 0] / means3d[:, :, 2] * fx + cx
        y = means3d[:, :, 1] / means3d[:, :, 2] * fy + cy
        
        valid = in_image(x, y, img_w, img_h)
        
        # sum up relevancy
        utility = torch.sum(params['relevancy'] * valid, dim=1)
        
    return utility

def evaluate_trajectories(trajs, params, intrinsics, img_w, img_h):
    # trajs should be a list of k (N, 7) tensors
    traj_sizes = [traj.shape[0] for traj in trajs]

    # Flatten trajs into a (k*N, 7) tensor
    rots = torch.cat([traj[:, :4] for traj in trajs], dim=0)
    trans = torch.cat([traj[:, 4:] for traj in trajs], dim=0)

    # Transform to frames batch expects (params, cam_unnorm_rots, cam_trans)
    # params dict
    # cam_unorm_rots quaternion (k*N, 4)
    # cam_tran xyz (k*N, 3)
    # returns transformed gaussians (k*N, M, 3) for M gaussians
    transformed_gaussians = transform_to_frame_batch(params, rots, trans, False, False)
    
    # compute utility for all nodes
    utility = get_utility_gradients_batch(transformed_gaussians, params, intrinsics, img_w, img_h)  # (k*N, 1)
    
    # split the utility back into the sizes of the original trajectories (k*N -> k, N)
    utility_list = torch.split(utility, traj_sizes)
    
    # for i in range(len(utility_list)):
    #     print(f"Trajectory {i} utility per node: {utility_list[i]}")
        
    # sum utility for each trajectory
    utility_list = [torch.sum(utility) for utility in utility_list]
    
    # for i in range(len(utility_list)):
    #     print(f"Trajectory {i} utility: {utility_list[i]}")
    
    # maybe argmax here or just return the utility list
    # print(utility_list)
    
    return utility_list

def evaluate_trajectories_relevancy(trajs, params, intrinsics, img_w, img_h):
    # trajs should be a list of k (N, 7) tensors
    traj_sizes = [traj.shape[0] for traj in trajs]

    # Flatten trajs into a (k*N, 7) tensor
    rots = torch.cat([traj[:, :4] for traj in trajs], dim=0)
    trans = torch.cat([traj[:, 4:] for traj in trajs], dim=0)

    # Transform to frames batch expects (params, cam_unnorm_rots, cam_trans)
    # params dict
    # cam_unorm_rots quaternion (k*N, 4)
    # cam_tran xyz (k*N, 3)
    # returns transformed gaussians (k*N, M, 3) for M gaussians
    transformed_gaussians = transform_to_frame_batch(params, rots, trans, False, False)
    
    # compute utility for all nodes
    utility = get_utility_relevancy_batch(transformed_gaussians, params, intrinsics, img_w, img_h)  # (k*N, 1)
    
    # split the utility back into the sizes of the original trajectories (k*N -> k, N)
    utility_list = torch.split(utility, traj_sizes)
    
    # for i in range(len(utility_list)):
    #     print(f"Trajectory {i} utility per node: {utility_list[i]}")
        
    # sum utility for each trajectory
    utility_list = [torch.sum(utility) for utility in utility_list]
    
    # for i in range(len(utility_list)):
    #     print(f"Trajectory {i} utility: {utility_list[i]}")
    
    # maybe argmax here or just return the utility list
    # print(utility_list)
    
    return utility_list

def get_gaussian_frontiers(params, cluster_size=0.5, nms_radius=1.5, filter_grad_mean=False):
    means3D = params['means3D'].detach()
    grad_means3D = params['grad_means3D'].detach()
    grad_means3D = grad_means3D.reshape(-1, 1)

    # print("Number of Gaussians:", len(means3D))

    # # Ignore outliers +-10
    # mask = (means3D.abs() < 10).all(dim=1)
    # means3D = means3D[mask]
    # grad_means3D = grad_means3D[mask]

    # cluster size
    # cluster_size = 0.5
    inverse_cluster_size = 1.0 / cluster_size

    # compute cluster indices
    cluster_indices = torch.floor(means3D * inverse_cluster_size).int()
    unique_cluster_indices, inverse_indices = torch.unique(cluster_indices, dim=0, return_inverse=True)

    sum_means = torch.zeros((len(unique_cluster_indices), 3)).cuda()
    sum_grads = torch.zeros((len(unique_cluster_indices), 1)).cuda()
    counts = torch.zeros(len(unique_cluster_indices), dtype=int).cuda()

    # compute means for each cluster
    sum_means = torch.index_add(sum_means, 0, inverse_indices, means3D)
    sum_grads = torch.index_add(sum_grads, 0, inverse_indices, grad_means3D)
    counts = torch.index_add(counts, 0, inverse_indices, torch.ones(len(inverse_indices)).long().cuda())
    mean_positions = torch.div(sum_means, counts.unsqueeze(1))
    mean_grads = torch.div(sum_grads, counts.unsqueeze(1))

    # only keep centroids above grad mean
    if filter_grad_mean:
        avg_grad = torch.mean(mean_grads)
        std_grad = torch.std(mean_grads)
        ths_grad = avg_grad# + std_grad
        grads_mask = mean_grads > ths_grad
        grads_mask = grads_mask.squeeze()
        mean_positions = mean_positions[grads_mask]
        mean_grads = mean_grads[grads_mask]

    cluster_centroids = mean_positions.cpu().numpy().squeeze()
    average_gradient_magnitudes = mean_grads.cpu().numpy().squeeze()

    # non-maximum suppression
    # nms_radius = 1.5
    # sort cluster_centroids by average_gradient_magnitudes
    #cluster_centroids = cluster_centroids[np.argsort(average_gradient_magnitudes)]
    #average_gradient_magnitudes = average_gradient_magnitudes[np.argsort(average_gradient_magnitudes)]
    #nms_clusters = []
    #nms_indices = []
    #valid_indices = []
    #for i, centroid in enumerate(cluster_centroids):
    #    if i in nms_indices:
    #        continue
    #    else:
    #        valid_indices.append(i)
    #    nms_clusters.append(centroid)
    #    nms_indices.append(i)
    #    for j, other_centroid in enumerate(cluster_centroids):
    #        if j in nms_indices:
    #            continue
    #        if np.linalg.norm(centroid - other_centroid) < nms_radius:
    #            nms_indices.append(j)

    #cluster_centroids = cluster_centroids[valid_indices]
    #average_gradient_magnitudes = average_gradient_magnitudes[valid_indices]
    
    sorted_indices = np.argsort(average_gradient_magnitudes)
    cluster_centroids = cluster_centroids[sorted_indices]
    average_gradient_magnitudes = average_gradient_magnitudes[sorted_indices]

    distances = np.linalg.norm(cluster_centroids[:, np.newaxis] - cluster_centroids[np.newaxis, :], axis=2)
    keep_indices = np.ones(len(cluster_centroids), dtype=bool)

    for i in range(len(cluster_centroids)):
        if not keep_indices[i]:
            continue

        # check if within radius
        within_radius = distances[i] < nms_radius
        keep_indices[within_radius] = False
        keep_indices[i] = True

    # filter
    nms_clusters = cluster_centroids[keep_indices]
    nms_indices = sorted_indices[keep_indices]
    nms_gradient_magnitudes = average_gradient_magnitudes[keep_indices]

    # sort by gradient magnitude
    num_clusters = len(nms_clusters)
    top_indices = np.argsort(nms_gradient_magnitudes)[-num_clusters:]
    cluster_centroids = nms_clusters[top_indices]
    average_gradient_magnitudes = nms_gradient_magnitudes[top_indices]
    
    return cluster_centroids, average_gradient_magnitudes

if __name__ == '__main__':
    
    # 2 trajectories, length 10 and length 5 
    trajs = [torch.randn(10, 7), torch.randn(5, 7)]
    # 100 gaussians
    params = {'means3D': torch.randn(10, 3).cuda(), 'log_scales': torch.randn(10, 1).cuda(), 'unnorm_rotations': torch.randn(10, 4).cuda()}
    intrinsics = torch.randn(3, 3)
    evaluate_trajectories(trajs, params, intrinsics)
