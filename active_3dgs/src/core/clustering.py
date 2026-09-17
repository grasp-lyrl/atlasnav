import torch
import torch.nn.functional as F
from sklearn.cluster import AgglomerativeClustering
    
def cluster_submap(means3d, feats, dsample_num=5000, weighted_avg=False, alpha=0.6):        
    # downsample
    dsample_num = min(dsample_num, len(feats))
    dsample_factor = len(feats) // dsample_num
    feats = feats[::dsample_factor]
    means3d = means3d[::dsample_factor]

    # compute cosine similarity between each point with all other points
    feats_norm = F.normalize(feats, dim=-1)
    S = torch.matmul(feats_norm, feats_norm.t())
    S = torch.sigmoid(S*100)
    S = S.detach().cpu().numpy()
    
    E = torch.norm(means3d[:, None] - means3d, dim=-1)
    E = E.detach().cpu().numpy()
    
    # weight
    alpha = 0.6
    n_clusters = min(100, len(feats))
    
    W = alpha * E + (1 - alpha) * (1 - S)
    
    # cluster
    clustering = AgglomerativeClustering(n_clusters=n_clusters,
                                        metric='precomputed',
                                        linkage='average',
                                        compute_distances=True)
    clusters = clustering.fit(W).labels_
    
    cluster_feats_list = []
    cluster_means_list = []
    for i in range(n_clusters):
        cluster_feats = feats[clusters == i]
        cluster_pts = means3d[clusters == i]
        
        cluster_means_mean = cluster_pts.mean(dim=0)
        if weighted_avg:
            # weighted average
            dists = torch.norm(cluster_pts - cluster_means_mean, dim=-1)
            dists = 1 - dists / dists.max()
            cluster_feats_mean = cluster_feats * dists[:, None]
            cluster_feats_mean = cluster_feats_mean.mean(dim=0)
        else:
            cluster_feats_mean = cluster_feats.mean(dim=0)
        
        cluster_means_list.append(cluster_means_mean)
        cluster_feats_list.append(cluster_feats_mean)
    
    cluster_means_list = torch.stack(cluster_means_list)            
    cluster_feats_list = torch.stack(cluster_feats_list)

    return cluster_means_list, cluster_feats_list

def cluster_submap_worker(idx, means3d, features):
    clustered_means, clustered_features = cluster_submap(means3d, features)
    return idx, clustered_means, clustered_features