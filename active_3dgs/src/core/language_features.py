import numpy as np
import matplotlib.pyplot as plt
import torch

import glob
from PIL import Image
import torchvision.transforms as T
import torch.nn.functional as F
from sklearn.cluster import AgglomerativeClustering

class LanguageFeaturesManager:
    def __init__(self, feature_mode='pca', clip_dim=512, pca_dim=12, 
                 data_path=None,
                 region_rel_th=0.5, use_vlm=False):
        self.feature_mode = feature_mode
        self.clip_dim = clip_dim
        self.pca_dim = pca_dim
        
        # Validate the supplied basis before downloading the CLIP backbone.
        pca = torch.load(data_path, map_location='cpu', weights_only=False)
        if np.shape(pca.get('mean')) != (clip_dim,) or np.shape(pca.get('V')) != (pca_dim, clip_dim):
            raise ValueError(f"PCA checkpoint must contain mean ({clip_dim},) and V ({pca_dim}, {clip_dim})")
        if not np.isfinite(pca['mean']).all() or not np.isfinite(pca['V']).all():
            raise ValueError("PCA checkpoint contains nonfinite values")
        try:
            from clip_dinoiser.clipdino_gpu import ClipDino
        except ImportError as exc:
            raise ImportError("Initialize the clip_dinoiser submodule and install requirements-language.txt") from exc
        self.clipdino = ClipDino(feature_mode=feature_mode, clip_dim=clip_dim, pca_dim=pca_dim, data_path=data_path)
        
        self.region_relevancy_threshold = region_rel_th
        self.n_clusters = 100
        
        self.use_vlm = use_vlm
        self.processor = None
        self.vlm = None
        self.task = 'door'

    def _load_vlm(self):
        from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration, BitsAndBytesConfig
        model_id = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
        self.processor = AutoProcessor.from_pretrained(model_id)
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        self.vlm = LlavaOnevisionForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True,
            quantization_config=quantization_config, device_map={"": torch.cuda.current_device()})
        
    def fit_pca(self, dir=None):
        if dir is not None:
            file_paths = sorted(glob.glob(dir + '/*.png'))
        else:
            raise ValueError('Provide an image directory to fit_pca')

        clip_feats = []
        for file_path in file_paths:
            # skip every other frame
            if file_paths.index(file_path) % 5 != 0:
                continue
            img = Image.open(file_path).convert('RGB')
            img = T.PILToTensor()(img).unsqueeze(0).to("cuda") / 255.
            clip_feat = self.clipdino.get_clipdino_features(img)
            clip_feats.append(clip_feat)
            
        clip_feats = torch.cat(clip_feats, dim=0)
        self.clipdino.fit_pca(clip_feats.detach())
        
    def set_prompt(self, prompt):
        if ',' in prompt:
            prompts = prompt.split(',')
            self.task = prompts[0]
            prompt_background = prompts[1]
        else:
            self.task = prompt
            prompt_background = 'background'
        self.clipdino.set_prompt(prompt)
    
    def get_features(self, color):
        print(color.shape)
        h,w = color.shape[1], color.shape[2]        
        feat_pca_img, feat_img  = self.clipdino.process_features(color.detach().squeeze())
        print(feat_pca_img.shape, feat_img.shape)
        feat_pca_img = F.interpolate(feat_pca_img, size=(h, w), mode='bilinear', align_corners=False).squeeze()
        feat_img = F.interpolate(feat_img, size=(h, w), mode='bilinear', align_corners=False).squeeze()
        
        return feat_pca_img.detach(), feat_img.detach()
    
    def get_relevancy(self, feat_img):
        h, w = feat_img.shape[-2:]
        relevancy_img = self.clipdino.get_relevancy_from_pca(feat_img.unsqueeze(0))[0][1]
        relevancy_img = F.interpolate(relevancy_img.unsqueeze(0).unsqueeze(0), size=(h, w), mode='bilinear', align_corners=False).squeeze()
        relevancy_img = torch.repeat_interleave(relevancy_img.unsqueeze(0), 3, dim=0)
        
        return relevancy_img.detach()
    
    def get_feat_relevancy(self, color):
        feat_img, _ = self.get_features(color)
        return feat_img, self.get_relevancy(feat_img)
    
    def compute_relevancy_ptcloud(self, feats):
        recovered_feat = torch.matmul(feats, self.clipdino.V)
        recovered_feat += self.clipdino.mean
        
        recovered_feat = recovered_feat / recovered_feat.norm(dim=-1, keepdim=True)
        text_embeds = self.clipdino.model.clip_backbone.decode_head.class_embeddings
        relevancy = torch.matmul(recovered_feat, text_embeds.t())
        relevancy = F.softmax(relevancy * 100, dim=-1)
        relevancy = relevancy[:, 1].detach()
        
        return relevancy
    
    def build_tree_structure(self, children, n_samples, n_clusters):
        tree = {}
        clusters = {i: [i] for i in range(n_samples)}
        saved_clusters = None

        for i, (c1, c2) in enumerate(children):
            new_cluster_id = n_samples + i
            clusters[new_cluster_id] = clusters.pop(c1) + clusters.pop(c2)

            if len(clusters) == n_clusters:
                saved_clusters = clusters.copy()
                # Mark the current clusters as leaves and stop adding to the tree
                for cluster_id in clusters.keys():
                    tree[cluster_id] = {'is_leaf': True, 'points': clusters[cluster_id]}
                # break
            else:
                tree[new_cluster_id] = {'left': c1, 'right': c2, 'is_leaf': False}
                
        return tree, saved_clusters
    
    def compute_node_positions(self, tree, means3d, relevancy):
        positions = {}
        depths = {}
        relevancies = {}
        
        def compute_node_positions_recursive(node_id, depth=0):
            node = tree[node_id]
            if node['is_leaf']:
                pos = means3d[node['points']].mean(axis=0)
                positions[node_id] = pos
                rel = relevancy[node['points']].mean()
                relevancies[node_id] = rel
            else:
                left_pos, left_rel = compute_node_positions_recursive(node['left'], depth + 1)
                right_pos, right_rel = compute_node_positions_recursive(node['right'], depth + 1)
                pos = (left_pos + right_pos) / 2
                positions[node_id] = pos
                # threshold rel
                if left_rel < self.region_relevancy_threshold:
                    left_rel = 0
                if right_rel < self.region_relevancy_threshold:
                    right_rel = 0
                # sum instead of max
                rel = sum([left_rel, right_rel])
                relevancies[node_id] = rel
                
            depths[node_id] = depth
                
            return positions[node_id], relevancies[node_id]
        
        root_id = max(tree.keys())
        compute_node_positions_recursive(root_id)
        
        return positions, depths, relevancies

    def cluster_submap(self, means3d, feats, dsample_num=5000, weighted_avg=False, alpha=0.6):
        # downsample
        dsample_num = min(dsample_num, len(feats))
        dsample_factor = len(feats) // dsample_num
        feats = feats[::dsample_factor].cuda()
        means3d = means3d[::dsample_factor].cuda()
        
        # compute cosine similarity between each point with all other points
        feats_norm = F.normalize(feats, dim=-1)
        S = torch.matmul(feats_norm, feats_norm.t())
        S = torch.sigmoid(S*100)
        S = S.detach().cpu().numpy()
        
        # compute Euclidean distance between each point with all other points
        E = torch.norm(means3d[:, None] - means3d, dim=-1)    
        E = E.detach().cpu().numpy()    
        
        # weighted clustering
        alpha = 0.6
        n_clusters = self.n_clusters
        
        W = alpha * E + (1 - alpha) * (1 - S)
        
        # Perform agglomerative clustering
        clustering = AgglomerativeClustering(n_clusters=n_clusters,
                                            metric='precomputed',
                                            linkage='average',
                                            compute_distances=True)
        clusters = clustering.fit(W).labels_
        
        # build tree
        tree, saved_clusters = self.build_tree_structure(clustering.children_, len(means3d), n_clusters)
        
        # create labels_ from the final clusters
        labels = np.empty(len(means3d), dtype=int)
        for cluster_id, points in enumerate(saved_clusters.values()):
            for point in points:
                labels[point] = cluster_id
                
        clusters = labels
        
        # compute relevancy
        recovered_feat = torch.matmul(feats.cuda(), self.clipdino.V)
        recovered_feat += self.clipdino.mean
        recovered_feat = recovered_feat / recovered_feat.norm(dim=-1, keepdim=True)
        text_embeds = self.clipdino.model.clip_backbone.decode_head.class_embeddings
        relevancy = torch.matmul(recovered_feat, text_embeds.t())
        relevancy = F.softmax(relevancy * 100, dim=-1)
        relevancy = relevancy[:, 1].detach().cpu().numpy()
        
        positions, depths, relevancies = self.compute_node_positions(tree, means3d, relevancy)
        # get points at depth 2
        depth2_points = [positions[node_id] for node_id, depth in depths.items() if depth == 2]
        depth2_relevancies = [relevancies[node_id] for node_id, depth in depths.items() if depth == 2]
        # get valid waypoints above relevancy threshold
        # get waypoints and their relevancies
        valid_waypoints = [(pos, rel) for pos, rel in zip(depth2_points, depth2_relevancies) if rel > 0] # 0 since sum is thresholded already
        valid_waypoints = [(pos.cpu().numpy(), rel) for pos, rel in valid_waypoints]
        
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

        return cluster_means_list, cluster_feats_list, valid_waypoints, tree, means3d, clusters
        
    def cluster_submaps(self, submap_manager, first_cam2world):
        for idx, submap in enumerate(submap_manager.submaps):
            if submap.updated:
                means3d = submap.params['means3D'].detach().clone().cuda()
                feats = submap.params['features'].detach().clone().cuda()
                if submap_manager.use_relative_pose:
                    means_homo = torch.cat((means3d, torch.ones(means3d.shape[0], 1).cuda()), dim=1)
                    means_world = torch.matmul(torch.from_numpy(submap.anchor_h).float().cuda(), means_homo.T)
                    means3d = means_world[:3, :].T
                cluster_means, cluster_feats, waypoints, tree, cluster_pts, cluster_labels = self.cluster_submap(means3d, feats)
                submap.cluster_means = cluster_means
                submap.cluster_feats = cluster_feats
                submap.cluster_pts = cluster_pts
                submap.labels = cluster_labels
                submap.regions = waypoints
                submap.tree = tree
                submap.updated = False
    
    def recompute_submap_relevancy(self, submap_manager, first_cam2world):
        for idx, submap in enumerate(submap_manager.submaps):
            if submap.tree is None:
                continue
            means3d = submap.params['means3D'].detach().clone().cuda()
            feats = submap.params['features'].detach().clone().cuda()
            
            if submap_manager.use_relative_pose and first_cam2world is not None:
                means_homo = torch.cat((means3d, torch.ones(means3d.shape[0], 1).cuda()), dim=1)
                means_world = torch.matmul(torch.from_numpy(submap.anchor_h).float().cuda(), means_homo.T)
                means3d = means_world[:3, :].T
        
            # compute relevancy
            recovered_feat = torch.matmul(feats.cuda(), self.clipdino.V)
            recovered_feat += self.clipdino.mean
            recovered_feat = recovered_feat / recovered_feat.norm(dim=-1, keepdim=True)
            text_embeds = self.clipdino.model.clip_backbone.decode_head.class_embeddings
            relevancy = torch.matmul(recovered_feat, text_embeds.t())
            relevancy = F.softmax(relevancy * 100, dim=-1)
            relevancy = relevancy[:, 1].detach().cpu().numpy()
            
            tree = submap.tree
            positions, depths, relevancies = self.compute_node_positions(tree, means3d, relevancy)
            # get points at depth 2
            depth2_points = [positions[node_id] for node_id, depth in depths.items() if depth == 2]
            depth2_relevancies = [relevancies[node_id] for node_id, depth in depths.items() if depth == 2]
            # get valid waypoints above relevancy threshold
            # get waypoints and their relevancies
            valid_waypoints = [(pos, rel) for pos, rel in zip(depth2_points, depth2_relevancies) if rel > 0] # 0 since sum is thresholded already
            valid_waypoints = [(pos.cpu().numpy(), rel) for pos, rel in valid_waypoints]
            
            submap.regions = valid_waypoints
            
    def check_termination(self, color_img, feat_img, depth_img, th=0.7):
        if not self.use_vlm:
            return None
        depth_th = 6.0
        
        # compute relevancy from feat img
        # rel_img = self.clipdino.get_relevancy_from_pca(feat_img.unsqueeze(0))[0][1]
        rel_img = self.clipdino.get_relevancy(feat_img.unsqueeze(0))[0][1]
        
        # get pixels above threshold
        mask = rel_img > th
        
        # check if mask is empty
        depth_img = depth_img.squeeze()
        if mask.sum() > 0:
            # check if any of these pixels are below depth threshold
            depth_valid = depth_img[mask]
            if depth_valid.min() < depth_th:
                vlm_query = self.query_vlm(color_img)
                return vlm_query
        return None
    
    def query_vlm(self, image):
        if not self.use_vlm:
            return None
        if self.vlm is None:
            self._load_vlm()
        task = self.task
        
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Are there {} in this image? Please answer with yes or no".format(task)},
                ],
            },
        ]
        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        if isinstance(image, torch.Tensor):
            image = Image.fromarray(image.detach().cpu().numpy().astype(np.uint8))
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.vlm.device, torch.float16)

        # autoregressively complete prompt
        output = self.vlm.generate(**inputs, max_new_tokens=100)
        out_msg = self.processor.decode(output[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        
        return out_msg
