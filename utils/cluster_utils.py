import os
import glob
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from collections import defaultdict

from tqdm import tqdm
from scene.gaussian_model import GaussianModel

import torch.nn.functional as F
import torch.nn as nn

from sklearn.neighbors import NearestNeighbors
from math import log, sqrt


# -----------------------------
# Debug helper
# -----------------------------
def debug_tensor(name, tensor):
    """
    Print detailed debug info about a tensor.

    Inputs:
        name: str        Name of the tensor
        tensor: torch.Tensor or None
    Outputs:
        Prints shape, dtype, device, min, max (or None)
    """
    if tensor is None:
        print(f"[DEBUG] {name}: None")
    else:
        print(
            f"[DEBUG] {name}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}, "
            f"device={tensor.device}, min={tensor.min().item()}, max={tensor.max().item()}"
        )

# -----------------------------
# Initialize Gaussian Instances (COLMAP seeds -> gaussians)
# -----------------------------
def initialize_gaussian_instances(gaussians, colmap_points, colmap_cluster_ids, temperature=0.1, max_dist=0.05):
    """
    Initialize Gaussian instance logits from COLMAP cluster seeds.

    Inputs:
        gaussians: GaussianModel (segmented)
        colmap_points: FloatTensor [Nc,3] COLMAP points
        colmap_cluster_ids: LongTensor [Nc] Cluster IDs
        temperature: float Softmax temperature
        max_dist: float Distance threshold for hard assignment

    Outputs:
        Updates gaussians.instance_logits [N_seg, G] and instance_ids [N_seg]
    """

    # Move everything to the same device as the Gaussian model
    device = gaussians.get_xyz.device
    xyz = gaussians.get_xyz.to(device)
    colmap_points = colmap_points.to(device)
    colmap_cluster_ids = colmap_cluster_ids.to(device)

    # Map COLMAP cluster IDs to a contiguous range [0, G-1]
    unique_ids = torch.unique(colmap_cluster_ids)  # unique cluster IDs
    G = unique_ids.numel()                          # number of clusters
    id_map = torch.full((int(colmap_cluster_ids.max().item()) + 1,), -1, dtype=torch.long, device=device)
    id_map[unique_ids] = torch.arange(G, device=device)
    colmap_cluster_ids = id_map[colmap_cluster_ids]  # remap cluster IDs

    # Prepare instance fields in the Gaussian model
    N = xyz.shape[0]  # number of Gaussians
    gaussians.set_instance_fields(N, G, device)

    # Compute pairwise distances between Gaussians and COLMAP points
    d = torch.cdist(xyz, colmap_points)  # shape [N, Nc]
    min_d, min_idx = d.min(dim=1)        # distance and index of closest COLMAP point for each Gaussian
    closest_cluster = colmap_cluster_ids[min_idx]  # cluster ID of closest point

    # Initialize logits for instance assignments
    logits = torch.zeros((N, G), device=device)

    # Hard assignment: assign fully to closest cluster if within max_dist
    mask = min_d <= max_dist
    logits[mask, closest_cluster[mask]] = 1.0

    # Soft/uniform assignment for Gaussians too far from any COLMAP point
    logits[~mask] = 1.0 / float(G)

    # Apply temperature-scaled softmax to get probability distribution over instances
    gaussians.instance_logits = torch.nn.Parameter(torch.softmax(logits / temperature, dim=1))

    # Hard assignment of each Gaussian to the cluster with highest probability
    gaussians.instance_ids = torch.argmax(gaussians.instance_logits, dim=1).detach()


def compute_topK_contributors(scene, K=8, bg_color=(0, 0, 0)):
    """
    Fast version: Computes the top-K contributing pixels for each Gaussian
    across all training cameras. No sorting, no Python Gaussian loops.

    Outputs:
        indices:   [N, C, K]  pixel indices (or -1)
        opacities: [N, C, K]  contribution weights (1.0)
    """
    
    from gaussian_renderer import render
    
    # ----------------------------------------
    # Prepare scene / buffers
    # ----------------------------------------
    gaussians = scene.gaussians
    N = gaussians.get_xyz.shape[0]
    cameras = scene.getTrainCameras()
    C = len(cameras)
    device = gaussians.get_xyz.device

    topK_indices = -torch.ones((N, C, K), dtype=torch.long, device=device)
    topK_opacities = torch.zeros((N, C, K), dtype=torch.float32, device=device)

    # insertion counters per camera
    # insert_pos[camera][gaussian] = next free slot [0..K]
    insert_pos = torch.zeros((C, N), dtype=torch.long, device=device)

    # Dummy render pipe
    class _DummyPipe:
        debug = False
        antialiasing = False
        compute_cov3D_python = False
        convert_SHs_python = False

    dummy_pipe = _DummyPipe()
    bg_col = torch.tensor(bg_color, dtype=torch.float32, device=device)

    # ----------------------------------------
    # Process each camera
    # ----------------------------------------
    for cam_idx, cam in enumerate(tqdm(cameras, desc="Computing top-K contributors")):
        with torch.no_grad():
            out = render(cam, gaussians, dummy_pipe, bg_col, contrib=True, K=K)

        if out is None or ("contrib_indices" not in out) or (out["contrib_indices"] is None):
            continue

        contrib = out["contrib_indices"]      # [H, W, K]
        H, W, _ = contrib.shape

        flat = contrib.view(-1, K)
        valid_mask = flat >= 0

        gauss_ids = flat[valid_mask].long()         # <-- FIX HERE
        pixel_ids = valid_mask.nonzero(as_tuple=True)[0]
        # Example:
        # flat:
        #     pixel 0: [ 4, -1, -1 ]
        #     pixel 1: [ 4,  7, -1 ]
        #
        # valid_mask:
        #     pixel 0: [1,0,0]
        #     pixel 1: [1,1,0]

        # ----------------------------------------
        # Assign pixel → Gaussian Top-K
        # ----------------------------------------
        # insertion position per Gaussian for this camera
        cam_insert = insert_pos[cam_idx]                  # [N]

        # positions where this Gaussian will be inserted
        pos = cam_insert[gauss_ids]                       # [M]

        # Only keep contributions where Gaussian has room (< K)
        keep = pos < K
        if not keep.any():
            continue

        gid = gauss_ids[keep]         # Gaussian ids to write
        pid = pixel_ids[keep]         # Pixel indices to write
        ppos = pos[keep].clamp(max=K-1)

        # Scatter into output tensors
        topK_indices[gid, cam_idx, ppos] = pid
        topK_opacities[gid, cam_idx, ppos] = 1.0

        # Advance counters
        cam_insert[gid] += keep.sum(dim=0) * 0  # only increase selected ones
        cam_insert[gid] = (cam_insert[gid] + 1).clamp(max=K)

        # Store back
        insert_pos[cam_idx] = cam_insert

    return {
        "indices": topK_indices,
        "opacities": topK_opacities,
    }


# -----------------------------
# Convert to pixel-centric responsibilities (vectorized)
# -----------------------------
def topK_to_responsibilities(topK_contrib):
    """
    Converts top-K contributors (Gaussian->pixels) -> pixel-centric format:
    r_point_idx, r_gauss_idx, r_vals (all tensors)
    """
    indices, opac = topK_contrib["indices"], topK_contrib["opacities"]
    device = indices.device

    N_gauss, C, K = indices.shape
    gauss_idx = torch.arange(N_gauss, device=device)[:, None, None].expand(-1, C, K)
    
    # Flatten
    flat_idx = indices.reshape(-1)
    flat_op = opac.reshape(-1)
    flat_gauss = gauss_idx.reshape(-1)

    # Keep only valid pixel indices
    mask = flat_idx >= 0
    if mask.sum() == 0:
        return (torch.empty(0, dtype=torch.long, device=device),
                torch.empty(0, dtype=torch.long, device=device),
                torch.empty(0, dtype=torch.float32, device=device))

    r_point_idx = flat_idx[mask]
    r_gauss_idx = flat_gauss[mask]
    r_vals = flat_op[mask]

    # Normalize per pixel using scatter_add
    uniq_pts, inv = torch.unique(r_point_idx, return_inverse=True)
    denom = torch.zeros_like(uniq_pts, dtype=r_vals.dtype, device=device)
    denom.scatter_add_(0, inv, r_vals)
    r_vals = r_vals / torch.clamp(denom[inv], min=1e-12)

    return r_point_idx, r_gauss_idx, r_vals


# -----------------------------
# Compute full -> segmented Gaussian mapping (vectorized if possible)
# -----------------------------
def compute_full_to_seg_map(trained_gs_seg, full_model):
    """
    Map full-resolution Gaussian indices -> segmented Gaussian indices.
    """
    device = full_model.get_xyz.device
    N_full, N_seg = full_model.get_xyz.shape[0], trained_gs_seg.get_xyz.shape[0]

    # If attribute exists, use it directly
    if hasattr(trained_gs_seg, "full_res_indices") and trained_gs_seg.full_res_indices is not None:
        kept_full = trained_gs_seg.full_res_indices.to(device).long()
        method = "attribute: full_res_indices"
    else:
        method = "fallback_nn_positions"
        full_xyz, seg_xyz = full_model.get_xyz.to(device), trained_gs_seg.get_xyz.to(device)
        batch = 8192
        ids = []
        for start in range(0, N_seg, batch):
            end = min(start + batch, N_seg)
            d = torch.cdist(seg_xyz[start:end], full_xyz)
            idx = d.argmin(dim=1)
            ids.append(idx)
        kept_full = torch.cat(ids).long()

    if (kept_full < 0).any() or (kept_full >= N_full).any():
        raise RuntimeError("kept_full contains invalid indices")

    # Map full -> segmented
    full_to_seg = torch.full((N_full,), -1, dtype=torch.long, device=device)
    full_to_seg[kept_full] = torch.arange(kept_full.numel(), device=device)

    print(f"[map] full_model N={N_full}, segmented N={N_seg}, method={method}")
    print(f"[map] mapped kept_full count = {kept_full.numel()}, mapped full->seg hits = {(full_to_seg>=0).sum().item()}")
    return full_to_seg, kept_full


# -----------------------------
# Map full Top-K -> segmented Top-K (vectorized)
# -----------------------------
def map_full_topK_to_segmented(topK_full, full_to_seg, kept_full):
    """
    Map full-resolution top-K Gaussian contributors -> segmented top-K
    """
    device = topK_full["indices"].device
    topK_indices_full = topK_full["indices"][kept_full]      # [N_seg, C, K]
    topK_opac_full = topK_full["opacities"][kept_full]      # [N_seg, C, K]

    # Map indices to segmented
    mapped_indices = topK_indices_full.clone()
    valid_mask = (mapped_indices >= 0) & (mapped_indices < full_to_seg.numel())
    mapped_indices[valid_mask] = full_to_seg[mapped_indices[valid_mask]]
    mapped_indices[~valid_mask] = -1

    # Zero out invalid opacities
    topK_opac_full[mapped_indices < 0] = 0.0

    # Debug
    # print(f"[DEBUG] topK_seg['indices'] min={mapped_indices.min().item()}, max={mapped_indices.max().item()}")
    # print(f"[DEBUG] topK_seg['opacities'] min={topK_opac_full.min().item()}, max={topK_opac_full.max().item()}")
    valid_contribs = (mapped_indices >= 0).sum().item()
    # print(f"[DEBUG] Segmented Gaussians with ≥1 contribution: {valid_contribs} / {kept_full.numel()}")

    return {"indices": mapped_indices, "opacities": topK_opac_full}


# -----------------------------
# Convert Gaussians -> pixels (vectorized version)
# -----------------------------
def convert_gauss_to_pixel_map(topK_full):
    """
    Convert Gaussian->pixel top-K map to pixel->Gaussian mapping (vectorized).
    """
    indices = topK_full["indices"]
    opacities = topK_full["opacities"]
    device = indices.device
    N_gauss, C, K = indices.shape

    # Flatten
    flat_idx = indices.reshape(-1)
    flat_gauss = torch.arange(N_gauss, device=device)[:, None, None].expand(-1, C, K).reshape(-1)
    flat_op = opacities.reshape(-1)

    # Filter valid contributions
    mask = (flat_idx >= 0) & (flat_op > 0)
    flat_idx, flat_gauss = flat_idx[mask], flat_gauss[mask]

    # Build dictionary: pixel -> list of gaussians
    pixel_to_gauss = {}
    for pix, g in zip(flat_idx.tolist(), flat_gauss.tolist()):
        pixel_to_gauss.setdefault(pix, []).append(g)

    # Flatten to list of tuples (tensor of gaussians, mask_val=1.0)
    mappings = [(torch.tensor(glist, dtype=torch.long, device=device), 1.0)
                for glist in pixel_to_gauss.values()]

    print(f"[INFO] Converted {len(mappings)} pixels → Gaussians")
    return mappings


# -----------------------------
# Compute instance coherence
# -----------------------------
def compute_instance_coherence(logits, ids):
    """
    logits: [N, C] tensor of per-Gaussian instance logits
    ids:    [N] tensor of final instance IDs

    Returns:
        coherence: [N] tensor of per-Gaussian coherence scores
    """
    probs = torch.softmax(logits, dim=1)
    coherence = probs[torch.arange(len(ids)), ids]
    return coherence


# -----------------------------
# Filter Coherent Gaussians
# -----------------------------

def filter_coherent_gaussians(gaussian_model, threshold=0.6):
    """
    Filter a GaussianModel by instance coherence.

    Inputs:
        gaussian_model: GaussianModel instance (segmented)
        threshold: float, coherence threshold (0.6 by default)

    Returns:
        filtered_gaussian_model: new GaussianModel containing only coherent Gaussians
        mask: boolean mask of kept Gaussians
    """
    # Retrieve data from the model
    xyz = gaussian_model.get_xyz
    logits = gaussian_model.instance_logits
    ids = gaussian_model.instance_ids

    # Compute coherence
    coherence = compute_instance_coherence(logits, ids)

    # --- DEBUG: Inspect coherence distribution ---
    # print("[DEBUG] Coherence stats:",
    #       f"min={coherence.min().detach().item():.4f}, "
    #       f"max={coherence.max().detach().item():.4f}, "
    #       f"mean={coherence.mean().detach().item():.4f}, "
    #       f"median={coherence.median().detach().item():.4f}")

    mask = coherence >= threshold
    kept_count = mask.sum().item()
    total_count = xyz.shape[0]
    # print(f"[INFO] Keeping {kept_count} / {total_count} coherent Gaussians (TH={threshold})")

    # Create new GaussianModel with filtered attributes
    filtered_gs = GaussianModel(sh_degree=gaussian_model.active_sh_degree)
    filtered_gs._xyz = nn.Parameter(xyz[mask].clone().detach())
    filtered_gs._features_dc = nn.Parameter(gaussian_model._features_dc[mask].clone().detach())
    filtered_gs._features_rest = nn.Parameter(gaussian_model._features_rest[mask].clone().detach())
    filtered_gs._scaling = nn.Parameter(gaussian_model._scaling[mask].clone().detach())
    filtered_gs._rotation = nn.Parameter(gaussian_model._rotation[mask].clone().detach())
    filtered_gs._opacity = nn.Parameter(gaussian_model._opacity[mask].clone().detach())
    filtered_gs.semantic_mask = nn.Parameter(gaussian_model.semantic_mask[mask].clone().detach()) if gaussian_model.semantic_mask is not None else None

    # Copy instance info
    filtered_gs.instance_logits = nn.Parameter(logits[mask].clone().detach())
    filtered_gs.instance_ids = ids[mask].clone().detach()

    return filtered_gs, mask


# -----------------------------
# Compute Mask Centroids
# -----------------------------

def compute_mask_centroids(mask_dir):
    """
    Compute centroids for all mask instances inside a folder.
    Accepts ANY image extension (png, jpg, jpeg, bmp, tif, tiff, webp, …)

    Returns:
        dict: { (frame_name, instance_idx): (cx, cy) }
    """
    mask_dir = Path(mask_dir)

    # Allowed extensions
    exts = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff", "*.webp"]

    # Collect files
    mask_files = []
    for ext in exts:
        mask_files.extend(mask_dir.glob(ext))

    centroids = {}

    for mask_path in mask_files:
        stem = mask_path.stem

        # Only accept files containing '_instance_X'
        if "_instance_" not in stem:
            continue

        try:
            frame_name, idx_str = stem.rsplit("_instance_", 1)
            midx = int(idx_str)
        except:
            continue

        # Load mask as numpy
        mask = np.array(Image.open(mask_path).convert("L"))
        ys, xs = np.where(mask > 128)  # foreground pixels

        if len(xs) == 0:
            continue

        cx = float(xs.mean())
        cy = float(ys.mean())

        centroids[(frame_name, midx)] = (cx, cy)

    return centroids



# --------------------------------------------------------------------
# Map dt_norm -> dt_real using dt_scale from centroid distances
# --------------------------------------------------------------------
def dt_norm_to_real(dt_norm, dt_scale):
    """
    Simple linear mapping: dt_norm in [0,1] maps to dt_real in pixels/units.
    """
    dt_real = dt_norm * dt_scale
    return dt_real

# --------------------------------------------------------------------
# Refine the clusters iteratively with a metrics
# --------------------------------------------------------------------
def refine_clusters_with_metric(
    self,
    max_iters=5,
    gather_alpha=0.5,
    explore_ratio=0.1,
    w_geom=1.0,
    w_coh=1.0,
    w_size=1.0,
    w_count=0.5,
    alpha=1.0,
    spatial_consistency_weight=0.1,
):
    """
    Refine self.point_clusters maximizing combined score:
        score = w_geom * geom_term * (coherence ** alpha)
              + w_size * size_balance
              + w_count * cluster_count_term
              + spatial consistency regularization
    """
    eps = 1e-8

    def get_xyz(pid):
        p = self.points3D[pid]
        return np.asarray(p.xyz, dtype=np.float32).reshape(-1)[:3]

    def compute_label_coherence(points3D, clusters_map, k=8):
        pids = [pid for pid in clusters_map.keys() if clusters_map[pid] != -1]
        if not pids:
            return 1.0
        X = np.vstack([get_xyz(pid) for pid in pids])
        labels = np.array([clusters_map[pid] for pid in pids], dtype=int)
        nbrs = NearestNeighbors(n_neighbors=min(k+1, X.shape[0]), algorithm='kd_tree').fit(X)
        _, indices = nbrs.kneighbors(X)

        coherences = []
        for i, neigh_idx in enumerate(indices):
            neigh_idx = neigh_idx[1:]
            same = (labels[neigh_idx] == labels[i])
            coherences.append(np.mean(same))
        return float(np.mean(coherences))

    clusters = dict(self.point_clusters)
    pids_sorted = sorted(self.points3D.keys())

    def build_cluster_points_map(clmap):
        cmap = defaultdict(list)
        for pid in pids_sorted:
            cid = clmap.get(pid, -1)
            if cid != -1:
                cmap[cid].append(pid)
        return cmap

    K0 = max(1, len(build_cluster_points_map(clusters)))

    for iteration in range(max_iters):
        cluster_points = build_cluster_points_map(clusters)
        centroids = {cid: np.mean([get_xyz(pid) for pid in pts], axis=0) for cid, pts in cluster_points.items()}
        cid_list = list(centroids.keys())

        intra_dists = {cid: np.linalg.norm(np.vstack([get_xyz(pid) for pid in pts]) - centroids[cid], axis=1).mean()
                       for cid, pts in cluster_points.items()}
        inter_dists = [np.linalg.norm(centroids[cid_list[i]] - centroids[cid_list[j]])
                       for i in range(len(cid_list)) for j in range(i + 1, len(cid_list))]

        mean_intra = float(np.mean(list(intra_dists.values()))) if intra_dists else 0.0
        mean_inter = float(np.mean(inter_dists)) if inter_dists else 0.0

        coherence = compute_label_coherence(self.points3D, clusters, k=8)

        counts = np.array([len(cluster_points[c]) for c in cluster_points.keys()], dtype=np.float32)
        N = max(1, counts.sum())
        p = counts / (N + eps)
        K = max(1, len(p))
        size_balance = 0.0 if K <= 1 else float(-np.sum(p * np.log(np.where(p > 0, p, 1.0) + eps)) / (np.log(K) + eps))
        cluster_count_term = float(sqrt(K) / (sqrt(K0) + eps))

        geom_term = mean_inter / (mean_intra + eps)
        geom_with_coh = geom_term * (coherence ** alpha)
        score = (w_geom * geom_with_coh) + (w_size * size_balance) + (w_count * cluster_count_term)

        # Gathering: merge close centroids
        thresh = gather_alpha * (mean_inter + eps)
        merge_candidates = [(cid_list[i], cid_list[j])
                            for i in range(len(cid_list)) for j in range(i+1, len(cid_list))
                            if np.linalg.norm(centroids[cid_list[i]] - centroids[cid_list[j]]) < thresh]
        for a, b in merge_candidates:
            for pid in cluster_points[b]:
                clusters[pid] = a

        # Reassignment with spatial consistency
        cluster_points = build_cluster_points_map(clusters)
        centroids = {cid: np.mean([get_xyz(pid) for pid in pts], axis=0) for cid, pts in cluster_points.items()}
        cid_list = list(centroids.keys())
        temperature = explore_ratio * (1 - iteration / max_iters)

        for pid in pids_sorted:
            if clusters[pid] == -1:
                continue
            current_cid = clusters[pid]
            if len(cluster_points.get(current_cid, [])) <= 1:
                continue

            best_cid = current_cid
            best_score = score
            for cid in cid_list:
                if cid == current_cid:
                    continue
                dist = np.linalg.norm(get_xyz(pid) - centroids[cid])
                spatial_factor = np.exp(-spatial_consistency_weight * dist)
                tmp_score = geom_with_coh * spatial_factor

                if tmp_score > best_score:
                    best_score = tmp_score
                    best_cid = cid
                elif temperature > 0:
                    delta = tmp_score - score
                    if delta < 0 and np.random.rand() < np.exp(delta / (temperature + eps)):
                        best_score = tmp_score
                        best_cid = cid
            clusters[pid] = best_cid

    self.point_clusters = clusters
    return clusters

