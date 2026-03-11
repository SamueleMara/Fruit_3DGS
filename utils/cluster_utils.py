import os
import glob
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from collections import defaultdict

from tqdm import tqdm
from scene.gaussian_model import GaussianModel
# from gaussian_renderer import render

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
def initialize_gaussian_instances(
    gaussians,
    colmap_points,
    colmap_cluster_ids,
    temperature=0.1,
    max_dist=0.05,
    batch_colmap=16384,
):
    """
    Initialize Gaussian instance logits from COLMAP cluster seeds.

    CPU-based NN assignment (OOM-safe).
    """

    device = gaussians.get_xyz.device

    # -------------------------------------------------
    # Move data to CPU for NN search
    # -------------------------------------------------
    xyz_cpu = gaussians.get_xyz.detach().cpu()          # [N, 3]
    colmap_xyz = torch.as_tensor(colmap_points, dtype=torch.float32).cpu()  # [Nc, 3]
    colmap_ids = torch.as_tensor(colmap_cluster_ids, dtype=torch.long)

    N = xyz_cpu.shape[0]

    # -------------------------------------------------
    # Map COLMAP cluster IDs → contiguous [0, G-1]
    # -------------------------------------------------
    unique_ids, inv_ids = torch.unique(colmap_ids, return_inverse=True)
    G = unique_ids.numel()
    colmap_ids = inv_ids.to(device)

    # -------------------------------------------------
    # Nearest neighbor search (CPU, streamed)
    # -------------------------------------------------
    best_dist = torch.full((N,), float("inf"))
    best_idx  = torch.full((N,), -1, dtype=torch.long)

    with torch.no_grad():
        for c0 in range(0, colmap_xyz.shape[0], batch_colmap):
            c1 = min(c0 + batch_colmap, colmap_xyz.shape[0])
            block = colmap_xyz[c0:c1]  # [B, 3]

            # squared distances
            d = (
                (xyz_cpu[:, None, :] - block[None, :, :])
                .pow(2)
                .sum(dim=2)
            )  # [N, B]

            block_min, block_idx = d.min(dim=1)
            better = block_min < best_dist

            best_dist[better] = block_min[better]
            best_idx[better]  = block_idx[better] + c0

            del d

    # Move NN results to GPU
    best_dist = best_dist.sqrt().to(device)
    best_idx  = best_idx.to(device)

    # -------------------------------------------------
    # Prepare Gaussian instance fields
    # -------------------------------------------------
    gaussians.set_instance_fields(N, G, device)

    closest_cluster = colmap_ids[best_idx]  # [N]

    # -------------------------------------------------
    # Initialize logits
    # -------------------------------------------------
    logits = torch.zeros((N, G), device=device)

    hard_mask = best_dist <= max_dist
    logits[hard_mask, closest_cluster[hard_mask]] = 1.0

    # uniform fallback
    logits[~hard_mask] = 1.0 / float(G)

    # temperature-scaled softmax
    logits = torch.softmax(logits / temperature, dim=1)

    gaussians.instance_logits = torch.nn.Parameter(logits)
    gaussians.instance_ids = logits.argmax(dim=1).detach()



def compute_topK_contributors(scene, K=8, bg_color=(0, 0, 0), CAM_BATCH=16):
    """
    Memory-efficient version:
    Computes top-K contributing pixels for each Gaussian across all cameras
    using camera batching so GPU memory stays small.

    Outputs:
        indices:   [N, C, K] on CPU
        opacities: [N, C, K] on CPU
    """
    from gaussian_renderer import render

    gaussians = scene.gaussians
    N = gaussians.get_xyz.shape[0]
    cameras = scene.getTrainCameras()
    C = len(cameras)
    device = gaussians.get_xyz.device

    # Store final outputs on CPU
    topK_indices = -torch.ones((N, C, K), dtype=torch.long, device="cpu")
    topK_opacities = torch.zeros((N, C, K), dtype=torch.float32, device="cpu")

    # Insertion positions per camera (CPU)
    insert_pos = torch.zeros((C, N), dtype=torch.long, device="cpu")

    # Dummy pipe
    class _DummyPipe:
        debug = False
        antialiasing = False
        compute_cov3D_python = False
        convert_SHs_python = False

    dummy_pipe = _DummyPipe()
    bg_col = torch.tensor(bg_color, dtype=torch.float32, device=device)

    # Batch cameras to avoid GPU memory overflow
    for cam_batch_start in range(0, C, CAM_BATCH):
        cam_batch_end = min(cam_batch_start + CAM_BATCH, C)
        cam_batch = cameras[cam_batch_start:cam_batch_end]

        for local_idx, cam in enumerate(
            tqdm(cam_batch, desc=f"Top-K (cams {cam_batch_start}-{cam_batch_end})")
        ):
            cam_idx = cam_batch_start + local_idx

            # AMP context (non-autocast)
            with torch.no_grad(), torch.amp.autocast("cuda", enabled=False):
                out = render(cam, gaussians, dummy_pipe, bg_col, contrib=True, K=K)

            # Skip if render failed
            if out is None or ("contrib_indices" not in out) or (out["contrib_indices"] is None):
                continue

            contrib = out["contrib_indices"]      # [H, W, K] on GPU
            H, W, _ = contrib.shape

            flat = contrib.view(-1, K)
            valid_mask = flat >= 0

            gauss_ids = flat[valid_mask].long()
            pixel_ids = valid_mask.nonzero(as_tuple=True)[0]

            cam_insert = insert_pos[cam_idx].to(device)
            pos = cam_insert[gauss_ids]
            keep = pos < K
            if not keep.any():
                del contrib, flat, valid_mask, gauss_ids, pixel_ids, cam_insert
                torch.cuda.empty_cache()
                continue

            gid = gauss_ids[keep]
            pid = pixel_ids[keep]
            ppos = pos[keep].clamp(max=K-1)

            # Move to CPU and scatter
            gid_cpu, pid_cpu, ppos_cpu = gid.cpu(), pid.cpu(), ppos.cpu()
            topK_indices[gid_cpu, cam_idx, ppos_cpu] = pid_cpu
            topK_opacities[gid_cpu, cam_idx, ppos_cpu] = 1.0

            # Update insertion positions
            cam_insert_cpu = cam_insert.cpu()
            cam_insert_cpu[gid_cpu] = (cam_insert_cpu[gid_cpu] + 1).clamp(max=K)
            insert_pos[cam_idx] = cam_insert_cpu

            # Clean GPU memory
            del contrib, flat, valid_mask, gauss_ids, pixel_ids, gid, pid, ppos, cam_insert, out
            torch.cuda.empty_cache()

    return {
        "indices": topK_indices,
        "opacities": topK_opacities,
    }

# -----------------------------
# Convert to pixel-centric responsibilities (vectorized)
# -----------------------------
def topK_to_responsibilities(topK_contrib, batch_size=2_000_000):
    """
    Memory-efficient version of topK_to_responsibilities().
    Processes top-K (N,C,K) tensor in batches instead of flattening everything.

    Returns:
        r_point_idx, r_gauss_idx, r_vals   (all on CPU)
    """
    indices = topK_contrib["indices"].cpu()     # [N, C, K]
    opac    = topK_contrib["opacities"].cpu()   # [N, C, K]

    N, C, K = indices.shape

    # Precompute Gaussian ids
    gauss_idx_full = (
        torch.arange(N, dtype=torch.long).view(N, 1, 1)
        .expand(N, C, K)
        .reshape(-1)
    )

    # Flatten ALL data ON CPU
    flat_idx_full  = indices.reshape(-1)        # pixel idx or -1
    flat_op_full   = opac.reshape(-1)

    # Mask of valid contributions
    valid = flat_idx_full >= 0
    total_valid = valid.sum().item()
    if total_valid == 0:
        return (
            torch.empty(0, dtype=torch.long),
            torch.empty(0, dtype=torch.long),
            torch.empty(0, dtype=torch.float32),
        )

    # Final result containers (CPU)
    r_point_idx_list = []
    r_gauss_idx_list = []
    r_vals_list      = []

    # ----------- PROCESS IN CHUNKS -----------
    # each chunk ~2M entries → safe for RAM/GPU
    pos_array = torch.nonzero(valid, as_tuple=False).view(-1).split(batch_size)

    for pos_chunk in pos_array:
        # gather chunk
        p_idx = flat_idx_full[pos_chunk]
        g_idx = gauss_idx_full[pos_chunk]
        vals  = flat_op_full[pos_chunk]

        r_point_idx_list.append(p_idx)
        r_gauss_idx_list.append(g_idx)
        r_vals_list.append(vals)

    # Concatenate final arrays
    r_point_idx = torch.cat(r_point_idx_list, dim=0)
    r_gauss_idx = torch.cat(r_gauss_idx_list, dim=0)
    r_vals      = torch.cat(r_vals_list, dim=0)

    # ----------- NORMALIZATION (BATCHED) -----------
    # normalize per pixel (point index)
    uniq_pts, inv = torch.unique(r_point_idx, return_inverse=True)

    # denom per unique pixel
    denom = torch.zeros_like(uniq_pts, dtype=torch.float32)
    denom.scatter_add_(0, inv, r_vals)

    # final normalized responsibilities
    r_vals = r_vals / torch.clamp(denom[inv], min=1e-12)

    # Clean up variables safely
    del indices, opac, flat_idx_full, flat_op_full, gauss_idx_full
    torch.cuda.empty_cache()

    return r_point_idx, r_gauss_idx, r_vals


# -----------------------------
# Compute full -> segmented Gaussian mapping (vectorized if possible)
# -----------------------------
def compute_full_to_seg_map(
    trained_gs_seg,
    full_model,
    batch_seg=2048,
    batch_full=16384,
):
    """
    CPU-based NN mapping (OOM-proof).
    Uses streamed blocks, no large intermediate allocations.
    """
    full_xyz = full_model.get_xyz.detach().cpu()        # [N_full, 3]
    seg_xyz  = trained_gs_seg.get_xyz.detach().cpu()   # [N_seg, 3]

    N_full = full_xyz.shape[0]
    N_seg  = seg_xyz.shape[0]

    kept_full = []

    print("[map] Computing NN positions (CPU streamed)...")

    # Precompute full-model squared norms once
    full_norm = (full_xyz ** 2).sum(dim=1)   # [N_full]

    # ---- tqdm over segmented Gaussians ----
    seg_range = range(0, N_seg, batch_seg)
    for s0 in tqdm(seg_range, desc="[map] Seg blocks", total=len(seg_range)):
        s1 = min(s0 + batch_seg, N_seg)
        seg_block = seg_xyz[s0:s1]           # [bs, 3]
        bs = seg_block.shape[0]

        # squared norms for seg block
        seg_norm = (seg_block ** 2).sum(dim=1)  # [bs]

        best_dist = torch.full((bs,), float("inf"))
        best_idx  = torch.full((bs,), -1, dtype=torch.long)

        # ---- streamed full-model blocks ----
        for f0 in range(0, N_full, batch_full):
            f1 = min(f0 + batch_full, N_full)
            full_block = full_xyz[f0:f1]     # [bf, 3]

            # dist^2 = ||x||^2 + ||y||^2 - 2 x·y
            d = (
                seg_norm[:, None]
                + full_norm[f0:f1][None, :]
                - 2.0 * (seg_block @ full_block.T)
            )  # [bs, bf]

            block_min_dist, block_min_pos = d.min(dim=1)

            better = block_min_dist < best_dist
            best_dist[better] = block_min_dist[better]
            best_idx[better]  = block_min_pos[better] + f0

            del d, full_block, block_min_dist, block_min_pos, better

        kept_full.append(best_idx)

        del seg_block, seg_norm, best_dist, best_idx

    kept_full = torch.cat(kept_full)  # [N_seg]

    # Move back to GPU only final outputs
    device = full_model.get_xyz.device
    kept_full = kept_full.to(device)

    full_to_seg = torch.full(
        (N_full,), -1, dtype=torch.long, device=device
    )
    full_to_seg[kept_full] = torch.arange(N_seg, device=device)

    print(f"[map] Done. Mapped {N_seg} seg → {N_full} full.")
    return full_to_seg, kept_full

# -----------------------------
# Map full Top-K -> segmented Top-K (vectorized)
# -----------------------------
def map_full_topK_to_segmented(topK_full, full_to_seg, kept_full, batch=4096):
    """
    Map full-model top-K contributions to segmented-model top-K.
    Batched, memory-safe, device-consistent, OOM-proof.

    Inputs:
        topK_full: dict with "indices" [N_full, C, K] and "opacities"
        full_to_seg: LongTensor [N_full] mapping full -> seg
        kept_full: LongTensor [N_seg], indices of full Gaussians kept
        batch: batch size for segmented Gaussians

    Returns:
        dict with "indices" and "opacities" for segmented Gaussians
    """
    device = full_to_seg.device
    topK_full_indices = topK_full["indices"]
    topK_full_opac   = topK_full["opacities"]

    N_seg = kept_full.numel()
    C, K = topK_full_indices.shape[1], topK_full_indices.shape[2]

    # Preallocate outputs on CPU to be safe
    mapped_indices = -torch.ones((N_seg, C, K), dtype=torch.long, device="cpu")
    mapped_opac    = torch.zeros((N_seg, C, K), dtype=torch.float32, device="cpu")

    for s0 in range(0, N_seg, batch):
        s1 = min(s0 + batch, N_seg)
        batch_kept = kept_full[s0:s1].cpu()  # keep indices on CPU

        # Gather full topK for this batch
        src_idx = topK_full_indices[batch_kept]  # [B, C, K]
        src_op  = topK_full_opac[batch_kept]

        # Map to segmented indices safely
        # - clamp to valid range first
        src_idx_clamped = src_idx.clone()
        src_idx_clamped = torch.clamp(src_idx_clamped, 0, full_to_seg.numel() - 1)

        seg_ids = full_to_seg[src_idx_clamped].cpu()  # map to seg, still CPU

        # Set invalid mappings to -1
        seg_ids[~(src_idx >= 0)] = -1
        src_op_mapped = torch.where(seg_ids >= 0, src_op.cpu(), torch.zeros_like(src_op.cpu()))

        # Write to preallocated outputs
        mapped_indices[s0:s1] = seg_ids
        mapped_opac[s0:s1] = src_op_mapped

        # Free memory
        del src_idx, src_op, src_idx_clamped, seg_ids, src_op_mapped
        torch.cuda.empty_cache()

    return {"indices": mapped_indices, "opacities": mapped_opac}


# -----------------------------
# Convert Gaussians -> pixels (vectorized version)
# -----------------------------
def convert_gauss_to_pixel_map(topK_full, batch=2_000_000):
    """
    Batched & memory-safe:
    Converts Gaussian→pixel top-K into pixel→Gaussian list.
    """
    indices = topK_full["indices"].cpu()
    op      = topK_full["opacities"].cpu()

    N, C, K = indices.shape
    gauss_idx_full = (
        torch.arange(N, dtype=torch.long)[:, None, None]
        .expand(N, C, K).reshape(-1)
    )

    flat_idx = indices.reshape(-1)
    flat_op  = op.reshape(-1)

    valid = (flat_idx >= 0) & (flat_op > 0)
    pos_batches = torch.nonzero(valid, as_tuple=False).view(-1).split(batch)

    pixel_map = {}

    for pos in pos_batches:
        pidx = flat_idx[pos]
        gidx = gauss_idx_full[pos]

        for p, g in zip(pidx.tolist(), gidx.tolist()):
            if p not in pixel_map:
                pixel_map[p] = [g]
            else:
                pixel_map[p].append(g)

    # Convert dict → required list format
    mappings = [(torch.tensor(glist, dtype=torch.long), 1.0)
                for glist in pixel_map.values()]

    print(f"[INFO] Pixel→Gaussian entries: {len(mappings)}")
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
