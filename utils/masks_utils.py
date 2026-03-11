# masks_utils.py
import json
import re
from pathlib import Path
from functools import lru_cache
from tqdm import tqdm
import numpy as np
from PIL import Image
import os
import itertools
import csv
from collections import defaultdict, deque, Counter
import cv2
import torch
import glob
import imageio
import psutil
import random

from .graphics_utils import getWorld2View2, getProjectionMatrix
from .read_write_model import qvec2rotmat

# ---------------- Mask Loading & Caching ---------------- #

@lru_cache(maxsize=1024)
def load_mask_cpu(mask_path, downsample=1):
    mask = np.array(Image.open(mask_path))
    if mask.ndim == 3:
        mask = mask[..., 0]
    mask = (mask > 0).astype(np.uint8)
    if downsample > 1:
        mask = mask[::downsample, ::downsample]
    return mask

def list_masks_for_frame(mask_dir, frame_name, log=print):
    mask_dir = Path(mask_dir)
    mask_files = sorted(
        mask_dir.glob(f"{frame_name}_instance_*.png"),
        key=lambda p: int(re.search(r'_instance_(\d+)', p.stem).group(1))
    )
    if not mask_files:
        log(f"[WARN] No mask instances found for frame {frame_name} in {mask_dir}")
    return mask_files

def compute_mask_instances_json(mask_dir, downsample=1, save_path=None, log=print):
    mask_dir = Path(mask_dir)
    mask_files = sorted(mask_dir.glob("*_instance_*.png"))
    mask_instances = {}

    for mask_path in tqdm(mask_files, desc="[mask_utils] Extracting mask_instances"):
        stem = mask_path.stem
        frame_name, midx_str = stem.rsplit("_instance_", 1)
        midx = int(midx_str)

        mask = load_mask_cpu(str(mask_path), downsample)
        if mask.sum() == 0:
            continue

        ys, xs = np.nonzero(mask)
        cx, cy = float(xs.mean()), float(ys.mean())
        xmin, xmax = int(xs.min()), int(xs.max())
        ymin, ymax = int(ys.min()), int(ys.max())
        area = int(mask.sum())

        if frame_name not in mask_instances:
            mask_instances[frame_name] = {}
        mask_instances[frame_name][midx] = {
            "centroid": [cx, cy],
            "bbox": [xmin, ymin, xmax, ymax],
            "area": area
        }

    if save_path:
        with open(save_path, "w") as f:
            json.dump(mask_instances, f, indent=2)
        log(f"[INFO] Saved mask_instances JSON to {save_path}")

    return mask_instances

def load_or_create_mask_instances(mask_dir, downsample=1, log=print):
    mask_dir = Path(mask_dir)
    json_path = mask_dir / "mask_instances.json"
    if json_path.exists():
        with open(json_path, "r") as f:
            mask_instances = json.load(f)
        log(f"[INFO] Loaded mask_instances from {json_path}")
    else:
        log("[INFO] No mask_instances.json found, computing...")
        mask_instances = compute_mask_instances_json(mask_dir, downsample, save_path=json_path, log=log)
    return mask_instances

def get_mask_info(mask_instances, frame_name, midx):
    """
    Retrieve mask instance info given frame_name and midx.
    Returns dict with keys: centroid, bbox, area, or None if missing.
    """
    frame_data = mask_instances.get(frame_name, {})
    # Try integer midx first, fallback to string key
    return frame_data.get(midx) or frame_data.get(str(midx))

def mask_centroid_and_bbox(mask_instances, frame_name, midx):
    info = get_mask_info(mask_instances, frame_name, midx)
    if info is None:
        return None, None
    cx, cy = info["centroid"]
    xmin, ymin, xmax, ymax = info["bbox"]
    return (cx, cy), (xmin, ymin, xmax, ymax)

def mask_area(mask_instances, frame_name, midx):
    info = get_mask_info(mask_instances, frame_name, midx)
    return info["area"] if info else 0

def to_python(obj):
    if isinstance(obj, dict):
        return {to_python(k): to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_python(v) for v in obj]
    elif isinstance(obj, tuple):
        return [to_python(v) for v in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    else:
        return obj


def load_full_mask_point_mapping(json_path, log=print):
    with open(json_path, "r") as f:
        data = json.load(f)

    mask_instances = data.get("mask_instances", {})
    point_to_masks = defaultdict(list)
    for pid_str, masks in data.get("point_to_masks", {}).items():
        point_to_masks[int(pid_str)] = [(f, int(midx)) for f, midx in masks]

    mask_to_points = defaultdict(list)
    for key, pids in data.get("mask_to_points", {}).items():
        frame, midx = key.rsplit("_", 1)
        mask_to_points[(frame, int(midx))] = [int(pid) for pid in pids]

    log(f"[INFO] Loaded full mask-point mapping from {json_path}")
    return mask_instances, point_to_masks, mask_to_points


def parse_mapping_from_file(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    # mask_instances: nested dict {frame: {midx: {...}}}
    mask_instances = data.get("mask_instances", {})
    # point_to_masks: keys are strings of pid -> list of [frame, midx]
    raw_pt2m = data.get("point_to_masks", {})
    point_to_masks = {int(pid): [(f, int(midx)) for f, midx in masks] for pid, masks in raw_pt2m.items()}
    # mask_to_points: keys are "frame_midx" strings -> list of pids
    raw_m2p = data.get("mask_to_points", {})
    mask_to_points = {}
    for key, pids in raw_m2p.items():
        # key may look like "frame_00030_4" or "frame_00030_4" (frame can have underscores)
        frame, midx = key.rsplit("_", 1)
        mask_to_points[(frame, int(midx))] = [int(p) for p in pids]
    return mask_instances, point_to_masks, mask_to_points


def compute_mask_overlaps(point_to_masks, mask_to_points, min_shared=1, top_k=200, log=print):
    """
    Efficiently compute overlapping counts between mask pairs using point->mask lists.
    Returns list of (maskA, maskB, shared_count, jaccard, sizeA, sizeB) sorted by shared_count desc.
    mask keys are tuples (frame, midx).
    """
    # Build map of mask -> size
    mask_size = {mask: len(pids) for mask, pids in mask_to_points.items()}

    # use co-occurrence counting: for each point, increment counter for all combinations of masks seeing that point
    overlap_counts = defaultdict(int)
    for pid, masks in point_to_masks.items():
        # masks is list of [frame, midx] -> convert to tuple keys
        mask_keys = [tuple(m) for m in masks]
        if len(mask_keys) < 2:
            continue
        for a, b in itertools.combinations(sorted(mask_keys), 2):
            overlap_counts[(a, b)] += 1

    # compute jaccard
    results = []
    for (a, b), shared in overlap_counts.items():
        sizeA = mask_size.get(a, 0)
        sizeB = mask_size.get(b, 0)
        union = sizeA + sizeB - shared if (sizeA + sizeB - shared) > 0 else 1
        jaccard = shared / union
        if shared >= min_shared:
            results.append((a, b, shared, jaccard, sizeA, sizeB))

    results = sorted(results, key=lambda x: (-x[2], -x[3]))  # sort by shared_count desc then jaccard
    log(f"[INFO] Computed overlaps for {len(results)} mask-pairs (min_shared={min_shared})")
    return results[:top_k]

# -------------------------
# Merge masks by Jaccard
# -------------------------
def merge_masks_by_jaccard(point_to_masks, mask_to_points, jaccard_threshold=0.5, min_shared=1):
    """
    Merge mask instances based on Jaccard similarity.

    Returns:
        merged_groups : list[list]
    """
    overlaps = compute_mask_overlaps(point_to_masks, mask_to_points, min_shared=min_shared, top_k=10**9, log=lambda *a, **k: None)

    parent = {}
    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for a, b, shared, jaccard, sA, sB in overlaps:
        if jaccard >= jaccard_threshold:
            union(a, b)

    groups = defaultdict(list)
    for m in mask_to_points.keys():
        groups[find(m)].append(m)

    merged_groups = [sorted(g) for g in groups.values() if len(g) > 1]
    return merged_groups

def bipartite_connected_components(point_to_masks, mask_to_points, log=print):
    """
    Compute connected components of bipartite graph (mask nodes and point nodes).
    Returns list of components each as {'masks': set(...), 'points': set(...)}.
    """
    # Build adjacency: nodes as 'p:{pid}' and 'm:frame_midx'
    adj = defaultdict(set)
    for pid, masks in point_to_masks.items():
        pid_node = f"p:{pid}"
        for f, midx in masks:
            m_node = f"m:{f}_{midx}"
            adj[pid_node].add(m_node)
            adj[m_node].add(pid_node)

    visited = set()
    components = []
    for node in adj.keys():
        if node in visited:
            continue
        q = deque([node])
        comp_nodes = set()
        while q:
            n = q.popleft()
            if n in visited:
                continue
            visited.add(n)
            comp_nodes.add(n)
            for nb in adj[n]:
                if nb not in visited:
                    q.append(nb)
        masks = {tuple(n.split("m:")[1].rsplit("_", 1)) if n.startswith("m:") else None for n in comp_nodes}
        # better parsing:
        comp_masks = set()
        comp_points = set()
        for n in comp_nodes:
            if n.startswith("m:"):
                rest = n[2:]
                frame, midx = rest.rsplit("_", 1)
                comp_masks.add((frame, int(midx)))
            elif n.startswith("p:"):
                comp_points.add(int(n[2:]))
        components.append({'masks': comp_masks, 'points': comp_points})
    log(f"[INFO] Found {len(components)} bipartite connected components")
    return components

def recommend_merge_parameter(point_to_masks, mask_to_points, GT=None, log=print):
    """
    Heuristic recommendation:
      - compute distribution of jaccard values and shared counts,
      - identify elbow (percentiles) and recommend candidate thresholds to try.
    If GT is provided, it also outputs closest thresholds that make #objects ~= GT.
    Returns a dict with candidate thresholds and simple guidance.
    """
    overlaps = compute_mask_overlaps(point_to_masks, mask_to_points, min_shared=1, top_k=10**6, log=log)
    if not overlaps:
        log("[WARN] No mask overlaps found")
        return {}

    jaccards = [j for (_, _, _, j, _, _) in overlaps]
    shareds = [s for (_, _, s, _, _, _) in overlaps]

   
    p50_j = float(np.percentile(jaccards, 50))
    p75_j = float(np.percentile(jaccards, 75))
    p90_j = float(np.percentile(jaccards, 90))
    p95_j = float(np.percentile(jaccards, 95))
    p50_s = int(np.percentile(shareds, 50))
    p75_s = int(np.percentile(shareds, 75))

    candidates = {
        'jaccard_candidates': [p75_j, p90_j, p95_j],
        'shared_count_candidates': [max(1, p50_s), p75_s]
    }
    log("Recommended Jaccard thresholds (try these): " + ", ".join(f"{x:.3f}" for x in candidates['jaccard_candidates']))
    log("Recommended shared-point counts (try these): " + ", ".join(str(x) for x in candidates['shared_count_candidates']))

    # quick mapping from threshold -> resulting number of merged groups (coarse)
    # (try few jaccard thresholds and compute resulting merged groups count)
    summary = {}
    for thr in candidates['jaccard_candidates']:
        groups = mask_merge_candidates_by_jaccard(point_to_masks, mask_to_points, jaccard_threshold=thr, min_shared=1, log=lambda *a, **k: None)
        # If we merge masks in each group into one mask and then recompute number of connected components by points->merged_mask,
        # the number of "objects" will be roughly: #components of merged bipartite graph. For speed, approximate by:
        merged_mask_count = len(mask_to_points) - sum(len(g)-1 for g in groups)  # naive approximate
        summary[f"jaccard_{thr:.3f}"] = {'groups': len(groups), 'approx_masks_after_merge': merged_mask_count}
    candidates['summary'] = summary
    return candidates


def plot_mask_instance_points(points3D, images, mask_instances, mask_dir, output_dir="debug_masks"):
    """
    For each mask instance in each frame:
        - Load the binary mask from mask_dir
        - Project all COLMAP 2D points on that frame
        - Highlight the points falling inside each mask instance
        - Save a debug visualization

    Args:
        points3D: COLMAP 3D points dictionary
        images: COLMAP image dictionary (contains xys, point2D_idxs)
        mask_instances: dict {frame_name: {instance_id: {...}}}
        mask_dir: directory containing files "*_instance_XXX.png"
        output_dir: directory where debug images will be saved
    """

    mask_dir = Path(mask_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # print(f"[DEBUG] Saving mask-instance visualizations in: {output_dir}")

    # ----------------------------------------------------
    # 1. Build mapping: frame_name → list of (pid, u, v)
    # ----------------------------------------------------
    frame_to_points = {}

    for pid, p in points3D.items():
        if not hasattr(p, 'image_ids'):
            continue

        for img_id, pt2d_idx in zip(p.image_ids, p.point2D_idxs):

            if img_id not in images:
                continue
            img = images[img_id]

            xys = np.array(getattr(img, "xys", []))
            if len(xys) == 0 or pt2d_idx >= len(xys):
                continue

            u, v = xys[pt2d_idx]
            frame_name = Path(img.name).stem

            frame_to_points.setdefault(frame_name, []).append((pid, int(u), int(v)))

    # ----------------------------------------------------
    # 2. Plot for each mask instance
    # ----------------------------------------------------
    for frame_name, inst_dict in mask_instances.items():

        for inst_id, props in inst_dict.items():

            # Reconstruct mask filename: {frame}_instance_{id}.png
            mask_path = mask_dir / f"{frame_name}_instance_{inst_id}.png"

            if not mask_path.exists():
                print(f"[WARN] Missing mask file: {mask_path}")
                continue

            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                print(f"[WARN] Failed to read mask: {mask_path}")
                continue

            h, w = mask.shape

            # green mask visualization
            vis = np.zeros((h, w, 3), dtype=np.uint8)
            vis[:, :, 1] = mask

            # Add COLMAP points for this frame
            pts = frame_to_points.get(frame_name, [])
            for pid, u, v in pts:
                if 0 <= u < w and 0 <= v < h:
                    if mask[v, u] > 0:
                        vis = cv2.circle(vis, (u, v), 3, (0, 0, 255), -1)

            # Save image
            out_path = output_dir / f"{frame_name}_instance_{inst_id}_debug.png"
            cv2.imwrite(str(out_path), vis)

            print(f"[OK] Saved: {out_path}")


def compute_and_propagate_masks(points3D, images, mask_dir, gs_cameras,
                                point_to_pixels=None, downsample=1, batch_size=100000,
                                device="cpu", save_path=None, log=print,
                                subset_frames=None):
    """
    Vectorized & OOM-safe mask mapping using precomputed point_to_pixels.

    Args:
        points3D: dict of COLMAP points
        images: dict of Image objects with `xys`
        mask_dir: directory containing instance masks
        point_to_pixels: optional precomputed mapping from build_point_pixel_mapping()
        downsample: unused (placeholder)
        batch_size: GPU batch size for mapping
        device: "cuda" or "cpu"
        save_path: optional JSON path to save masks/mapping
        subset_frames: optional subset of frames to process

    Returns:
        mask_instances, point_to_masks, mask_to_points, mapping
    """
    device = torch.device(device)
    mask_dir = Path(mask_dir)

    log(f"[DEBUG] Total COLMAP points: {len(points3D)}")
    log(f"[DEBUG] Total images with masks on disk: {len(set(Path(f).stem.split('_instance_')[0] for f in mask_dir.iterdir()))}")
    
    if save_path is None:
        json_path = mask_dir / "mask_mapping.json"
    else:
        json_path = Path(save_path)

    # ----------------- 0) Load cached JSON -----------------
    if json_path.exists():
        log(f"[INFO] Found cached JSON at {json_path}, loading...")
        with open(json_path, "r") as f:
            data = json.load(f)
        mask_instances = data.get("mask_instances", {})
        point_to_masks = {int(k): v for k, v in data.get("point_to_masks", {}).items()}
        mask_to_points = {}
        for k, v in data.get("mask_to_points", {}).items():
            f_name, m_id = k.rsplit("_", 1)
            mask_to_points[(f_name, int(m_id))] = v
        mapping = {}
        for k, v in data.get("mapping", {}).items():
            f_name, m_id = k.rsplit("_", 1)
            mapping[(f_name, int(m_id))] = v
        log("[INFO] Loaded cached masks and mapping.")
        return mask_instances, point_to_masks, mask_to_points, mapping

    # ----------------- 1) Load mask instances -----------------
    mask_instances = {}
    frame_masks = {}
    frame_mids = {}
    mask_shapes = {}

    exts = {"png","jpg","jpeg","bmp","tif","tiff"}
    exts |= {e.upper() for e in exts}
    mask_files = [f for f in mask_dir.iterdir() if "_instance_" in f.stem and f.suffix[1:] in exts]
    log(f"[DEBUG] Found {len(mask_files)} mask instances files on disk")
    if len(mask_files) == 0:
        log("[WARN] No mask files found")
        return {}, defaultdict(list), defaultdict(list), defaultdict(list)

    frame_to_list = defaultdict(list)
    for f in mask_files:
        try:
            frame_name, midx_str = f.stem.rsplit("_instance_", 1)
            midx = int(midx_str)
        except:
            continue
        frame_to_list[frame_name].append((midx, f))

    # Limit subset if needed
    frame_names = list(frame_to_list.keys())
    if subset_frames is not None:
        if isinstance(subset_frames, int):
            frame_names = random.sample(frame_names, min(subset_frames, len(frame_names)))
        elif isinstance(subset_frames, str):
            frame_names = [subset_frames]
        else:
            frame_names = [f for f in frame_names if f in subset_frames]
    frame_to_list = {k: v for k, v in frame_to_list.items() if k in frame_names}

    # Load masks into CPU tensors (bool)
    for frame_name, mids_paths in tqdm(frame_to_list.items(), desc="[mask_utils] Loading masks"):
        imgs = []
        mids_order = []
        H = W = None
        for midx, fpath in mids_paths:
            img = cv2.imread(str(fpath), cv2.IMREAD_GRAYSCALE)
            if img is None: 
                continue
            if H is None:
                H, W = img.shape
            else:
                img = cv2.resize(img, (W,H), interpolation=cv2.INTER_NEAREST)
            imgs.append(img)
            mids_order.append(midx)

            ys, xs = np.nonzero(img>0)
            if ys.size>0:
                mask_instances.setdefault(frame_name, {})[midx] = {
                    "centroid": (float(xs.mean()), float(ys.mean())),
                    "bbox": (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())),
                    "area": int(ys.size),
                    "mask_path": str(fpath)
                }

        if len(imgs)==0:
            continue

        mask_stack = torch.from_numpy(np.stack(imgs,0)>0)  # (M,H,W) bool
        frame_masks[frame_name] = mask_stack
        frame_mids[frame_name] = mids_order
        mask_shapes[frame_name] = (H,W)

    # ----------------- 2) Build point→image map using point_to_pixels -----------------
    if point_to_pixels is None:
        raise RuntimeError("point_to_pixels must be provided or precomputed in Scene")
    
    # ---- DEBUG: point_to_pixels coverage ----
    pts_with_obs = [pid for pid, obs in point_to_pixels.items() if len(obs) > 0]
    log(f"[DEBUG] Points with at least one image observation: {len(pts_with_obs)} / {len(points3D)}")

    pts_no_obs = len(points3D) - len(pts_with_obs)
    if pts_no_obs > 0:
        log(f"[WARN] Points with ZERO image observations: {pts_no_obs}")

    point_img_map = defaultdict(list)
    for pid, obs_list in point_to_pixels.items():
        for obs in obs_list:
            frame_name = Path(obs["image_name"]).stem
            if frame_name not in mask_shapes:  # skip frames without masks
                continue
            H,W = mask_shapes[frame_name]

            u_float, v_float = obs["xy"]
            u_scaled = int(round(u_float))
            v_scaled = int(round(v_float))
    
            point_img_map[frame_name].append((pid, u_scaled, v_scaled))

    # ----------------- DEBUG: check coordinate scaling -----------------
    total_proj = 0
    out_of_bounds = 0
    for frame_name, pts in point_img_map.items():
        H,W = mask_shapes[frame_name]
        for pid,u,v in pts:
            total_proj += 1
            if u<0 or u>=W or v<0 or v>=H:
                out_of_bounds += 1
    print(f"[DEBUG] Total projected points: {total_proj}, out-of-bounds: {out_of_bounds}")


    # ----------------- DEBUG: frame coverage -----------------
    total_proj = sum(len(v) for v in point_img_map.values())
    unique_proj_pids = {pid for pts in point_img_map.values() for pid,_,_ in pts}

    log(f"[DEBUG] Projections landing on frames with masks:")
    log(f"        Total projections: {total_proj}")
    log(f"        Unique points projected: {len(unique_proj_pids)} / {len(points3D)}")

    missing = set(points3D.keys()) - unique_proj_pids
    if len(missing) > 0:
        log(f"[WARN] Points NEVER projected into any masked frame: {len(missing)}")


    # ----------------- 3) Point → mask INSTANCE assignment (centroid-based) -----------------
    point_to_masks = defaultdict(list)
    mask_to_points = defaultdict(list)

    total_assignments = 0

    for frame_name, pts in tqdm(point_img_map.items(),
                                desc="[mask_utils] Instance assignment (centroid)"):

        if frame_name not in mask_instances:
            continue

        insts = mask_instances[frame_name]   # mid -> dict with centroid, bbox, area

        if len(insts) == 0:
            continue

        # Pre-pack centroids
        mids = list(insts.keys())
        centroids = np.array(
            [insts[mid]["centroid"] for mid in mids],
            dtype=np.float32
        )  # (M,2)

        for pid, u, v in pts:
            du = centroids[:, 0] - u
            dv = centroids[:, 1] - v
            d2 = du * du + dv * dv

            best_idx = int(np.argmin(d2))
            best_mid = mids[best_idx]

            point_to_masks[pid].append((frame_name, best_mid))
            mask_to_points[(frame_name, best_mid)].append(pid)

            total_assignments += 1

    log(f"[INFO] Total instance assignments: {total_assignments}")
    log(f"[INFO] Points mapped: {len(point_to_masks)}, masks: {len(mask_to_points)}")

    # ---- DEBUG: direct instance coverage ----
    pts_with_direct_mask = set(point_to_masks.keys())
    log(f"[DEBUG] Points with ≥1 DIRECT mask hit: "
        f"{len(pts_with_direct_mask)} / {len(points3D)}")

    missing_direct = set(points3D.keys()) - pts_with_direct_mask
    if len(missing_direct) > 0:
        log(f"[WARN] Points with NO direct mask hit: {len(missing_direct)}")


    # ----------------- 4) Propagate points to other frames (OOM-safe batches) -----------------
    pid_to_idx = {pid:i for i,pid in enumerate(points3D.keys())}
    pids = list(points3D.keys())
    X_all = torch.tensor(np.stack([np.append(points3D[pid].xyz,1.0) for pid in pids]),
                        dtype=torch.float32, device=device).T  # 4xN

    full_proj = {frame: (cam.projection_matrix.to(device) @ cam.world_view_transform.to(device))
                for frame, cam in gs_cameras.items()}

    # Frame-wise point indices
    frame_point_idx_map = defaultdict(list)
    for pid, obs_list in point_to_pixels.items():
        for obs in obs_list:
            frame = Path(obs["image_name"]).stem
            frame_point_idx_map[frame].append(pid_to_idx[pid])

    mapping = defaultdict(list)
    points_in_masks = defaultdict(list)

    # First propagation: map points into masks in the same frame
    for frame_name, mask_stack_cpu in tqdm(frame_masks.items(), desc="[propagate] mapping"):
        if frame_name not in full_proj: 
            continue

        mask_stack_gpu = mask_stack_cpu.to(device)  # (M,H,W)
        H, W = mask_shapes[frame_name]
        idx_list = frame_point_idx_map.get(frame_name, [])
        if len(idx_list) == 0: 
            continue

        idx_arr = np.array(idx_list, dtype=np.int64)
        for start in range(0, len(idx_arr), batch_size):
            chunk_idx = idx_arr[start:start+batch_size].tolist()
            X_sel = X_all[:, chunk_idx]
            proj = full_proj[frame_name] @ X_sel
            u = torch.clamp((proj[0]/proj[2]).long(), 0, W-1)
            v = torch.clamp((proj[1]/proj[2]).long(), 0, H-1)
            valid = proj[2] > 0
            inb = valid & (u>=0) & (u<W) & (v>=0) & (v<H)
            if not inb.any(): 
                continue

            u_valid = u[inb]
            v_valid = v[inb]
            sel_idx = np.array(chunk_idx)[inb.cpu().numpy()]  # indices into pids

            # hits: (num_masks, num_valid_points)
            hits = mask_stack_gpu[:, v_valid, u_valid]  
            M, B = hits.shape

            # safe iteration
            nz = torch.nonzero(hits, as_tuple=False)  # (mask_idx, point_idx)
            for mask_idx_tensor, local_idx_tensor in nz:
                mask_idx = int(mask_idx_tensor.item())
                local_idx = int(local_idx_tensor.item())
                pid_val = pids[sel_idx[local_idx]]
                mid_val = frame_mids[frame_name][mask_idx]
                points_in_masks[(frame_name, mid_val)].append(pid_val)

        del mask_stack_gpu
        torch.cuda.empty_cache()

    # ---- DEBUG: propagation stats ----
    propagated_pts = set()
    for (_, _), pids_list in points_in_masks.items():
        propagated_pts |= set(pids_list)

    log(f"[DEBUG] Points recovered via propagation: {len(propagated_pts)}")
    still_missing = set(points3D.keys()) - pts_with_direct_mask - propagated_pts
    log(f"[DEBUG] Points still without ANY mask after propagation: {len(still_missing)}")

    # ----------------- 5) Propagate to all destination frames -----------------
    src_items = list(points_in_masks.items())
    for (src_frame, src_mid), pid_list in tqdm(src_items, desc="[propagate] to dest frames"):
        if len(pid_list) == 0: 
            continue

        indices = np.array([pid_to_idx[pid] for pid in pid_list], dtype=np.int64)
        for start in range(0, len(indices), batch_size):
            chunk = indices[start:start+batch_size].tolist()
            X_sel = X_all[:, chunk]

            for dst_frame, mask_stack_cpu in frame_masks.items():
                if dst_frame not in full_proj: 
                    continue

                mask_stack_gpu = mask_stack_cpu.to(device)
                H, W = mask_shapes[dst_frame]

                proj = full_proj[dst_frame] @ X_sel
                u = torch.clamp((proj[0]/proj[2]).long(), 0, W-1)
                v = torch.clamp((proj[1]/proj[2]).long(), 0, H-1)
                valid = proj[2] > 0
                inb = valid & (u>=0) & (u<W) & (v>=0) & (v<H)
                if not inb.any(): 
                    del mask_stack_gpu
                    continue

                u_valid = u[inb]
                v_valid = v[inb]
                sel_idx = np.array(chunk)[inb.cpu().numpy()]
                hits = mask_stack_gpu[:, v_valid, u_valid]
                nz = torch.nonzero(hits, as_tuple=False)

                for mask_idx_tensor, local_idx_tensor in nz:
                    mask_idx = int(mask_idx_tensor.item())
                    local_idx = int(local_idx_tensor.item())
                    pid_val = pids[sel_idx[local_idx]]
                    mid_val = frame_mids[dst_frame][mask_idx]
                    mapping[(src_frame, src_mid)].append((dst_frame, mid_val))

                del mask_stack_gpu
                torch.cuda.empty_cache()

    # ----------------- 6) Save JSON -----------------
    os.makedirs(json_path.parent, exist_ok=True)
    serializable = {
        "mask_instances": mask_instances,
        "point_to_masks": {str(int(pid)): lst for pid, lst in point_to_masks.items()},
        "mask_to_points": {f"{f}_{int(m)}": pids for (f, m), pids in mask_to_points.items()},
        "mapping": {f"{f}_{int(m)}": dsts for (f, m), dsts in mapping.items()}
    }

    serializable = to_python(serializable)

    with open(json_path, "w") as f:
        json.dump(serializable, f, indent=2)

    log(f"[INFO] Saved JSON → {json_path}")

    return mask_instances, point_to_masks, mask_to_points, mapping
