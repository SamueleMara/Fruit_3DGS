#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#

# For inquiries contact  george.drettakis@inria.fr
#

import csv
import os
import sys
import uuid
from pathlib import Path
from random import randint

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
from tqdm import tqdm
from argparse import ArgumentParser, Namespace

from gaussian_renderer import render, network_gui
from scene import Scene, GaussianModel
from scene.dataset_readers import fetchPly
from utils.general_utils import safe_state, get_expon_lr_func
from utils.image_utils import psnr
from utils.graphics_utils import BasicPointCloud
from utils.read_write_model import read_model
from utils import depth_utils
from utils.loss_utils import l1_loss, ssim, binary_mask_render_loss
from utils.occlusion_layers import extract_layered_depths, render_semantic_peeled_depth, occlusion_order_loss
from utils.depth_order_loss import compute_depth_order_loss
from utils.depth_optimization import (
    compute_multiscale_depth_order_loss,
    compute_depth_gradient_smoothness,
    compute_depth_magnitude_consistency,
    compute_depth_range_loss,
    compute_adaptive_depth_weighting,
    DepthTrainingScheduler,
    estimate_depth_quality_score,
)
from utils.depth_anything_runtime import build_depth_anything_priors
from utils.depth_seed_runtime import (
    add_depth_seed_points_from_maps,
    export_depth_point_cloud_from_maps,
    load_colmap_model_with_fallback,
    scale_depth_maps_to_colmap,
)
from utils.mv_depth_runtime import build_multiview_depth_priors
from utils.feature_seed_runtime import add_feature_seed_points_from_maps
from utils.deep_colmap_runtime import run_full_colmap_style_deep_reconstruction
from arguments import ModelParams, PipelineParams, OptimizationParams

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except Exception:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except Exception:
    SPARSE_ADAM_AVAILABLE = False

try:
    from filter import filter_and_save
except Exception:
    filter_and_save = None


class TrainingCSVLogger:
    def __init__(self, path, fieldnames):
        self.path = path
        self.fieldnames = fieldnames
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._file = open(path, "a", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=fieldnames)
        if os.path.getsize(path) == 0:
            self._writer.writeheader()
            self._file.flush()

    def log(self, row):
        self._writer.writerow(row)
        self._file.flush()

    def close(self):
        self._file.close()


def plot_training_log(csv_path, out_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("[WARNING] matplotlib not available; skipping loss plot.")
        return

    if not os.path.exists(csv_path):
        print(f"[WARNING] Log file not found: {csv_path}; skipping loss plot.")
        return

    def _to_float(value):
        try:
            return float(value)
        except Exception:
            return np.nan

    rows = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    if not rows:
        print("[WARNING] Log file is empty; skipping loss plot.")
        return

    iters = np.array([_to_float(r.get("iter", "")) for r in rows])
    loss = np.array([_to_float(r.get("loss", "")) for r in rows])
    loss_ema = np.array([_to_float(r.get("loss_ema", "")) for r in rows])

    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.plot(iters, loss, label="loss", linewidth=1.0)
    if np.isfinite(loss_ema).any():
        ax.plot(iters, loss_ema, label="loss_ema", linewidth=1.2)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Total loss")
    ax.grid(True, alpha=0.2)
    ax.legend()

    fig.tight_layout()
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[OK] Loss plot saved to: {out_path}")


def get_depth_stats(scene):
    cameras = scene.getTrainCameras()
    total = len(cameras)
    loaded = 0
    reliable = 0
    for cam in cameras:
        if getattr(cam, "invdepthmap", None) is not None:
            loaded += 1
        if getattr(cam, "depth_reliable", False):
            reliable += 1
    return total, loaded, reliable


def compute_adaptive_depth_clipping(scene, percentile_min=5, percentile_max=95):
    """
    Compute depth clipping bounds from loaded depth priors using percentiles.
    
    Parameters:
    -----------
    scene : Scene
        Training scene with cameras
    percentile_min : int
        Lower percentile (e.g., 5 removes bottom 5%)
    percentile_max : int
        Upper percentile (e.g., 95 removes top 5%)
    
    Returns:
    --------
    tuple (depth_clip_min, depth_clip_max) or (0.0, 0.0) if no depths found
    """
    cameras = scene.getTrainCameras()
    all_depths = []
    
    for cam in cameras:
        invdepth_map = getattr(cam, "invdepthmap", None)
        if invdepth_map is not None:
            inv_values = invdepth_map.cpu().numpy().flatten()
            valid_mask = np.isfinite(inv_values) & (inv_values > 1e-8)
            if valid_mask.sum() > 0:
                # Cameras store inverse depth; convert to metric depth so clip bounds
                # are consistent with min/max depth filters used for seed generation.
                depth_values = 1.0 / inv_values[valid_mask]
                all_depths.extend(depth_values.tolist())
    
    if len(all_depths) == 0:
        print("[WARNING] No valid depth values found for adaptive clipping")
        return 0.0, 0.0
    
    all_depths = np.array(all_depths)
    clip_min = float(np.percentile(all_depths, percentile_min))
    clip_max = float(np.percentile(all_depths, percentile_max))
    
    n_depths = len(all_depths)
    print(f"[INFO] Adaptive depth clipping from {n_depths:,} values:")
    print(f"       Range: [{all_depths.min():.6f}, {all_depths.max():.6f}]")
    print(f"       Clipping: [{clip_min:.6f}, {clip_max:.6f}] (p{percentile_min}-p{percentile_max})")
    
    return clip_min, clip_max


def sanitize_depth_prior(depth_map: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """
    Make arbitrary monocular depth prior numerically safe:
    - removes NaN/Inf
    - shifts values positive when needed (order preserved)
    - clamps away from zero to keep inverse-depth conversion stable
    """
    depth = torch.nan_to_num(depth_map.float(), nan=0.0, posinf=0.0, neginf=0.0)
    valid = torch.isfinite(depth)
    if valid.any():
        min_val = depth[valid].min()
        if min_val <= 0:
            depth = depth - min_val + eps
    return depth.clamp_min(eps)


def normalize_depth_prior_per_frame(depth_map: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Normalize one depth map to [0,1] while keeping it numerically safe.
    """
    depth = sanitize_depth_prior(depth_map, eps=max(eps, 1e-6))
    dmin = depth.min()
    dmax = depth.max()
    span = dmax - dmin
    if torch.isfinite(span).item() and span.item() > eps:
        depth = (depth - dmin) / (span + eps)
    return sanitize_depth_prior(depth, eps=max(eps, 1e-6))


def _depth_to_numpy_2d(depth_map):
    if hasattr(depth_map, "detach"):
        depth_map = depth_map.detach().cpu().numpy()
    elif hasattr(depth_map, "cpu"):
        depth_map = depth_map.cpu().numpy()
    depth = np.asarray(depth_map, dtype=np.float32)
    if depth.ndim == 3:
        if depth.shape[0] == 1:
            depth = depth[0]
        else:
            depth = depth[..., 0]
    return np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)


def _depth_preview_u16(depth_map: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth_map, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 1e-8)
    out = np.zeros(depth.shape, dtype=np.uint16)
    if valid.sum() == 0:
        return out
    lo = float(np.percentile(depth[valid], 1.0))
    hi = float(np.percentile(depth[valid], 99.0))
    if hi <= lo:
        hi = lo + 1e-6
    norm = np.clip((depth - lo) / (hi - lo), 0.0, 1.0)
    out[valid] = (norm[valid] * 65535.0).astype(np.uint16)
    return out


def _depth_preview_u16_scaled(depth_map: np.ndarray) -> np.ndarray:
    """
    Robust visualization for COLMAP-scaled maps.

    Metric-depth maps can have a long far-depth tail (e.g., tiny disparity regions),
    which makes linear p1-p99 normalization look nearly black. Detect this case and
    switch to inverse-depth visualization for readability.
    """
    depth = np.asarray(depth_map, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 1e-8)
    if int(valid.sum()) == 0:
        return np.zeros(depth.shape, dtype=np.uint16)

    vals = depth[valid]
    p50 = float(np.percentile(vals, 50.0))
    p99 = float(np.percentile(vals, 99.0))
    if p50 > 0.0 and p99 > 25.0 * p50:
        depth_inv = np.zeros_like(depth, dtype=np.float32)
        depth_inv[valid] = 1.0 / np.clip(depth[valid], 1e-4, None)
        return _depth_preview_u16(depth_inv)

    return _depth_preview_u16(depth)


def _depth_norm_u8(depth_map: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth_map, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 1e-8)
    out = np.zeros(depth.shape, dtype=np.uint8)
    if valid.sum() == 0:
        return out
    lo = float(depth[valid].min())
    hi = float(depth[valid].max())
    if hi <= lo:
        hi = lo + 1e-6
    norm = np.clip((depth - lo) / (hi - lo), 0.0, 1.0)
    out[valid] = (norm[valid] * 255.0).astype(np.uint8)
    return out


def _resolve_image_path_for_stem(image_base_dir: Path, stem: str):
    exts = ("png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp")
    for ext in exts:
        candidate = image_base_dir / f"{stem}.{ext}"
        if candidate.exists():
            return candidate
    return None


def _auto_find_mask_dir(source_path: str):
    if not source_path:
        return None
    root = Path(source_path)
    if not root.is_dir():
        return None

    preferred = (
        "masks",
        "mask",
        "instance_masks",
        "segmentation_masks",
        "sam_masks",
    )
    for name in preferred:
        candidate = root / name
        if candidate.is_dir():
            return str(candidate)

    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        cname = child.name.lower()
        if ("mask" not in cname) and ("seg" not in cname):
            continue
        try:
            has_png = any(child.glob("*.png")) or any(child.glob("*/*.png"))
        except Exception:
            has_png = False
        if has_png:
            return str(child)
    return None


def export_depth_artifacts_from_maps(
    source_name: str,
    depth_maps: dict,
    depth_is_inverse: bool,
    dataset,
    output_root: str,
    pointcloud_mask_dir: str | None = None,
    pointcloud_stride: int = 4,
    depth_scale_mode: str = "median",
    depth_min_matches: int = 50,
    depth_scale_clamp: float = 0.0,
    depth_skip_unscaled: bool = False,
    depth_ransac: bool = False,
    depth_ransac_thresh: float = 0.1,
    depth_ransac_iters: int = 100,
    depth_ransac_min_inliers: int = 0,
    pointcloud_min_depth: float = 0.0,
    pointcloud_max_depth: float = 0.0,
    depth_align_mode: str = "scale",
    depth_max_reproj_error: float = 1.5,
):
    if not depth_maps:
        print(f"[WARNING] Depth artifact export skipped for '{source_name}': empty depth maps.")
        return

    src_name = str(source_name).strip().replace(" ", "_")
    source_dir = Path(output_root) / src_name
    npy_dir = source_dir / "maps_npy"
    preview_dir = source_dir / "maps_preview_png"
    debug_dir = source_dir / "debug_views"
    npy_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)
    image_base_dir = Path(dataset.source_path) / dataset.images

    saved = 0
    normalized_maps = {}
    for stem, depth in depth_maps.items():
        name = Path(str(stem)).stem
        depth_np = _depth_to_numpy_2d(depth).astype(np.float32)
        valid = np.isfinite(depth_np) & (depth_np > 1e-8)
        if saved < 3:
            if valid.any():
                vals = depth_np[valid]
                print(
                    f"[DepthDebug:{src_name}] {name}: "
                    f"shape={depth_np.shape}, dtype={depth_np.dtype}, "
                    f"min={vals.min():.6f}, max={vals.max():.6f}, mean={vals.mean():.6f}"
                )
            else:
                print(
                    f"[DepthDebug:{src_name}] {name}: "
                    f"shape={depth_np.shape}, dtype={depth_np.dtype}, no valid finite values"
                )
        np.save(npy_dir / f"{name}.npy", depth_np)
        imageio.imwrite(preview_dir / f"{name}.png", _depth_preview_u16(depth_np))
        normalized_maps[name] = depth_np

        rgb_path = _resolve_image_path_for_stem(image_base_dir, name)
        if rgb_path is not None:
            rgb = imageio.imread(rgb_path)
            if rgb.ndim == 2:
                rgb = np.stack([rgb, rgb, rgb], axis=-1)
            if rgb.shape[2] > 3:
                rgb = rgb[..., :3]
            if rgb.dtype != np.uint8:
                if np.max(rgb) <= 1.0:
                    rgb = (rgb * 255.0).clip(0, 255).astype(np.uint8)
                else:
                    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
            if rgb.shape[:2] != depth_np.shape[:2]:
                rgb = cv2.resize(rgb, (depth_np.shape[1], depth_np.shape[0]), interpolation=cv2.INTER_LINEAR)

            raw_u16 = _depth_preview_u16(depth_np)
            raw_u8 = (raw_u16 / 257.0).astype(np.uint8)
            raw_vis = cv2.applyColorMap(raw_u8, cv2.COLORMAP_INFERNO)
            raw_vis = cv2.cvtColor(raw_vis, cv2.COLOR_BGR2RGB)

            norm_u8 = _depth_norm_u8(depth_np)
            norm_vis = cv2.applyColorMap(norm_u8, cv2.COLORMAP_INFERNO)
            norm_vis = cv2.cvtColor(norm_vis, cv2.COLOR_BGR2RGB)

            overlay = cv2.addWeighted(rgb, 0.65, norm_vis, 0.35, 0.0)
            panel = np.concatenate([rgb, raw_vis, norm_vis, overlay], axis=1)
            imageio.imwrite(debug_dir / f"{name}.png", panel)

        saved += 1
    print(
        f"[INFO] Saved {saved} depth maps for '{src_name}' "
        f"(npy: {npy_dir}, preview: {preview_dir})"
    )
    print(
        f"[INFO] Saved depth debug composites for '{src_name}' "
        f"to {debug_dir} (RGB | raw-depth | normalized-depth | overlay)"
    )

    # Also save COLMAP-scaled metric-depth maps for easier inspection.
    try:
        scaled_by_stem, _ = scale_depth_maps_to_colmap(
            depth_maps_by_stem=normalized_maps,
            dataset=dataset,
            depth_is_inverse=depth_is_inverse,
            depth_scale_mode=depth_scale_mode,
            depth_min_matches=depth_min_matches,
            depth_scale_clamp=depth_scale_clamp,
            skip_unscaled=depth_skip_unscaled,
            depth_ransac=depth_ransac,
            depth_ransac_thresh=depth_ransac_thresh,
            depth_ransac_iters=depth_ransac_iters,
            depth_ransac_min_inliers=depth_ransac_min_inliers,
            depth_align_mode=depth_align_mode,
            depth_max_reproj_error=depth_max_reproj_error,
        )
        if scaled_by_stem:
            scaled_npy_dir = source_dir / "maps_colmap_scaled_npy"
            scaled_preview_dir = source_dir / "maps_colmap_scaled_preview_png"
            scaled_npy_dir.mkdir(parents=True, exist_ok=True)
            scaled_preview_dir.mkdir(parents=True, exist_ok=True)
            for stem, depth_scaled in scaled_by_stem.items():
                name = Path(str(stem)).stem
                depth_scaled = _depth_to_numpy_2d(depth_scaled).astype(np.float32)
                np.save(scaled_npy_dir / f"{name}.npy", depth_scaled)
                imageio.imwrite(scaled_preview_dir / f"{name}.png", _depth_preview_u16_scaled(depth_scaled))
            print(
                f"[INFO] Saved COLMAP-scaled depth maps for '{src_name}' "
                f"(npy: {scaled_npy_dir}, preview: {scaled_preview_dir})"
            )
    except Exception as exc:
        print(f"[WARNING] Could not export COLMAP-scaled depth maps for '{src_name}': {exc}")

    try:
        export_depth_point_cloud_from_maps(
            depth_maps=normalized_maps,
            dataset=dataset,
            output_dir=str(source_dir),
            mask_dir=pointcloud_mask_dir,
            depth_is_inverse=depth_is_inverse,
            depth_scale_mode=depth_scale_mode,
            depth_min_matches=depth_min_matches,
            depth_scale_clamp=depth_scale_clamp,
            skip_unscaled=depth_skip_unscaled,
            depth_ransac=depth_ransac,
            depth_ransac_thresh=depth_ransac_thresh,
            depth_ransac_iters=depth_ransac_iters,
            depth_ransac_min_inliers=depth_ransac_min_inliers,
            depth_align_mode=depth_align_mode,
            depth_max_reproj_error=depth_max_reproj_error,
            sample_stride=max(1, int(pointcloud_stride)),
            min_depth=pointcloud_min_depth,
            max_depth=pointcloud_max_depth,
            auto_clip_percentile_min=1.0,
            auto_clip_percentile_max=99.0,
            output_name=f"{src_name}_points.ply",
            log=print,
        )
    except Exception as exc:
        print(f"[WARNING] Failed depth point cloud export for '{src_name}': {exc}")


def add_depth_seed_points(
    scene,
    dataset,
    depth_seed_dir,
    depth_seed_mask_dir,
    depth_seed_suffix,
    depth_seed_stride,
    depth_seed_max_points,
    depth_seed_min_depth,
    depth_seed_max_depth,
    depth_seed_inverse,
    depth_seed_scale_mode,
    depth_seed_min_matches,
    depth_seed_random_seed,
    depth_seed_skip_unscaled,
    depth_seed_scale_clamp,
    depth_seed_ransac=False,
    depth_seed_ransac_thresh=0.1,
    depth_seed_ransac_iters=100,
    depth_seed_ransac_min_inliers=0,
    depth_clip_min=0.0,
    depth_clip_max=0.0,
):
    if depth_seed_dir is None:
        return False
    if not os.path.isdir(depth_seed_dir):
        print(f"[WARNING] depth_seed_dir not found: {depth_seed_dir}")
        return False

    if depth_seed_mask_dir is None:
        print("[WARNING] depth_seed_mask_dir not provided; depth seeds skipped.")
        return False
    if not os.path.isdir(depth_seed_mask_dir):
        print(f"[WARNING] depth_seed_mask_dir not found: {depth_seed_mask_dir}")
        return False

    _, cameras, images, points3D = load_colmap_model_with_fallback(
        dataset.source_path,
        log=print,
        context="DepthSeeds",
    )
    if cameras is None or images is None or points3D is None:
        print(f"[WARNING] Unable to read COLMAP model under: {dataset.source_path}")
        return False

    image_base_dir = os.path.join(dataset.source_path, dataset.images)
    if not os.path.isdir(image_base_dir):
        print(f"[WARNING] Image directory not found: {image_base_dir}")
        return False

    base_ply = os.path.join(scene.model_path, "input.ply")
    if not os.path.exists(base_ply):
        print(f"[WARNING] Base point cloud not found: {base_ply}")
        return False

    base_pcd = fetchPly(base_ply)
    if base_pcd is None:
        print("[WARNING] Failed to load base point cloud; depth seeds skipped.")
        return False

    # Apply clip bounds to seed point generation (use stricter bounds if clip params are provided)
    seed_min_depth = depth_seed_min_depth
    seed_max_depth = depth_seed_max_depth
    
    if depth_clip_min > 0:
        seed_min_depth = max(seed_min_depth, depth_clip_min) if seed_min_depth > 0 else depth_clip_min
    
    if depth_clip_max > 0:
        seed_max_depth = min(seed_max_depth, depth_clip_max) if seed_max_depth > 0 else depth_clip_max
    
    if depth_clip_min > 0 or depth_clip_max > 0:
        print(f"[INFO] Applying depth clipping to seed points: min_depth={seed_min_depth}, max_depth={seed_max_depth}")
    
    depth_xyz, depth_rgb, _ = depth_utils.generate_depth_seed_points(
        images=images,
        cameras=cameras,
        points3D=points3D,
        mask_dir=depth_seed_mask_dir,
        depth_dir=depth_seed_dir,
        image_base_dir=image_base_dir,
        depth_suffix=depth_seed_suffix,
        depth_is_inverse=depth_seed_inverse,
        depth_scale_mode=depth_seed_scale_mode,
        depth_min_matches=depth_seed_min_matches,
        mask_stride=depth_seed_stride,
        max_points_per_mask=depth_seed_max_points,
        min_depth=seed_min_depth,
        max_depth=seed_max_depth,
        random_seed=depth_seed_random_seed,
        skip_unscaled=depth_seed_skip_unscaled,
        depth_scale_clamp=depth_seed_scale_clamp,
        scale_ransac=depth_seed_ransac,
        scale_ransac_thresh=depth_seed_ransac_thresh,
        scale_ransac_iters=depth_seed_ransac_iters,
        scale_ransac_min_inliers=depth_seed_ransac_min_inliers,
        log=print,
    )


    # Save depth seed point cloud as PLY for comparison
    try:
        from pathlib import Path
        out_dir = Path(scene.model_path)
        out_dir.mkdir(parents=True, exist_ok=True)
        ply_path = out_dir / "depth_seeds.ply"
        with open(ply_path, "w") as f:
            f.write("ply\nformat ascii 1.0\n")
            f.write(f"element vertex {depth_xyz.shape[0]}\n")
            f.write("property float x\nproperty float y\nproperty float z\n")
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
            f.write("end_header\n")
            for pt, color in zip(depth_xyz, depth_rgb):
                r, g, b = (color * 255).astype(int)

                f.write(f"{pt[0]} {pt[1]} {pt[2]} {r} {g} {b}\n")
        print(f"[INFO] Saved depth seed point cloud to {ply_path}")
    except Exception as e:
        print(f"[WARNING] Could not save depth seed point cloud: {e}")

    depth_normals = np.zeros_like(depth_xyz, dtype=np.float32)

    combined_pcd = BasicPointCloud(
        points=np.concatenate([base_pcd.points, depth_xyz], axis=0),
        colors=np.concatenate([base_pcd.colors, depth_rgb], axis=0),
        normals=np.concatenate([base_pcd.normals, depth_normals], axis=0),
    )

    scene.gaussians.create_from_pcd(
        combined_pcd,
        scene.getTrainCameras(),
        scene.cameras_extent,
    )
    print(f"[INFO] Added {depth_xyz.shape[0]} depth seed points. Total points: {combined_pcd.points.shape[0]}")
    # ------------------------------------------------------------------
    # Improve influence of depth-seed points on initial geometry:
    # - Give seed points higher initial opacity so they contribute to renders.
    # - Reduce their spatial scale to make them more localized (sharper).
    # - If semantic mask exists, bias seed semantics to 1.0 so mask losses prefer them.
    # These changes help depth seeds guide reconstruction more strongly.
    # ------------------------------------------------------------------
    try:
        seed_count = int(depth_xyz.shape[0])
        total_count = int(combined_pcd.points.shape[0])
        seed_start = total_count - seed_count
        g = scene.gaussians
        if seed_count > 0 and hasattr(g, "_opacity") and g._opacity is not None:
            with torch.no_grad():
                # Increase seed opacity (target 0.4-0.6 instead of default ~0.1)
                target_opacity = 0.5
                inv_op = g.inverse_opacity_activation(torch.full((seed_count, 1), target_opacity, device=g._opacity.device))
                g._opacity.data[seed_start: total_count] = inv_op

                # Reduce seed spatial scale to make them sharper/localized
                try:
                    desired_scale = max(1e-6, 0.02 * float(scene.cameras_extent))
                    new_scale = torch.log(torch.full((seed_count, 3), desired_scale, device=g._scaling.device))
                    g._scaling.data[seed_start: total_count, :] = new_scale
                except Exception:
                    pass

                # Bias semantic mask for seed points if semantic mask exists
                if hasattr(g, "semantic_mask") and g.semantic_mask is not None:
                    try:
                        g.semantic_mask.data[seed_start: total_count] = 1.0
                    except Exception:
                        pass
        print(f"[INFO] Boosted depth-seed influence: opacity->{target_opacity}, scale~{desired_scale}")
    except Exception as e:
        print(f"[WARNING] Could not boost seed point influence: {e}")
    return True


def training(
    dataset,
    opt,
    pipe,
    testing_iterations,
    saving_iterations,
    checkpoint_iterations,
    checkpoint,
    debug_from,
    mask_dir=None,
    topk_contrib=8,
    semantic_threshold=0.3,
    render_semantic=False,
    use_contrib_arg=None,
    topk_depth_weight=0.0,
    topk_depth_sigma=1.0,
    topk_depth_sort=False,
    depth_seed_dir=None,
    depth_seed_mask_dir=None,
    depth_seed_suffix="",
    depth_seed_stride=4,
    depth_seed_max_points=20000,
    depth_seed_min_depth=0.0,
    depth_seed_max_depth=0.0,
    depth_seed_inverse=False,
    depth_seed_scale_mode="median",
    depth_seed_min_matches=50,
    depth_seed_random_seed=42,
    depth_seed_skip_unscaled=False,
    depth_seed_scale_clamp=0.0,
    depth_seed_ransac=False,
    depth_seed_ransac_thresh=0.1,
    depth_seed_ransac_iters=100,
    depth_seed_ransac_min_inliers=0,
    disable_depth_seeds=False,
    feature_seed_points=False,
    feature_seed_type="deep",
    feature_seed_max_points_per_image=1200,
    feature_seed_mask_dir=None,
    feature_seed_min_depth=0.0,
    feature_seed_max_depth=0.0,
    feature_seed_dedup_voxel=0.0,
    feature_seed_pair_window=1,
    feature_match_sim_thresh=0.75,
    feature_match_depth_consistency=0.15,
    feature_deep_device=None,
    feature_deep_backbone="resnet18",
    feature_export_colmap=False,
    feature_export_colmap_dir=None,
    deep_colmap_reconstruction=False,
    deep_colmap_output_dir=None,
    deep_colmap_extractor="disk",
    deep_colmap_matcher="disk+lightglue",
    deep_colmap_pair_mode="exhaustive",
    deep_colmap_pair_window=5,
    deep_colmap_max_image_size=1600,
    deep_colmap_overwrite=False,
    disable_depth_loss=False,
    use_colmap_points=True,
    depth_loss_pred="inverse",
    depth_loss_mask="none",
    depth_loss_mask_thresh=0.5,
    depth_clip_mode="static",
    depth_clip_min=0.0,
    depth_clip_max=0.0,
    depth_clip_percentile_min=5,
    depth_clip_percentile_max=95,
    depth_anything=False,
    depth_anything_variant="vitl14",
    depth_anything_input_size=518,
    depth_anything_repo="Depth-Anything",
    depth_anything_device=None,
    depth_anything_inverse=False,
    depth_anything_pointcloud_dir=None,
    depth_anything_pointcloud_stride=4,
    export_depth_artifacts=True,
    depth_artifacts_dir=None,
    depth_artifacts_pointcloud_stride=None,
    mv_depth=False,
    mv_depth_weights=None,
    mv_depth_num_views=3,
    mv_depth_min_depth=0.1,
    mv_depth_max_depth=100.0,
    mv_depth_downscale=1,
    mv_depth_cache_dir=None,
    mv_depth_device=None,
    mv_depth_fp16=False,
    mv_depth_vit_type="vits",
    mv_depth_num_scales=1,
    mv_depth_upsample_factor=4,
    mv_depth_lowest_feature_resolution=4,
    mv_depth_unet_channels=128,
    mv_depth_num_depth_candidates=128,
    mv_depth_grid_sample_disable_cudnn=False,
    mv_depth_inverse=False,
    mv_depth_use_seeds=False,
    mv_depth_override_depths=False,
    mask_start_mode="iter",
    depth_start_mode="iter",
    phase_check_interval=200,
    mask_plateau_rel=0.002,
    mask_plateau_patience=10,
    depth_plateau_rel=0.002,
    depth_plateau_patience=10,
    plot_losses=False,
    plot_losses_out=None,
    occ_smooth_weight=0.0,
    occ_order_weight=0.0,
    occ_prior_weight=0.0,
    occ_sem_thresh=0.5,
    occ_target_class=1,
    occ_front_class=2,
    occ_back_class=1,
    occ_geometric_mode="nearest",
    mask_depth_weight=0.0,
    mask_depth_sigma=0.05,
    log_file=None,
    log_interval=10,
    early_stop=False,
    early_stop_metric="photo",
    early_stop_mode="min",
    early_stop_patience=20,
    early_stop_rel=0.001,
    early_stop_min_iter=0,
    early_stop_interval=200,
    early_stop_save_final=True,
):
    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit("Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    if mask_dir is None and depth_seed_mask_dir is not None and os.path.isdir(depth_seed_mask_dir):
        mask_dir = depth_seed_mask_dir
        print(f"[INFO] Using depth_seed_mask_dir as mask_dir: {mask_dir}")
    elif mask_dir is None:
        auto_mask_dir = _auto_find_mask_dir(dataset.source_path)
        if auto_mask_dir is not None:
            mask_dir = auto_mask_dir
            print(f"[INFO] Auto-detected mask directory: {mask_dir}")

    use_mask_filtering = False
    if mask_dir is not None:
        if os.path.isdir(mask_dir):
            use_mask_filtering = True
            print(f"[INFO] Mask directory found. Filtering will run at final iteration: {mask_dir}")
        else:
            print(f"[WARNING] mask_dir provided but is not a valid folder: {mask_dir}")
            print("[WARNING] Final point cloud filtering will be skipped.")
    else:
        print(
            "[INFO] No mask_dir provided. Depth export will rely on COLMAP support filtering only. "
            "For best geometry pass --mask_dir /path/to/masks"
        )

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    log_path = log_file or os.path.join(dataset.model_path, "training_log.csv")
    log_fields = [
        "iter",
        "loss",
        "l1",
        "ssim",
        "mask_loss",
        "mask_weight",
        "depth_loss_raw",
        "depth_loss_main",
        "depth_loss",
        "depth_weight",
        "depth_cams_total",
        "depth_cams_loaded",
        "depth_cams_reliable",
        "depths_dir",
        "loss_ema",
        "depth_loss_ema",
        "mask_loss_ema",
        "lr_xyz",
        "iter_time_ms",
        "num_gaussians",
    ]
    log_writer = TrainingCSVLogger(log_path, log_fields)

    if deep_colmap_reconstruction:
        deep_out = deep_colmap_output_dir
        if deep_out is None:
            deep_out = os.path.join(dataset.model_path, "deep_colmap")
        print("[INFO] Running full COLMAP-style deep reconstruction before training.")
        deep_source = run_full_colmap_style_deep_reconstruction(
            source_path=dataset.source_path,
            images_subdir=dataset.images,
            output_root=deep_out,
            extractor=deep_colmap_extractor,
            matcher=deep_colmap_matcher,
            pair_mode=deep_colmap_pair_mode,
            pair_window=deep_colmap_pair_window,
            max_image_size=deep_colmap_max_image_size,
            overwrite=deep_colmap_overwrite,
        )
        if deep_source is not None:
            dataset.source_path = os.path.abspath(deep_source)
            print(f"[INFO] Switched dataset source_path to deep COLMAP model: {dataset.source_path}")
        else:
            print("[WARNING] Deep COLMAP reconstruction failed. Continuing with original source_path.")

    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians, mask_dir=mask_dir)
    
    # If not using COLMAP points, clear the initialized points
    if not use_colmap_points:
        print("[INFO] Skipping COLMAP initialization. Clearing loaded point cloud.")
        print("[INFO] Will use ONLY depth seed points for initialization.")
        # Reset gaussians to empty state
        num_points = gaussians.get_xyz.shape[0]
        print(f"[INFO] Clearing {num_points} COLMAP points from initialization.")
        gaussians._xyz = torch.empty((0, 3), dtype=torch.float32, device="cuda")
        gaussians._features_dc = torch.empty((0, 1, 3), dtype=torch.float32, device="cuda")
        gaussians._features_rest = torch.empty((0, 15, 3), dtype=torch.float32, device="cuda")
        gaussians._scaling = torch.empty((0, 3), dtype=torch.float32, device="cuda")
        gaussians._rotation = torch.empty((0, 4), dtype=torch.float32, device="cuda")
        gaussians._opacity = torch.empty((0, 1), dtype=torch.float32, device="cuda")
        gaussians._xyz_gradient_accum = torch.empty((0, 1), dtype=torch.float32, device="cuda")
        gaussians._denom = torch.empty((0, 1), dtype=torch.float32, device="cuda")
        gaussians.max_radii2D = torch.empty((0,), dtype=torch.int32, device="cuda")
        print("[INFO] Point cloud reset to empty state.")
    
    depth_total, depth_loaded, depth_reliable = get_depth_stats(scene)
    depths_dir = getattr(dataset, "depths", "")
    depth_artifacts_root = (
        depth_artifacts_dir
        or depth_anything_pointcloud_dir
        or os.path.join(dataset.model_path, "depth_artifacts")
    )
    depth_artifacts_stride_val = (
        depth_anything_pointcloud_stride
        if depth_artifacts_pointcloud_stride is None
        else depth_artifacts_pointcloud_stride
    )
    
    # Compute adaptive depth clipping if requested
    if depth_clip_mode == "auto":
        clip_min, clip_max = compute_adaptive_depth_clipping(
            scene, 
            percentile_min=depth_clip_percentile_min,
            percentile_max=depth_clip_percentile_max
        )
        depth_clip_min = clip_min
        depth_clip_max = clip_max
        print(f"[INFO] Using adaptive depth clipping: min={depth_clip_min:.6f}, max={depth_clip_max:.6f}")
    elif depth_clip_mode == "none":
        depth_clip_min = 0.0
        depth_clip_max = 0.0
        print("[INFO] Depth clipping disabled")
    else:
        if depth_clip_min > 0 or depth_clip_max > 0:
            print(f"[INFO] Using static depth clipping: min={depth_clip_min:.6f}, max={depth_clip_max:.6f}")
    
    if depth_anything:
        print("[INFO] Depth-Anything enabled; depth maps on disk are optional.")
    if mv_depth:
        print("[INFO] Multi-view depth enabled; depth maps on disk are optional.")
    if not depth_anything and not mv_depth:
        if depth_loaded == 0:
            print(
                "[WARNING] No depth maps loaded. Depth loss will be zero. "
                "Provide --depths/-d and ensure depth_params.json exists in sparse/0."
            )
        elif depth_reliable == 0:
            print(
                "[WARNING] Depth maps loaded but none are reliable. "
                "Check depth_params.json scaling and depth map format."
            )

    depth_anything_priors = None
    depth_anything_priors_raw = None
    depth_anything_cameras = None
    depth_anything_is_inverse = depth_anything_inverse
    if depth_anything:
        depth_anything_cameras = scene.getTrainCameras().copy()
        depth_anything_priors_raw = build_depth_anything_priors(
            cameras=depth_anything_cameras,
            variant=depth_anything_variant,
            input_size=depth_anything_input_size,
            device=depth_anything_device,
            repo_path=depth_anything_repo,
            normalize=False,
            debug=bool(export_depth_artifacts),
        )

        # Keep raw priors for geometry/export and normalized priors for depth-loss stability.
        depth_anything_priors_raw = [sanitize_depth_prior(d) for d in depth_anything_priors_raw]
        depth_anything_priors = [normalize_depth_prior_per_frame(d) for d in depth_anything_priors_raw]

        if export_depth_artifacts:
            # Use stricter defaults so noisy/unscaled frames do not dominate the export.
            pcl_min_depth = depth_seed_min_depth if depth_seed_min_depth > 0 else depth_clip_min
            pcl_max_depth = depth_seed_max_depth if depth_seed_max_depth > 0 else depth_clip_max
            pcl_skip_unscaled = True if depth_seed_scale_mode != "none" else False
            pcl_scale_mode = "none" if depth_seed_scale_mode == "none" else "global"
            pcl_align_mode = "scale" if pcl_scale_mode == "none" else "affine"
            export_depth_artifacts_from_maps(
                source_name="depth_anything",
                depth_maps={
                    Path(cam.image_name).stem: depth_anything_priors_raw[idx].squeeze().cpu().numpy()
                    for idx, cam in enumerate(depth_anything_cameras)
                },
                depth_is_inverse=depth_anything_is_inverse,
                dataset=dataset,
                output_root=depth_artifacts_root,
                pointcloud_mask_dir=(depth_seed_mask_dir if depth_seed_mask_dir is not None else mask_dir),
                pointcloud_stride=depth_artifacts_stride_val,
                depth_scale_mode=pcl_scale_mode,
                depth_min_matches=depth_seed_min_matches,
                depth_scale_clamp=depth_seed_scale_clamp,
                depth_skip_unscaled=pcl_skip_unscaled,
                depth_ransac=depth_seed_ransac,
                depth_ransac_thresh=depth_seed_ransac_thresh,
                depth_ransac_iters=depth_seed_ransac_iters,
                depth_ransac_min_inliers=depth_seed_ransac_min_inliers,
                pointcloud_min_depth=pcl_min_depth,
                pointcloud_max_depth=pcl_max_depth,
                depth_align_mode=pcl_align_mode,
                depth_max_reproj_error=1.5,
            )

        if depth_loaded == 0:
            for idx, cam in enumerate(depth_anything_cameras):
                depth = sanitize_depth_prior(depth_anything_priors_raw[idx].to(cam.data_device))
                if depth_anything_inverse:
                    invdepth = depth
                else:
                    invdepth = 1.0 / depth
                cam.invdepthmap = invdepth
                cam.depth_reliable = True
                cam.depth_mask = torch.ones_like(cam.alpha_mask)

    mv_depth_priors = None
    mv_depth_is_inverse = False
    if mv_depth:
        mv_depth_cameras = scene.getTrainCameras().copy()
        mv_depth_device = mv_depth_device or ("cuda" if torch.cuda.is_available() else "cpu")
        mv_depth_priors = build_multiview_depth_priors(
            cameras=mv_depth_cameras,
            num_views=mv_depth_num_views,
            weights_path=mv_depth_weights,
            device=mv_depth_device,
            min_depth=mv_depth_min_depth,
            max_depth=mv_depth_max_depth,
            downscale=mv_depth_downscale,
            cache_dir=mv_depth_cache_dir,
            use_fp16=mv_depth_fp16,
            vit_type=mv_depth_vit_type,
            num_scales=mv_depth_num_scales,
            upsample_factor=mv_depth_upsample_factor,
            lowest_feature_resolution=mv_depth_lowest_feature_resolution,
            depth_unet_channels=mv_depth_unet_channels,
            num_depth_candidates=mv_depth_num_depth_candidates,
            grid_sample_disable_cudnn=mv_depth_grid_sample_disable_cudnn,
        )
        mv_depth_is_inverse = mv_depth_inverse
        if mv_depth_priors is None or len(mv_depth_priors) == 0:
            print("[WARNING] Multi-view depth enabled but no priors computed.")
        elif mv_depth_override_depths or depth_loaded == 0:
            for idx, cam in enumerate(mv_depth_cameras):
                depth = sanitize_depth_prior(mv_depth_priors[idx].to(cam.data_device))
                if mv_depth_is_inverse:
                    invdepth = depth
                else:
                    invdepth = 1.0 / depth
                cam.invdepthmap = invdepth
                cam.depth_reliable = True
                cam.depth_mask = torch.ones_like(cam.alpha_mask)
            depth_total, depth_loaded, depth_reliable = get_depth_stats(scene)
        if export_depth_artifacts and mv_depth_priors is not None and len(mv_depth_priors) > 0:
            mv_scale_mode = "none" if depth_seed_scale_mode == "none" else "global"
            mv_align_mode = "scale" if mv_scale_mode == "none" else "affine"
            export_depth_artifacts_from_maps(
                source_name="mv_depth",
                depth_maps={
                    Path(cam.image_name).stem: mv_depth_priors[idx].squeeze().cpu().numpy()
                    for idx, cam in enumerate(mv_depth_cameras)
                },
                depth_is_inverse=mv_depth_is_inverse,
                dataset=dataset,
                output_root=depth_artifacts_root,
                pointcloud_mask_dir=(depth_seed_mask_dir if depth_seed_mask_dir is not None else mask_dir),
                pointcloud_stride=depth_artifacts_stride_val,
                depth_scale_mode=mv_scale_mode,
                depth_min_matches=depth_seed_min_matches,
                depth_scale_clamp=depth_seed_scale_clamp,
                depth_skip_unscaled=(depth_seed_scale_mode != "none"),
                depth_ransac=depth_seed_ransac,
                depth_ransac_thresh=depth_seed_ransac_thresh,
                depth_ransac_iters=depth_seed_ransac_iters,
                depth_ransac_min_inliers=depth_seed_ransac_min_inliers,
                pointcloud_min_depth=depth_seed_min_depth if depth_seed_min_depth > 0 else depth_clip_min,
                pointcloud_max_depth=depth_seed_max_depth if depth_seed_max_depth > 0 else depth_clip_max,
                depth_align_mode=mv_align_mode,
                depth_max_reproj_error=1.5,
            )

    # Recompute auto clipping after runtime priors are injected.
    if depth_clip_mode == "auto" and (depth_anything_priors is not None or mv_depth_priors is not None):
        clip_min, clip_max = compute_adaptive_depth_clipping(
            scene,
            percentile_min=depth_clip_percentile_min,
            percentile_max=depth_clip_percentile_max
        )
        depth_clip_min = clip_min
        depth_clip_max = clip_max
        if depth_clip_min > 0 or depth_clip_max > 0:
            print(f"[INFO] Updated adaptive depth clipping from runtime priors: min={depth_clip_min:.6f}, max={depth_clip_max:.6f}")

    if export_depth_artifacts and bool(depths_dir) and not depth_anything and not mv_depth:
        loaded_depth_maps = {}
        for cam in scene.getTrainCameras():
            inv = getattr(cam, "invdepthmap", None)
            if inv is None:
                continue
            inv_np = _depth_to_numpy_2d(inv)
            inv_np = np.clip(inv_np, 1e-3, None)
            depth_metric = (1.0 / inv_np).astype(np.float32)
            loaded_depth_maps[Path(cam.image_name).stem] = depth_metric
        if loaded_depth_maps:
            export_depth_artifacts_from_maps(
                source_name="scene_loaded_depth",
                depth_maps=loaded_depth_maps,
                depth_is_inverse=False,
                dataset=dataset,
                output_root=depth_artifacts_root,
                pointcloud_mask_dir=(depth_seed_mask_dir if depth_seed_mask_dir is not None else mask_dir),
                pointcloud_stride=depth_artifacts_stride_val,
                depth_scale_mode="none",
                depth_min_matches=depth_seed_min_matches,
                depth_scale_clamp=0.0,
                depth_skip_unscaled=False,
                depth_ransac=False,
                depth_ransac_thresh=0.1,
                depth_ransac_iters=100,
                depth_ransac_min_inliers=0,
                pointcloud_min_depth=depth_seed_min_depth if depth_seed_min_depth > 0 else depth_clip_min,
                pointcloud_max_depth=depth_seed_max_depth if depth_seed_max_depth > 0 else depth_clip_max,
                depth_align_mode="scale",
                depth_max_reproj_error=1.5,
            )

    if checkpoint:
        print("[INFO] Checkpoint provided; depth seed augmentation is skipped.")
    elif disable_depth_seeds:
        print("[INFO] Depth seed augmentation disabled.")
    else:
        if depth_seed_mask_dir is None:
            depth_seed_mask_dir = mask_dir

        base_points = int(gaussians.get_xyz.shape[0]) if gaussians.get_xyz is not None else 0
        allow_auto_depth_seeds = True
        if depth_seed_dir is None and use_colmap_points and base_points < max(100, int(depth_seed_min_matches)):
            allow_auto_depth_seeds = False
            print(
                "[WARNING] COLMAP initialization is too sparse for reliable depth-seed scaling "
                f"({base_points} points). Auto depth seeds are disabled to avoid noisy geometry."
            )

        use_mv_seeds = depth_seed_dir is None and mv_depth_priors is not None and mv_depth_use_seeds and allow_auto_depth_seeds
        use_anything_seeds = (
            depth_seed_dir is None
            and depth_anything_priors is not None
            and not use_mv_seeds
            and getattr(opt, "depth_anything_use_seeds", False)
            and allow_auto_depth_seeds
        )
        if depth_seed_dir is None and depth_anything_priors is not None and not use_mv_seeds and not use_anything_seeds:
            if allow_auto_depth_seeds:
                print("[INFO] Depth-Anything seed point augmentation is disabled (pass --depth_anything_use_seeds to enable).")
            else:
                print("[INFO] Depth-Anything seed point augmentation is skipped due to sparse COLMAP points.")
        if use_mv_seeds:
            depth_maps = {
                Path(cam.image_name).stem: mv_depth_priors[idx].squeeze().cpu().numpy()
                for idx, cam in enumerate(mv_depth_cameras)
            }
            seed_inverse = mv_depth_is_inverse
            add_depth_seed_points_from_maps(
                scene=scene,
                dataset=dataset,
                depth_maps=depth_maps,
                depth_seed_mask_dir=depth_seed_mask_dir,
                depth_seed_suffix=depth_seed_suffix,
                depth_seed_stride=depth_seed_stride,
                depth_seed_max_points=depth_seed_max_points,
                depth_seed_min_depth=depth_seed_min_depth,
                depth_seed_max_depth=depth_seed_max_depth,
                depth_seed_inverse=seed_inverse,
                depth_seed_scale_mode=depth_seed_scale_mode,
                depth_seed_min_matches=depth_seed_min_matches,
                depth_seed_random_seed=depth_seed_random_seed,
                depth_seed_skip_unscaled=depth_seed_skip_unscaled,
                depth_seed_scale_clamp=depth_seed_scale_clamp,
                depth_seed_ransac=depth_seed_ransac,
                depth_seed_ransac_thresh=depth_seed_ransac_thresh,
                depth_seed_ransac_iters=depth_seed_ransac_iters,
                depth_seed_ransac_min_inliers=depth_seed_ransac_min_inliers,
            )
        elif use_anything_seeds:
            depth_maps = {
                Path(cam.image_name).stem: depth_anything_priors_raw[idx].squeeze().cpu().numpy()
                for idx, cam in enumerate(depth_anything_cameras)
            }
            seed_inverse = depth_anything_inverse
            add_depth_seed_points_from_maps(
                scene=scene,
                dataset=dataset,
                depth_maps=depth_maps,
                depth_seed_mask_dir=depth_seed_mask_dir,
                depth_seed_suffix=depth_seed_suffix,
                depth_seed_stride=depth_seed_stride,
                depth_seed_max_points=depth_seed_max_points,
                depth_seed_min_depth=depth_seed_min_depth,
                depth_seed_max_depth=depth_seed_max_depth,
                depth_seed_inverse=seed_inverse,
                depth_seed_scale_mode=depth_seed_scale_mode,
                depth_seed_min_matches=depth_seed_min_matches,
                depth_seed_random_seed=depth_seed_random_seed,
                depth_seed_skip_unscaled=depth_seed_skip_unscaled,
                depth_seed_scale_clamp=depth_seed_scale_clamp,
                depth_seed_ransac=depth_seed_ransac,
                depth_seed_ransac_thresh=depth_seed_ransac_thresh,
                depth_seed_ransac_iters=depth_seed_ransac_iters,
                depth_seed_ransac_min_inliers=depth_seed_ransac_min_inliers,
            )
        else:
            add_depth_seed_points(
                scene=scene,
                dataset=dataset,
                depth_seed_dir=depth_seed_dir,
                depth_seed_mask_dir=depth_seed_mask_dir,
                depth_seed_suffix=depth_seed_suffix,
                depth_seed_stride=depth_seed_stride,
                depth_seed_max_points=depth_seed_max_points,
                depth_seed_min_depth=depth_seed_min_depth,
                depth_seed_max_depth=depth_seed_max_depth,
                depth_seed_inverse=depth_seed_inverse,
                depth_seed_scale_mode=depth_seed_scale_mode,
                depth_seed_min_matches=depth_seed_min_matches,
                depth_seed_random_seed=depth_seed_random_seed,
                depth_seed_skip_unscaled=depth_seed_skip_unscaled,
                depth_seed_scale_clamp=depth_seed_scale_clamp,
                depth_seed_ransac=depth_seed_ransac,
                depth_seed_ransac_thresh=depth_seed_ransac_thresh,
                depth_seed_ransac_iters=depth_seed_ransac_iters,
                depth_seed_ransac_min_inliers=depth_seed_ransac_min_inliers,
                depth_clip_min=depth_clip_min,
                depth_clip_max=depth_clip_max,
            )

    if (not checkpoint) and feature_seed_points:
        feature_depth_maps = {}
        feature_depth_is_inverse = False

        if mv_depth_priors is not None:
            feature_depth_maps = {
                Path(cam.image_name).stem: sanitize_depth_prior(mv_depth_priors[idx]).squeeze().cpu().numpy()
                for idx, cam in enumerate(mv_depth_cameras)
            }
            feature_depth_is_inverse = mv_depth_is_inverse
        elif depth_anything_priors is not None:
            feature_depth_maps = {
                Path(cam.image_name).stem: sanitize_depth_prior(depth_anything_priors_raw[idx]).squeeze().cpu().numpy()
                for idx, cam in enumerate(depth_anything_cameras)
            }
            feature_depth_is_inverse = depth_anything_inverse
        else:
            for cam in scene.getTrainCameras():
                inv = getattr(cam, "invdepthmap", None)
                if inv is None:
                    continue
                feature_depth_maps[Path(cam.image_name).stem] = inv.squeeze().detach().cpu().numpy()
            feature_depth_is_inverse = True

        if feature_depth_maps:
            add_feature_seed_points_from_maps(
                scene=scene,
                dataset=dataset,
                depth_maps=feature_depth_maps,
                depth_is_inverse=feature_depth_is_inverse,
                feature_mask_dir=(feature_seed_mask_dir or mask_dir),
                feature_type=feature_seed_type,
                feature_max_points_per_image=feature_seed_max_points_per_image,
                feature_min_depth=feature_seed_min_depth,
                feature_max_depth=feature_seed_max_depth,
                feature_dedup_voxel=feature_seed_dedup_voxel,
                feature_pair_window=feature_seed_pair_window,
                feature_match_sim_thresh=feature_match_sim_thresh,
                feature_match_depth_consistency=feature_match_depth_consistency,
                feature_deep_device=feature_deep_device,
                feature_deep_backbone=feature_deep_backbone,
                feature_export_colmap=feature_export_colmap,
                feature_export_colmap_dir=feature_export_colmap_dir,
            )
        else:
            print("[WARNING] Feature seeds requested but no usable depth priors were found.")

    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE
    depth_l1_steps = max(1, opt.iterations - int(getattr(opt, "depth_l1_start_iter", 0)))
    depth_l1_weight = get_expon_lr_func(
        opt.depth_l1_weight_init,
        opt.depth_l1_weight_final,
        max_steps=depth_l1_steps
    )

    # Honor explicit disable flag: force depth schedule to zero if requested
    if disable_depth_loss:
        depth_l1_weight = (lambda step: 0.0)
    
    # Initialize adaptive depth scheduler if enabled
    depth_scheduler = None
    if getattr(opt, "depth_loss_warmup_steps", 0) > 0 or getattr(opt, "depth_loss_plateau_boost", False):
        depth_scheduler = DepthTrainingScheduler(
            weight_init=opt.depth_l1_weight_init,
            weight_final=opt.depth_l1_weight_final,
            max_steps=opt.iterations,
            warmup_steps=getattr(opt, "depth_loss_warmup_steps", 0),
            plateau_patience=getattr(opt, "depth_loss_plateau_patience", 15),
            plateau_threshold=0.002,
        )
        print(f"[INFO] Depth scheduler enabled with warmup={getattr(opt, 'depth_loss_warmup_steps', 0)} steps")

    mask_steps = max(1, opt.iterations - int(getattr(opt, "mask_start_iter", 0)))
    mask_weight_schedule = get_expon_lr_func(
        opt.mask_weight_init,
        opt.mask_weight_final,
        max_steps=mask_steps
    )

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    ema_loss_for_log = 0.0
    ema_photo_loss = None
    ema_Ll1depth_for_log = 0.0
    ema_Ll1depth_raw_for_log = 0.0
    ema_mask_loss_for_log = None

    mask_phase_on = False
    depth_phase_on = False
    mask_phase_iter0 = None
    depth_phase_iter0 = None
    best_photo = float("inf")
    best_depth = float("inf")
    no_improve_photo = 0
    no_improve_depth = 0
    early_stop_best = None
    early_stop_no_improve = 0
    early_stop_triggered = False
    early_stop_iteration = None

    if early_stop:
        print(
            "[INFO] Early stopping enabled: "
            f"metric={early_stop_metric}, mode={early_stop_mode}, "
            f"patience={early_stop_patience}, rel={early_stop_rel}, "
            f"min_iter={early_stop_min_iter}, interval={early_stop_interval}"
        )
    # Occlusion loss enabled only if flag is set AND at least one weight > 0
    occ_enabled = args.train_occlusion and ((occ_smooth_weight > 0.0) or (occ_order_weight > 0.0) or (occ_prior_weight > 0.0))

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn is None:
            network_gui.try_connect()
        while network_gui.conn is not None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam is not None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifier=scaling_modifer, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception:
                network_gui.conn = None

        iter_start.record()
        lr_xyz = gaussians.update_learning_rate(iteration)

        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        viewpoint_idx = viewpoint_indices.pop(rand_idx)

        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        # Phase gating: iter-based or plateau-based
        if mask_start_mode == "iter":
            if (not mask_phase_on) and iteration >= getattr(opt, "mask_start_iter", 0):
                mask_phase_on = True
                mask_phase_iter0 = iteration
                print(f"[Phase] Mask loss enabled at iter {iteration} (iter mode).")
        if depth_start_mode == "iter":
            if (not depth_phase_on) and iteration >= getattr(opt, "depth_l1_start_iter", 0):
                depth_phase_on = True
                depth_phase_iter0 = iteration
                print(f"[Phase] Depth loss enabled at iter {iteration} (iter mode).")

        compute_mask_loss = viewpoint_cam.mask is not None and mask_phase_on
        occ_active = occ_enabled and mask_phase_on
        # Avoid expensive top-K contrib rasterization when mask loss weight is effectively zero.
        mask_weight_for_contrib = 0.0
        if compute_mask_loss:
            mask_step = iteration - (mask_phase_iter0 or iteration)
            mask_weight_for_contrib = float(mask_weight_schedule(mask_step))
        use_contrib = occ_active or (compute_mask_loss and mask_weight_for_contrib > 1e-12)
        # allow CLI override of contrib usage when explicitly set
        if use_contrib_arg is not None:
            use_contrib = use_contrib_arg

        render_pkg = render(
            viewpoint_cam,
            gaussians,
            pipe,
            bg,
            use_trained_exp=dataset.train_test_exp,
            separate_sh=SPARSE_ADAM_AVAILABLE,
            contrib=use_contrib,
            K=topk_contrib,
            render_semantic=render_semantic,
        )

        image = render_pkg.get("render", None)
        viewspace_point_tensor = render_pkg["viewspace_points"]
        visibility_filter = render_pkg.get("visibility_filter", None)
        radii = render_pkg.get("radii", None)

        contrib_indices = None
        contrib_opacities = None
        if use_contrib:
            contrib_indices = render_pkg.get("contrib_indices", None)
            contrib_opacities = render_pkg.get("contrib_opacities", None)

        alpha_mask = None
        if viewpoint_cam.alpha_mask is not None:
            alpha_mask = viewpoint_cam.alpha_mask.cuda()
            image *= alpha_mask

        gt_image = viewpoint_cam.original_image.cuda()
        photo_mask = None
        if opt.photo_loss_mask in ("semantic", "semantic_alpha") and viewpoint_cam.mask is not None:
            photo_mask = (viewpoint_cam.mask.to(image.device) > opt.photo_loss_mask_thresh).float()
        if opt.photo_loss_mask in ("alpha", "semantic_alpha") and alpha_mask is not None:
            photo_mask = alpha_mask if photo_mask is None else (photo_mask * alpha_mask)
        if photo_mask is not None:
            if photo_mask.ndim == 2:
                photo_mask = photo_mask.unsqueeze(0)
            image = image * photo_mask
            gt_image = gt_image * photo_mask
        Ll1 = l1_loss(image, gt_image)
        if FUSED_SSIM_AVAILABLE:
            ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        else:
            ssim_value = ssim(image, gt_image)

        photo_loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)
        loss = photo_loss

        mask_loss_val = None
        mask_weight_val = None
        if compute_mask_loss and contrib_indices is not None and contrib_opacities is not None:
            torch.cuda.empty_cache()
            gt_mask = viewpoint_cam.mask.cuda()
            mask_pos_weight = getattr(opt, "mask_loss_pos_weight", 1.0)
            if mask_pos_weight <= 0:
                pos = float(gt_mask.sum().item())
                neg = float(gt_mask.numel() - pos)
                if pos > 0:
                    mask_pos_weight = min(max(1.0, neg / (pos + 1e-6)), 50.0)
                else:
                    mask_pos_weight = 1.0
            contrib_opacities_weighted = contrib_opacities
            if mask_depth_weight > 0.0 and viewpoint_cam.depth_reliable:
                prior_depth = 1.0 / (viewpoint_cam.invdepthmap + 1e-6)
                if prior_depth.ndim == 3:
                    prior_depth = prior_depth[0]
                depth_mask = viewpoint_cam.depth_mask
                if depth_mask.ndim == 3:
                    depth_mask = depth_mask[0]
                layer_depths, _ = extract_layered_depths(
                    viewspace_point_tensor,
                    contrib_indices,
                    contrib_opacities
                )
                depth_diff = torch.abs(layer_depths - prior_depth.unsqueeze(-1))
                sigma = max(mask_depth_sigma, 1e-6)
                depth_weights = torch.exp(-depth_diff / sigma)
                depth_weights = depth_weights * depth_mask.unsqueeze(-1)
                contrib_opacities_weighted = contrib_opacities * (
                    (1.0 - mask_depth_weight) + mask_depth_weight * depth_weights
                )
            mask_loss = binary_mask_render_loss(
                gaussians.semantic_mask,
                contrib_indices,
                contrib_opacities_weighted,
                gt_mask,
                alpha_mask=viewpoint_cam.alpha_mask.cuda() if viewpoint_cam.alpha_mask is not None else None,
                pos_weight=mask_pos_weight
            )
            mask_step = iteration - (mask_phase_iter0 or iteration)
            mask_weight_val = mask_weight_schedule(mask_step)
            loss += mask_weight_val * mask_loss
            mask_loss_val = mask_loss.item()

        # Occlusion-aware losses (optional, require contrib buffers + --train_occlusion flag)
        occ_smooth_loss = None
        occ_order_loss = None
        occ_prior_loss = None
        if occ_active and contrib_indices is not None and contrib_opacities is not None and args.train_occlusion:
            sem_logits = gaussians.semantic_mask
            if sem_logits is not None:
                if sem_logits.dim() > 1:
                    sem_labels = torch.argmax(sem_logits, dim=1)
                else:
                    sem_labels = (torch.sigmoid(sem_logits) > occ_sem_thresh).long()
                sem_labels = sem_labels.to(viewspace_point_tensor.device)

                peeled_depth, peeled_mask = render_semantic_peeled_depth(
                    target_class_id=occ_target_class,
                    semantic_mask=sem_labels,
                    viewspace_points=viewspace_point_tensor,
                    contrib_indices=contrib_indices,
                    contrib_opacities=contrib_opacities,
                    geometric_prior_mode=occ_geometric_mode
                )

                if occ_smooth_weight > 0.0 and peeled_mask.sum() > 0:
                    d_dx = torch.abs(peeled_depth[:, 1:] - peeled_depth[:, :-1])
                    d_dy = torch.abs(peeled_depth[1:, :] - peeled_depth[:-1, :])
                    mask_dx = peeled_mask[:, 1:] * peeled_mask[:, :-1]
                    mask_dy = peeled_mask[1:, :] * peeled_mask[:-1, :]
                    occ_smooth_loss = (d_dx * mask_dx).mean() + (d_dy * mask_dy).mean()
                    loss += occ_smooth_weight * occ_smooth_loss

                if occ_prior_weight > 0.0 and viewpoint_cam.depth_reliable:
                    prior_depth = 1.0 / (viewpoint_cam.invdepthmap + 1e-6)
                    if prior_depth.ndim == 3:
                        prior_depth = prior_depth[0]
                    depth_mask = viewpoint_cam.depth_mask
                    if depth_mask.ndim == 3:
                        depth_mask = depth_mask[0]
                    valid_mask = (peeled_mask > 0) * (depth_mask > 0)
                    if valid_mask.sum() > 0:
                        occ_prior_loss = (torch.abs(peeled_depth - prior_depth) * valid_mask).mean()
                        loss += occ_prior_weight * occ_prior_loss

                if occ_order_weight > 0.0:
                    front_depth, front_mask = render_semantic_peeled_depth(
                        target_class_id=occ_front_class,
                        semantic_mask=sem_labels,
                        viewspace_points=viewspace_point_tensor,
                        contrib_indices=contrib_indices,
                        contrib_opacities=contrib_opacities,
                        geometric_prior_mode=occ_geometric_mode
                    )
                    back_depth, back_mask = render_semantic_peeled_depth(
                        target_class_id=occ_back_class,
                        semantic_mask=sem_labels,
                        viewspace_points=viewspace_point_tensor,
                        contrib_indices=contrib_indices,
                        contrib_opacities=contrib_opacities,
                        geometric_prior_mode=occ_geometric_mode
                    )
                    occ_order_loss = occlusion_order_loss(
                        front_depth=front_depth,
                        front_mask=front_mask,
                        back_depth=back_depth,
                        back_mask=back_mask
                    )
                    loss += occ_order_weight * occ_order_loss

        # Depth-order regularization (relative ordering, scale-invariant)
        Ll1depth_pure = 0.0
        Ll1depth_raw = 0.0
        Ll1depth_main = 0.0
        Ll1depth = 0.0
        depth_global_scale = float(getattr(opt, "depth_loss_scale", 1.0))
        if not depth_phase_on:
            depth_weight_val = 0.0
        else:
            depth_step = iteration - (depth_phase_iter0 or iteration)
            if depth_scheduler is not None:
                # Use adaptive scheduler
                depth_weight_val = depth_scheduler.get_weight(depth_step)
            else:
                # Use exponential decay
                depth_weight_val = depth_l1_weight(depth_step)
        
        use_mv_prior = mv_depth_priors is not None
        use_anything_prior = depth_anything_priors is not None
        if depth_weight_val > 0 and (viewpoint_cam.depth_reliable or use_mv_prior or use_anything_prior):
            pred_depth = render_pkg["depth"]
            pred_is_inverse = True
            if depth_loss_pred == "depth":
                pred_depth = 1.0 / pred_depth.clamp_min(1e-3)
                pred_is_inverse = False

            if use_mv_prior:
                prior_depth = mv_depth_priors[viewpoint_idx].to(pred_depth.device)
                prior_is_inverse = mv_depth_is_inverse
                if viewpoint_cam.alpha_mask is not None:
                    depth_mask = viewpoint_cam.alpha_mask.cuda()
                else:
                    depth_mask = torch.ones_like(pred_depth)
            elif use_anything_prior:
                prior_depth = depth_anything_priors[viewpoint_idx].to(pred_depth.device)
                prior_is_inverse = depth_anything_is_inverse
                if viewpoint_cam.alpha_mask is not None:
                    depth_mask = viewpoint_cam.alpha_mask.cuda()
                else:
                    depth_mask = torch.ones_like(pred_depth)
            else:
                prior_depth = viewpoint_cam.invdepthmap.cuda()
                prior_is_inverse = True
                depth_mask = viewpoint_cam.depth_mask.cuda()

            prior_depth = sanitize_depth_prior(prior_depth)
            if prior_is_inverse != pred_is_inverse:
                prior_depth = 1.0 / prior_depth.clamp_min(1e-3)

            # Clip depth values to avoid far away points
            # For inverse depth: clipping min depth means larger inverse values, max depth means smaller inverse values
            if depth_clip_max > 0:
                if pred_is_inverse:
                    pred_depth = torch.clamp(pred_depth, min=1.0 / (depth_clip_max + 1e-6))
                else:
                    pred_depth = torch.clamp(pred_depth, max=depth_clip_max)
                
                if prior_is_inverse:
                    prior_depth = torch.clamp(prior_depth, min=1.0 / (depth_clip_max + 1e-6))
                else:
                    prior_depth = torch.clamp(prior_depth, max=depth_clip_max)
            
            if depth_clip_min > 0:
                if pred_is_inverse:
                    pred_depth = torch.clamp(pred_depth, max=1.0 / (depth_clip_min + 1e-6))
                else:
                    pred_depth = torch.clamp(pred_depth, min=depth_clip_min)
                
                if prior_is_inverse:
                    prior_depth = torch.clamp(prior_depth, max=1.0 / (depth_clip_min + 1e-6))
                else:
                    prior_depth = torch.clamp(prior_depth, min=depth_clip_min)

            if prior_depth.shape != pred_depth.shape:
                prior_depth = torch.nn.functional.interpolate(
                    prior_depth.unsqueeze(0),
                    size=pred_depth.shape[-2:],
                    mode="bilinear",
                    align_corners=True,
                ).squeeze(0)
                depth_mask = torch.nn.functional.interpolate(
                    depth_mask.unsqueeze(0),
                    size=pred_depth.shape[-2:],
                    mode="nearest",
                ).squeeze(0)

            if depth_loss_mask in ("semantic", "semantic_alpha") and viewpoint_cam.mask is not None:
                semantic = (viewpoint_cam.mask.cuda() > depth_loss_mask_thresh).float()
                if semantic.sum() > 0:
                    depth_mask = depth_mask * semantic
            if depth_loss_mask in ("alpha", "semantic_alpha") and viewpoint_cam.alpha_mask is not None:
                depth_mask = depth_mask * viewpoint_cam.alpha_mask.cuda()

            if depth_mask.sum() <= 0:
                Ll1depth_main = 0.0
                Ll1depth = 0.0
                Ll1depth_pure = 0.0
            else:
                # ============================================================
                # ENHANCED DEPTH LOSS COMPUTATION
                # ============================================================
                
                # Main depth loss: order (may use multi-scale version)
                if getattr(opt, "depth_loss_multiscale", False):
                    scales = [int(s) for s in str(getattr(opt, "depth_loss_scales", "1,2,4")).split(",")]
                    Ll1depth_pure = compute_multiscale_depth_order_loss(
                        depth=pred_depth,
                        prior_depth=prior_depth,
                        scene_extent=scene.cameras_extent,
                        max_pixel_shift_ratio=0.05,
                        scales=scales,
                        normalize_loss=True,
                        log_space=True,
                        log_scale=20.0,
                        mask=depth_mask,
                    )
                else:
                    # Original single-scale order loss
                    Ll1depth_pure = compute_depth_order_loss(
                        depth=pred_depth,
                        prior_depth=prior_depth,
                        scene_extent=scene.cameras_extent,
                        max_pixel_shift_ratio=0.05,
                        normalize_loss=True,
                        log_space=True,
                        log_scale=20.0,
                        reduction="mean",
                        mask=depth_mask,
                    )
                
                # Start with weighted order loss
                Ll1depth_main = depth_global_scale * depth_weight_val * Ll1depth_pure
                loss += Ll1depth_main
                Ll1depth = Ll1depth_main
                Ll1depth_raw = Ll1depth_pure.item() if isinstance(Ll1depth_pure, torch.Tensor) else float(Ll1depth_pure)
                
                # ============================================================
                # DEPTH REGULARIZATION TERMS (Enhanced)
                # ============================================================
                
                # Smoothness regularization
                if getattr(opt, "depth_smooth_weight", 0.0) > 0:
                    # Optionally use image for edge awareness
                    edge_aware = None
                    if getattr(opt, "use_edge_aware_depth_weighting", False) and image is not None:
                        # Compute image gradients for edge detection
                        img_gray = image.mean(dim=0) if image.dim() == 3 else image
                        img_grad_x = torch.abs(img_gray[:, 1:] - img_gray[:, :-1])
                        img_grad_y = torch.abs(img_gray[1:, :] - img_gray[:-1, :])
                        # Bring both directional gradients to HxW before fusion.
                        img_grad_x = torch.nn.functional.pad(img_grad_x, (0, 1, 0, 0))
                        img_grad_y = torch.nn.functional.pad(img_grad_y, (0, 0, 0, 1))
                        img_grad = torch.maximum(img_grad_x, img_grad_y)
                        # Normalize
                        img_grad = img_grad / (img_grad.max() + 1e-6)
                        edge_aware = img_grad.clamp(0, 1)
                    
                    loss_smooth = compute_depth_gradient_smoothness(
                        pred_depth,
                        mask=depth_mask,
                        edge_aware_weight=edge_aware,
                        lambda_depth_grad=getattr(opt, "depth_smooth_weight", 0.0)
                    )
                    loss_smooth = depth_global_scale * loss_smooth
                    loss += loss_smooth
                    Ll1depth += loss_smooth
                    if tb_writer and iteration % 100 == 0:
                        tb_writer.add_scalar("depth/smooth_loss", loss_smooth.item(), iteration)
                
                # Magnitude consistency loss
                if getattr(opt, "depth_magnitude_weight", 0.0) > 0:
                    loss_mag = compute_depth_magnitude_consistency(
                        rendered_depth=pred_depth,
                        prior_depth=prior_depth,
                        scene_extent=scene.cameras_extent,
                        lambda_magnitude=getattr(opt, "depth_magnitude_weight", 0.0),
                        mask=depth_mask,
                    )
                    loss_mag = depth_global_scale * loss_mag
                    loss += loss_mag
                    Ll1depth += loss_mag
                    if tb_writer and iteration % 100 == 0:
                        tb_writer.add_scalar("depth/magnitude_loss", loss_mag.item(), iteration)
                
                # Range regularization
                if getattr(opt, "depth_range_weight", 0.0) > 0:
                    loss_range = compute_depth_range_loss(
                        pred_depth,
                        prior_depth,
                        lambda_range=getattr(opt, "depth_range_weight", 0.0),
                        mask=depth_mask,
                    )
                    loss_range = depth_global_scale * loss_range
                    loss += loss_range
                    Ll1depth += loss_range
                    if tb_writer and iteration % 100 == 0:
                        tb_writer.add_scalar("depth/range_loss", loss_range.item(), iteration)
                
                # ============================================================
                # ADAPTIVE DEPTH SCHEDULER UPDATE
                # ============================================================
                if depth_scheduler is not None and depth_phase_on:
                    if iteration % getattr(opt, "phase_check_interval", 200) == 0:
                        boosted = depth_scheduler.update_plateau_detection(Ll1depth_pure.item())
                        if boosted:
                            print(f"[Phase] Depth loss boosted at iter {iteration} (plateau detection)")
                
                Ll1depth_main = Ll1depth_main.item() if isinstance(Ll1depth_main, torch.Tensor) else Ll1depth_main
                Ll1depth = Ll1depth.item() if isinstance(Ll1depth, torch.Tensor) else Ll1depth
        else:
            Ll1depth_main = 0.0
            Ll1depth = 0.0

        loss.backward()
        iter_end.record()

        with torch.no_grad():
            if ema_photo_loss is None:
                ema_photo_loss = photo_loss.item()
            else:
                ema_photo_loss = 0.4 * photo_loss.item() + 0.6 * ema_photo_loss
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log
            ema_Ll1depth_raw_for_log = 0.4 * Ll1depth_raw + 0.6 * ema_Ll1depth_raw_for_log

            if mask_loss_val is not None:
                if ema_mask_loss_for_log is None:
                    ema_mask_loss_for_log = mask_loss_val
                else:
                    ema_mask_loss_for_log = 0.4 * mask_loss_val + 0.6 * ema_mask_loss_for_log

            if phase_check_interval and iteration % phase_check_interval == 0:
                # mask phase (auto)
                if mask_start_mode == "plateau" and (not mask_phase_on) and iteration >= getattr(opt, "mask_start_iter", 0):
                    if best_photo == float("inf") or ema_photo_loss < best_photo * (1.0 - mask_plateau_rel):
                        best_photo = ema_photo_loss
                        no_improve_photo = 0
                    else:
                        no_improve_photo += 1
                    if no_improve_photo >= mask_plateau_patience:
                        mask_phase_on = True
                        mask_phase_iter0 = iteration
                        print(f"[Phase] Mask loss enabled at iter {iteration} (plateau on photo).")

                # depth phase (auto)
                if depth_start_mode == "plateau" and (not depth_phase_on) and iteration >= getattr(opt, "depth_l1_start_iter", 0):
                    if mask_dir is not None and not mask_phase_on:
                        pass
                    else:
                        metric = ema_mask_loss_for_log if (mask_phase_on and ema_mask_loss_for_log is not None) else ema_photo_loss
                        if best_depth == float("inf") or metric < best_depth * (1.0 - depth_plateau_rel):
                            best_depth = metric
                            no_improve_depth = 0
                        else:
                            no_improve_depth += 1
                        if no_improve_depth >= depth_plateau_patience:
                            depth_phase_on = True
                            depth_phase_iter0 = iteration
                            print(f"[Phase] Depth loss enabled at iter {iteration} (plateau).")

            if early_stop and (not early_stop_triggered) and iteration >= int(early_stop_min_iter):
                check_every = max(1, int(early_stop_interval))
                if iteration % check_every == 0:
                    metric_val = None
                    if early_stop_metric == "photo":
                        metric_val = ema_photo_loss
                    elif early_stop_metric == "total":
                        metric_val = ema_loss_for_log
                    elif early_stop_metric == "mask":
                        if mask_phase_on and (ema_mask_loss_for_log is not None):
                            metric_val = ema_mask_loss_for_log
                    elif early_stop_metric == "depth":
                        if depth_phase_on:
                            metric_val = ema_Ll1depth_for_log

                    if metric_val is not None and np.isfinite(metric_val):
                        metric_val = float(metric_val)
                        if early_stop_best is None:
                            early_stop_best = metric_val
                            early_stop_no_improve = 0
                        else:
                            if early_stop_mode == "min":
                                improved = metric_val < early_stop_best * (1.0 - float(early_stop_rel))
                            else:
                                improved = metric_val > early_stop_best * (1.0 + float(early_stop_rel))

                            if improved:
                                early_stop_best = metric_val
                                early_stop_no_improve = 0
                            else:
                                early_stop_no_improve += 1

                        if early_stop_no_improve >= int(early_stop_patience):
                            early_stop_triggered = True
                            early_stop_iteration = iteration
                            print(
                                f"[EarlyStop] Triggered at iter {iteration}: "
                                f"metric={early_stop_metric}, best={early_stop_best:.6g}, current={metric_val:.6g}"
                            )

            is_final_iteration = (iteration == opt.iterations) or early_stop_triggered

            postfix_dict = {
                "Loss": f"{ema_loss_for_log:.7f}",
                "Depth Loss": f"{ema_Ll1depth_for_log:.3e}",
                "Depth Raw": f"{ema_Ll1depth_raw_for_log:.3e}",
            }
            if ema_mask_loss_for_log is not None:
                postfix_dict["Mask Loss"] = f"{ema_mask_loss_for_log:.7f}"
            if early_stop and early_stop_best is not None:
                postfix_dict["ES Best"] = f"{early_stop_best:.3e}"

            if iteration % 10 == 0:
                progress_bar.set_postfix(postfix_dict)
                progress_bar.update(10)
            if is_final_iteration:
                progress_bar.close()

            if tb_writer:
                if ema_mask_loss_for_log is not None:
                    tb_writer.add_scalar("train_loss_patches/mask_loss", ema_mask_loss_for_log, iteration)

            training_report(
                tb_writer, iteration, Ll1, loss, l1_loss,
                iter_start.elapsed_time(iter_end),
                testing_iterations, scene, render,
                (pipe, background, 1.0, SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp),
                dataset.train_test_exp
            )

            save_for_early_stop = early_stop_triggered and bool(early_stop_save_final)
            if iteration in saving_iterations or save_for_early_stop:
                print(f"\n[ITER {iteration}] Saving Gaussians")
                scene.save(iteration)

            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii)

                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            if iteration < opt.iterations and not early_stop_triggered:
                gaussians.exposure_optimizer.step()
                gaussians.exposure_optimizer.zero_grad(set_to_none=True)
                if use_sparse_adam:
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none=True)
                else:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none=True)

            if iteration in checkpoint_iterations:
                print(f"\n[ITER {iteration}] Saving Checkpoint")
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

            if use_mask_filtering and is_final_iteration:
                if filter_and_save is None:
                    print("[WARNING] filter_and_save not available; skipping point cloud filtering.")
                else:
                    print(f"\n[ITER {iteration}] Filtering final point cloud using masks...")
                    filter_and_save(scene, mask_dir=mask_dir, iteration=iteration, K=topk_contrib, semantic_threshold=semantic_threshold)

            if log_interval and (iteration % log_interval == 0 or iteration == 1 or is_final_iteration):
                log_writer.log({
                    "iter": iteration,
                    "loss": loss.item(),
                    "l1": Ll1.item(),
                    "ssim": ssim_value.item() if hasattr(ssim_value, "item") else float(ssim_value),
                    "mask_loss": mask_loss_val,
                    "mask_weight": mask_weight_val,
                    "depth_loss_raw": Ll1depth_raw,
                    "depth_loss_main": Ll1depth_main,
                    "depth_loss": Ll1depth,
                    "depth_weight": depth_weight_val,
                    "depth_cams_total": depth_total,
                    "depth_cams_loaded": depth_loaded,
                    "depth_cams_reliable": depth_reliable,
                    "depths_dir": depths_dir,
                    "loss_ema": ema_loss_for_log,
                    "depth_loss_ema": ema_Ll1depth_for_log,
                    "mask_loss_ema": ema_mask_loss_for_log,
                    "lr_xyz": lr_xyz,
                    "iter_time_ms": iter_start.elapsed_time(iter_end),
                    "num_gaussians": gaussians.get_xyz.shape[0],
                })

            if early_stop_triggered:
                print(f"[INFO] Early stopping finished training at iteration {early_stop_iteration}.")
                break

    log_writer.close()
    if plot_losses:
        plot_path = plot_losses_out or os.path.join(dataset.model_path, "loss_trends.png")
        plot_training_log(log_path, plot_path)


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv("OAR_JOB_ID"):
            unique_str = os.getenv("OAR_JOB_ID")
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), "w") as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene: Scene, renderFunc, renderArgs, train_test_exp):
    if tb_writer:
        tb_writer.add_scalar("train_loss_patches/l1_loss", Ll1.item(), iteration)
        tb_writer.add_scalar("train_loss_patches/total_loss", loss.item(), iteration)
        tb_writer.add_scalar("iter_time", elapsed, iteration)

    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = (
            {"name": "test", "cameras": scene.getTestCameras()},
            {"name": "train", "cameras": [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]},
        )

        for config in validation_configs:
            if config["cameras"] and len(config["cameras"]) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config["cameras"]):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if train_test_exp:
                        image = image[..., image.shape[-1] // 2:]
                        gt_image = gt_image[..., gt_image.shape[-1] // 2:]
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config["name"] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config["name"] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config["cameras"])
                l1_test /= len(config["cameras"])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config["name"], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config["name"] + "/loss_viewpoint - l1_loss", l1_test, iteration)
                    tb_writer.add_scalar(config["name"] + "/loss_viewpoint - psnr", psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar("total_points", scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = ArgumentParser(description="Semantic GS training with depth seed points")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6009)
    parser.add_argument("--debug_from", type=int, default=-1)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 20_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--disable_viewer", action="store_true", default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)

    parser.add_argument("--mask_dir", type=str, required=False, default=None,
                        help="Path to folder containing binary masks for filtering the final point cloud")
    parser.add_argument("--topk_contrib", type=int, default=8,
                        help="Number of top contributors per pixel for filtering")
    parser.add_argument("--sem_threshold", type=float, default=0.3,
                        help="Semantic threshold for final filtering")

    parser.add_argument("--depth_seed_dir", type=str, default=None,
                        help="Directory with monocular depth maps for depth seed points")
    parser.add_argument("--depth_seed_mask_dir", type=str, default=None,
                        help="Directory with instance masks (frame_instance_*.png) for depth seeding")
    parser.add_argument("--depth_seed_suffix", type=str, default="",
                        help="Optional suffix before extension for depth files")
    parser.add_argument("--depth_seed_stride", type=int, default=4,
                        help="Stride for sampling pixels inside instance masks")
    parser.add_argument("--depth_seed_max_points", type=int, default=20000,
                        help="Max depth seed points per instance mask (0 = unlimited)")
    parser.add_argument("--depth_seed_min_depth", type=float, default=0.0,
                        help="Minimum depth value to keep (0 = disabled)")
    parser.add_argument("--depth_seed_max_depth", type=float, default=0.0,
                        help="Maximum depth value to keep (0 = disabled)")
    parser.add_argument("--depth_seed_inverse", action="store_true",
                        help="Interpret depth maps as inverse depth")
    parser.add_argument("--depth_seed_scale_mode", type=str, default="median",
                        choices=["median", "none", "global"],
                        help="How to align monocular depth to COLMAP scale")
    parser.add_argument("--depth_seed_min_matches", type=int, default=50,
                        help="Min COLMAP point matches to compute depth scale")
    parser.add_argument("--depth_seed_random_seed", type=int, default=42,
                        help="Random seed for depth seed sampling")
    parser.add_argument("--depth_seed_skip_unscaled", action="store_true",
                        help="Skip frames with insufficient COLMAP matches for depth scale")
    parser.add_argument("--depth_seed_scale_clamp", type=float, default=0.0,
                        help="Clamp per-frame scale to global median * clamp (0 = disabled)")
    parser.add_argument("--depth_seed_ransac", action="store_true",
                        help="Use RANSAC to estimate per-frame depth scale")
    parser.add_argument("--depth_seed_ransac_thresh", type=float, default=0.1,
                        help="Relative inlier threshold for RANSAC (fraction of scale)")
    parser.add_argument("--depth_seed_ransac_iters", type=int, default=100,
                        help="RANSAC iterations for depth scale")
    parser.add_argument("--depth_seed_ransac_min_inliers", type=int, default=0,
                        help="Min inliers for RANSAC (0 = use depth_seed_min_matches)")
    parser.add_argument("--disable_depth_seeds", action="store_true",
                        help="Disable depth seed point augmentation")
    parser.add_argument("--feature_seed_points", action="store_true",
                        help="Augment initialization with feature-based 3D points")
    parser.add_argument("--feature_seed_type", type=str, default="deep",
                        choices=["deep", "orb", "gftt"],
                        help="Feature pipeline type (deep recommended)")
    parser.add_argument("--feature_seed_max_points_per_image", type=int, default=1200,
                        help="Max feature points to backproject per image")
    parser.add_argument("--feature_seed_mask_dir", type=str, default=None,
                        help="Optional mask dir to limit feature seeds (defaults to --mask_dir)")
    parser.add_argument("--feature_seed_min_depth", type=float, default=0.0,
                        help="Min depth for feature seed backprojection (0 disables)")
    parser.add_argument("--feature_seed_max_depth", type=float, default=0.0,
                        help="Max depth for feature seed backprojection (0 disables)")
    parser.add_argument("--feature_seed_dedup_voxel", type=float, default=0.0,
                        help="Voxel size for deduplicating feature seeds (0 disables)")
    parser.add_argument("--feature_seed_pair_window", type=int, default=1,
                        help="Match each image with the next N frames for deep feature tracking")
    parser.add_argument("--feature_match_sim_thresh", type=float, default=0.75,
                        help="Cosine similarity threshold for deep feature matches")
    parser.add_argument("--feature_match_depth_consistency", type=float, default=0.15,
                        help="Relative 3D consistency threshold for matched points")
    parser.add_argument("--feature_deep_device", type=str, default=None,
                        help="Device for deep feature extraction (cuda/cpu, default auto)")
    parser.add_argument("--feature_deep_backbone", type=str, default="resnet18",
                        choices=["resnet18", "resnet50"],
                        help="Deep backbone for feature extraction")
    parser.add_argument("--feature_export_colmap", action="store_true",
                        help="Export matched deep feature tracks as COLMAP text model")
    parser.add_argument("--feature_export_colmap_dir", type=str, default=None,
                        help="Output dir for COLMAP text export (default: <model_path>/feature_seed_colmap)")
    parser.add_argument("--deep_colmap_reconstruction", action="store_true",
                        help="Run full COLMAP-style SfM with deep features+matching before training")
    parser.add_argument("--deep_colmap_output_dir", type=str, default=None,
                        help="Work/output dir for deep COLMAP reconstruction")
    parser.add_argument("--deep_colmap_extractor", type=str, default="disk",
                        help="hloc deep feature extractor config name")
    parser.add_argument("--deep_colmap_matcher", type=str, default="disk+lightglue",
                        help="hloc deep matcher config name")
    parser.add_argument("--deep_colmap_pair_mode", type=str, default="exhaustive",
                        choices=["exhaustive", "window"],
                        help="Pair generation for deep COLMAP reconstruction")
    parser.add_argument("--deep_colmap_pair_window", type=int, default=5,
                        help="Neighbor window when pair mode is 'window'")
    parser.add_argument("--deep_colmap_max_image_size", type=int, default=1600,
                        help="Max image size for deep extractor preprocessing (0 disables resizing)")
    parser.add_argument("--deep_colmap_overwrite", action="store_true",
                        help="Overwrite deep COLMAP output directory if it already exists")
    parser.add_argument("--use_colmap_points", action="store_true", default=True,
                        help="Use COLMAP points for initialization (default: True)")
    parser.add_argument("--no_use_colmap_points", dest="use_colmap_points", action="store_false",
                        help="Skip COLMAP points, use only depth seed points")

    parser.add_argument("--depth_loss_pred", type=str, default="inverse",
                        choices=["inverse", "depth"],
                        help="Render depth interpretation for loss (inverse or depth)")
    parser.add_argument("--depth_loss_mask", type=str, default="none",
                        choices=["none", "semantic", "alpha", "semantic_alpha"],
                        help="Mask to apply to depth loss")
    parser.add_argument("--depth_loss_mask_thresh", type=float, default=0.5,
                        help="Threshold for semantic mask (0-1)")
    parser.add_argument("--depth_clip_mode", type=str, default="static",
                        choices=["static", "auto", "none"],
                        help="Depth clipping mode: static (fixed values), auto (from depth priors), none (disabled)")
    parser.add_argument("--depth_clip_min", type=float, default=0.0,
                        help="Min depth clip (used if mode=static)")
    parser.add_argument("--depth_clip_max", type=float, default=0.0,
                        help="Max depth clip (used if mode=static)")
    parser.add_argument("--depth_clip_percentile_min", type=int, default=5,
                        help="Lower percentile for auto clipping (1-49)")
    parser.add_argument("--depth_clip_percentile_max", type=int, default=95,
                        help="Upper percentile for auto clipping (51-99)")
    parser.add_argument("--depth_anything", action="store_true",
                        help="Compute Depth-Anything priors inside training")
    parser.add_argument("--depth_anything_variant", type=str, default="vitl14",
                        choices=["vits14", "vitb14", "vitl14"],
                        help="Depth-Anything variant (v1)")
    parser.add_argument("--depth_anything_input_size", type=int, default=518,
                        help="Depth-Anything input size (typically 518)")
    parser.add_argument("--depth_anything_repo", type=str, default="Depth-Anything",
                        help="Path to Depth-Anything repo (if not installed)")
    parser.add_argument("--depth_anything_device", type=str, default=None,
                        help="Device for Depth-Anything inference (cuda or cpu)")
    parser.add_argument("--depth_anything_inverse", action="store_true",
                        help="Treat Depth-Anything output as inverse depth")
    parser.add_argument("--depth_anything_pointcloud_dir", type=str, default=None,
                        help="Directory to save point cloud generated from Depth-Anything priors")
    parser.add_argument("--depth_anything_pointcloud_stride", type=int, default=4,
                        help="Pixel stride for Depth-Anything point cloud export (higher = fewer points)")
    parser.add_argument("--export_depth_artifacts", dest="export_depth_artifacts", action="store_true", default=True,
                        help="Export depth maps + depth point clouds when depth parameters are used (default: enabled)")
    parser.add_argument("--no_export_depth_artifacts", dest="export_depth_artifacts", action="store_false",
                        help="Disable exporting depth maps and depth point clouds")
    parser.add_argument("--depth_artifacts_dir", type=str, default=None,
                        help="Root directory for exported depth artifacts (default: <model_path>/depth_artifacts)")
    parser.add_argument("--depth_artifacts_pointcloud_stride", type=int, default=None,
                        help="Pixel stride for exported depth point clouds (default: --depth_anything_pointcloud_stride)")

    parser.add_argument("--mv_depth", action="store_true",
                        help="Compute multi-view depth priors using DepthSplat UniMatch")
    parser.add_argument("--mv_depth_weights", type=str, default=None,
                        help="Path to MultiViewUniMatch weights (DepthSplat pretrained)")
    parser.add_argument("--mv_depth_num_views", type=int, default=3,
                        help="Number of views per multi-view depth inference (incl. target)")
    parser.add_argument("--mv_depth_min_depth", type=float, default=0.1,
                        help="Minimum metric depth for multi-view module")
    parser.add_argument("--mv_depth_max_depth", type=float, default=100.0,
                        help="Maximum metric depth for multi-view module")
    parser.add_argument("--mv_depth_downscale", type=int, default=1,
                        help="Downscale factor for multi-view inference (1 = full res)")
    parser.add_argument("--mv_depth_cache_dir", type=str, default=None,
                        help="Optional cache dir for multi-view depth maps")
    parser.add_argument("--mv_depth_device", type=str, default=None,
                        help="Device for multi-view depth inference (cuda or cpu)")
    parser.add_argument("--mv_depth_fp16", action="store_true",
                        help="Enable fp16 autocast for multi-view depth inference")
    parser.add_argument("--mv_depth_vit_type", type=str, default="vits",
                        choices=["vits", "vitb", "vitl"],
                        help="ViT backbone type for MultiViewUniMatch")
    parser.add_argument("--mv_depth_num_scales", type=int, default=1,
                        help="Number of scales for MultiViewUniMatch")
    parser.add_argument("--mv_depth_upsample_factor", type=int, default=4,
                        help="Upsample factor for MultiViewUniMatch")
    parser.add_argument("--mv_depth_lowest_feature_resolution", type=int, default=4,
                        help="Lowest feature resolution for MultiViewUniMatch")
    parser.add_argument("--mv_depth_unet_channels", type=int, default=128,
                        help="UNet channels for MultiViewUniMatch")
    parser.add_argument("--mv_depth_num_depth_candidates", type=int, default=128,
                        help="Depth candidates for MultiViewUniMatch")
    parser.add_argument("--mv_depth_grid_sample_disable_cudnn", action="store_true",
                        help="Disable cuDNN for grid_sample in multi-view depth")
    parser.add_argument("--mv_depth_inverse", action="store_true",
                        help="Treat multi-view depth output as inverse depth")
    parser.add_argument("--mv_depth_use_seeds", action="store_true",
                        help="Use multi-view priors for depth seed points when available")
    parser.add_argument("--mv_depth_override_depths", action="store_true",
                        help="Override loaded depth maps with multi-view priors")

    parser.add_argument("--train_occlusion", action="store_true",
                        help="Enable occlusion-aware loss training (requires semantic masks + contrib)")
    parser.add_argument("--occ_smooth_weight", type=float, default=0.0,
                        help="Weight for occlusion depth smoothness loss")
    parser.add_argument("--occ_order_weight", type=float, default=0.0,
                        help="Weight for occlusion order loss")
    parser.add_argument("--occ_prior_weight", type=float, default=0.0,
                        help="Weight for occlusion depth-prior loss")
    parser.add_argument("--occ_sem_thresh", type=float, default=0.5,
                        help="Threshold for binary semantic labels (0-1)")
    parser.add_argument("--occ_target_class", type=int, default=1,
                        help="Target class ID for occlusion smoothing/prior")
    parser.add_argument("--occ_front_class", type=int, default=2,
                        help="Front class ID for occlusion order loss")
    parser.add_argument("--occ_back_class", type=int, default=1,
                        help="Back class ID for occlusion order loss")
    parser.add_argument("--occ_geometric_mode", type=str, default="nearest",
                        choices=["nearest", "weighted"],
                        help="Depth peeling mode for occlusion losses")
    if "--render_semantic" not in parser._option_string_actions:
        parser.add_argument("--render_semantic", action="store_true",
                            help="Enable semantic rendering outputs from the rasterizer")
    if "--use_contrib" not in parser._option_string_actions:
        parser.add_argument("--use_contrib", dest="use_contrib", action="store_true",
                            help="Force using contrib buffers from rasterizer (overrides auto)")
    if "--no_use_contrib" not in parser._option_string_actions:
        parser.add_argument("--no_use_contrib", dest="use_contrib", action="store_false",
                            help="Force NOT using contrib buffers from rasterizer")
    # default None: let training() decide if not specified
    if not hasattr(parser, "_use_contrib_default_set"):
        parser.set_defaults(use_contrib=None)
        parser._use_contrib_default_set = True
    if "--topk_depth_weight" not in parser._option_string_actions:
        parser.add_argument("--topk_depth_weight", type=float, default=0.0,
                            help="Optional weight for top-k depth heuristics (unused unless wired)")
    if "--topk_depth_sigma" not in parser._option_string_actions:
        parser.add_argument("--topk_depth_sigma", type=float, default=1.0,
                            help="Optional sigma for top-k depth heuristics")
    if "--topk_depth_sort" not in parser._option_string_actions:
        parser.add_argument("--topk_depth_sort", action="store_true",
                            help="Optional flag to enable top-k depth sorting behavior")
    parser.add_argument("--mask_depth_weight", type=float, default=0.0,
                        help="Blend depth-based weights into mask loss (0 disables)")
    parser.add_argument("--mask_depth_sigma", type=float, default=0.05,
                        help="Depth weighting sigma for mask loss (in depth units)")

    parser.add_argument("--mask_start_mode", type=str, default="iter",
                        choices=["iter", "plateau"],
                        help="How to enable mask loss (fixed iter or auto plateau)")
    parser.add_argument("--depth_start_mode", type=str, default="iter",
                        choices=["iter", "plateau"],
                        help="How to enable depth loss (fixed iter or auto plateau)")
    parser.add_argument("--phase_check_interval", type=int, default=200,
                        help="Iterations between plateau checks (auto mode)")
    parser.add_argument("--mask_plateau_rel", type=float, default=0.002,
                        help="Relative improvement needed to avoid mask plateau")
    parser.add_argument("--mask_plateau_patience", type=int, default=10,
                        help="Plateau checks without improvement before enabling mask")
    parser.add_argument("--depth_plateau_rel", type=float, default=0.002,
                        help="Relative improvement needed to avoid depth plateau")
    parser.add_argument("--depth_plateau_patience", type=int, default=10,
                        help="Plateau checks without improvement before enabling depth")

    parser.add_argument("--disable_depth_loss", action="store_true",
                        help="Disable depth L1/order loss regardless of weight settings")

    # ========== Enhanced Depth Optimization (NEW) ==========
    # Some environments register these arguments twice (via the ParamGroup
    # helpers). To be robust regardless of ordering, try to add them and
    # ignore argparse conflicts if they already exist.
    try:
        parser.add_argument("--depth_smooth_weight", type=float, default=0.0,
                            help="Smoothness regularization weight (0.05-0.1 recommended)")
        parser.add_argument("--depth_magnitude_weight", type=float, default=0.0,
                            help="Magnitude consistency loss weight (0.02-0.05 recommended)")
        parser.add_argument("--depth_range_weight", type=float, default=0.0,
                            help="Range regularization weight (0.01-0.02 recommended)")
        parser.add_argument("--depth_consistency_weight", type=float, default=0.0,
                            help="Consistency loss weight for multiple depth estimates")
        parser.add_argument("--use_edge_aware_depth_weighting", action="store_true",
                            help="Enable edge-aware adaptive weighting of depth loss")
        parser.add_argument("--depth_loss_multiscale", action="store_true",
                            help="Enable multi-scale depth order loss computation")
        parser.add_argument("--depth_loss_scales", type=str, default="1,2,4",
                            help="Scales for multi-scale depth loss (comma-separated, e.g. '1,2,4')")
        parser.add_argument("--depth_loss_warmup_steps", type=int, default=0,
                            help="Number of steps for depth loss warmup (0 = disabled, 2000 recommended)")
        parser.add_argument("--depth_loss_plateau_boost", action="store_true",
                            help="Enable adaptive depth loss boosting on plateau detection")
        parser.add_argument("--depth_loss_plateau_patience", type=int, default=15,
                            help="Plateau patience before depth loss boost")
        parser.add_argument("--depth_loss_plateau_boost_factor", type=float, default=2.0,
                            help="Multiplier for depth loss boost")
    except Exception:
        # argparse.ArgumentError or others: ignore and continue
        pass

    parser.add_argument("--plot_losses", action="store_true",
                        help="Save loss trend plot (PNG) at end of training")
    parser.add_argument("--plot_losses_out", type=str, default=None,
                        help="Output path for loss plot (default: <model_path>/loss_trends.png)")

    parser.add_argument("--log_file", type=str, default=None,
                        help="Path to CSV log file (default: <model_path>/training_log.csv)")
    parser.add_argument("--log_interval", type=int, default=10,
                        help="Log every N iterations to CSV")
    parser.add_argument("--early_stop", action="store_true",
                        help="Enable early stopping based on smoothed training metrics")
    parser.add_argument("--early_stop_metric", type=str, default="photo",
                        choices=["photo", "total", "mask", "depth"],
                        help="Metric monitored by early stopping")
    parser.add_argument("--early_stop_mode", type=str, default="min",
                        choices=["min", "max"],
                        help="Optimization direction for early-stop metric")
    parser.add_argument("--early_stop_patience", type=int, default=20,
                        help="Number of check intervals without improvement before stopping")
    parser.add_argument("--early_stop_rel", type=float, default=0.001,
                        help="Relative improvement threshold required to reset patience")
    parser.add_argument("--early_stop_min_iter", type=int, default=0,
                        help="Do not trigger early stopping before this iteration")
    parser.add_argument("--early_stop_interval", type=int, default=200,
                        help="Iterations between early-stop checks")
    parser.add_argument("--early_stop_save_final", dest="early_stop_save_final", action="store_true", default=True,
                        help="Save a final Gaussian snapshot when early stop triggers (default: enabled)")
    parser.add_argument("--no_early_stop_save_final", dest="early_stop_save_final", action="store_false",
                        help="Disable automatic final save when early stop triggers")

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    safe_state(args.quiet)

    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    training(
        lp.extract(args),
        op.extract(args),
        pp.extract(args),
        args.test_iterations,
        args.save_iterations,
        args.checkpoint_iterations,
        args.start_checkpoint,
        args.debug_from,
        mask_dir=args.mask_dir,
        topk_contrib=args.topk_contrib,
        semantic_threshold=args.sem_threshold,
        depth_seed_dir=args.depth_seed_dir,
        depth_seed_mask_dir=args.depth_seed_mask_dir,
        depth_seed_suffix=args.depth_seed_suffix,
        depth_seed_stride=args.depth_seed_stride,
        depth_seed_max_points=args.depth_seed_max_points,
        depth_seed_min_depth=args.depth_seed_min_depth,
        depth_seed_max_depth=args.depth_seed_max_depth,
        depth_seed_inverse=args.depth_seed_inverse,
        depth_seed_scale_mode=args.depth_seed_scale_mode,
        depth_seed_min_matches=args.depth_seed_min_matches,
        depth_seed_random_seed=args.depth_seed_random_seed,
        depth_seed_skip_unscaled=args.depth_seed_skip_unscaled,
        depth_seed_scale_clamp=args.depth_seed_scale_clamp,
        depth_seed_ransac=args.depth_seed_ransac,
        depth_seed_ransac_thresh=args.depth_seed_ransac_thresh,
        depth_seed_ransac_iters=args.depth_seed_ransac_iters,
        depth_seed_ransac_min_inliers=args.depth_seed_ransac_min_inliers,
        disable_depth_seeds=args.disable_depth_seeds,
        feature_seed_points=args.feature_seed_points,
        feature_seed_type=args.feature_seed_type,
        feature_seed_max_points_per_image=args.feature_seed_max_points_per_image,
        feature_seed_mask_dir=args.feature_seed_mask_dir,
        feature_seed_min_depth=args.feature_seed_min_depth,
        feature_seed_max_depth=args.feature_seed_max_depth,
        feature_seed_dedup_voxel=args.feature_seed_dedup_voxel,
        feature_seed_pair_window=args.feature_seed_pair_window,
        feature_match_sim_thresh=args.feature_match_sim_thresh,
        feature_match_depth_consistency=args.feature_match_depth_consistency,
        feature_deep_device=args.feature_deep_device,
        feature_deep_backbone=args.feature_deep_backbone,
        feature_export_colmap=args.feature_export_colmap,
        feature_export_colmap_dir=args.feature_export_colmap_dir,
        deep_colmap_reconstruction=args.deep_colmap_reconstruction,
        deep_colmap_output_dir=args.deep_colmap_output_dir,
        deep_colmap_extractor=args.deep_colmap_extractor,
        deep_colmap_matcher=args.deep_colmap_matcher,
        deep_colmap_pair_mode=args.deep_colmap_pair_mode,
        deep_colmap_pair_window=args.deep_colmap_pair_window,
        deep_colmap_max_image_size=args.deep_colmap_max_image_size,
        deep_colmap_overwrite=args.deep_colmap_overwrite,
        use_colmap_points=args.use_colmap_points,
        depth_loss_pred=args.depth_loss_pred,
        depth_loss_mask=args.depth_loss_mask,
        depth_loss_mask_thresh=args.depth_loss_mask_thresh,
        depth_clip_mode=args.depth_clip_mode,
        depth_clip_min=args.depth_clip_min,
        depth_clip_max=args.depth_clip_max,
        depth_clip_percentile_min=args.depth_clip_percentile_min,
        depth_clip_percentile_max=args.depth_clip_percentile_max,
        depth_anything=args.depth_anything,
        depth_anything_variant=args.depth_anything_variant,
        depth_anything_input_size=args.depth_anything_input_size,
        depth_anything_repo=args.depth_anything_repo,
        depth_anything_device=args.depth_anything_device,
        depth_anything_inverse=args.depth_anything_inverse,
        depth_anything_pointcloud_dir=args.depth_anything_pointcloud_dir,
        depth_anything_pointcloud_stride=args.depth_anything_pointcloud_stride,
        export_depth_artifacts=args.export_depth_artifacts,
        depth_artifacts_dir=args.depth_artifacts_dir,
        depth_artifacts_pointcloud_stride=args.depth_artifacts_pointcloud_stride,
        mv_depth=args.mv_depth,
        mv_depth_weights=args.mv_depth_weights,
        mv_depth_num_views=args.mv_depth_num_views,
        mv_depth_min_depth=args.mv_depth_min_depth,
        mv_depth_max_depth=args.mv_depth_max_depth,
        mv_depth_downscale=args.mv_depth_downscale,
        mv_depth_cache_dir=args.mv_depth_cache_dir,
        mv_depth_device=args.mv_depth_device,
        mv_depth_fp16=args.mv_depth_fp16,
        mv_depth_vit_type=args.mv_depth_vit_type,
        mv_depth_num_scales=args.mv_depth_num_scales,
        mv_depth_upsample_factor=args.mv_depth_upsample_factor,
        mv_depth_lowest_feature_resolution=args.mv_depth_lowest_feature_resolution,
        mv_depth_unet_channels=args.mv_depth_unet_channels,
        mv_depth_num_depth_candidates=args.mv_depth_num_depth_candidates,
        mv_depth_grid_sample_disable_cudnn=args.mv_depth_grid_sample_disable_cudnn,
        mv_depth_inverse=args.mv_depth_inverse,
        mv_depth_use_seeds=args.mv_depth_use_seeds,
        mv_depth_override_depths=args.mv_depth_override_depths,
        occ_smooth_weight=args.occ_smooth_weight,
        occ_order_weight=args.occ_order_weight,
        occ_prior_weight=args.occ_prior_weight,
        occ_sem_thresh=args.occ_sem_thresh,
        occ_target_class=args.occ_target_class,
        occ_front_class=args.occ_front_class,
        occ_back_class=args.occ_back_class,
        occ_geometric_mode=args.occ_geometric_mode,
        render_semantic=args.render_semantic,
        use_contrib_arg=args.use_contrib,
        topk_depth_weight=args.topk_depth_weight,
        topk_depth_sigma=args.topk_depth_sigma,
        topk_depth_sort=args.topk_depth_sort,
        mask_depth_weight=args.mask_depth_weight,
        mask_depth_sigma=args.mask_depth_sigma,
        mask_start_mode=args.mask_start_mode,
        depth_start_mode=args.depth_start_mode,
        phase_check_interval=args.phase_check_interval,
        mask_plateau_rel=args.mask_plateau_rel,
        mask_plateau_patience=args.mask_plateau_patience,
        depth_plateau_rel=args.depth_plateau_rel,
        depth_plateau_patience=args.depth_plateau_patience,
        plot_losses=args.plot_losses,
        plot_losses_out=args.plot_losses_out,
        disable_depth_loss=args.disable_depth_loss,
        log_file=args.log_file,
        log_interval=args.log_interval,
        early_stop=args.early_stop,
        early_stop_metric=args.early_stop_metric,
        early_stop_mode=args.early_stop_mode,
        early_stop_patience=args.early_stop_patience,
        early_stop_rel=args.early_stop_rel,
        early_stop_min_iter=args.early_stop_min_iter,
        early_stop_interval=args.early_stop_interval,
        early_stop_save_final=args.early_stop_save_final,
    )

    print("\nTraining complete.")
