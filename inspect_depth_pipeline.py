#!/usr/bin/env python3
"""Utility to inspect depth maps and the point cloud produced from them.

This script uses the same code paths that the project already has (see
``filter_train_depth.py``, ``export_depth_pointcloud.py`` and
``utils/depth_seed_runtime.py``) but adds a few additional sanity checks
and prints statistics that make it easy to spot the kind of "stacked
sheet" problem you described.

Usage example:

    python inspect_depth_pipeline.py \
        --source_path /path/to/depth_estimation/colmap \
        --depth_dir /path/to/depth_maps \
        --mask_dir /path/to/masks  # optional

The script will optionally call the exporter to create a temporary PLY;
it will then load the cloud and compute a number of diagnostics.  If
you already have a PLY from ``export_depth_pointcloud.py`` you can pass
``--ply existing.ply`` and the exporter step will be skipped.

"""

import argparse
import os
from pathlib import Path

import numpy as np

# reuse utilities from the repository
from utils.depth_seed_runtime import (
    load_colmap_model_with_fallback,
    export_depth_point_cloud_from_maps,
)
from utils.depth_utils import load_depth_map, DEFAULT_DEPTH_EXTS

try:
    # the scene module is already in the tree; needed to read PLY
    from scene.dataset_readers import fetchPly
except ImportError:
    fetchPly = None


def _normalize_exts(exts_arg):
    if exts_arg is None or exts_arg.strip() == "" or exts_arg.strip().lower() == "auto":
        return list(DEFAULT_DEPTH_EXTS)
    exts = []
    for token in exts_arg.split(","):
        token = token.strip()
        if not token:
            continue
        if not token.startswith("."):
            token = "." + token
        exts.append(token.lower())
    return exts if exts else list(DEFAULT_DEPTH_EXTS)


def load_depth_maps(depth_dir: str, suffix: str, exts: list):
    """Load every map in ``depth_dir`` matching the suffix and extensions."""
    depth_maps = {}
    root = Path(depth_dir)
    for ext in exts:
        for p in sorted(root.glob(f"*{suffix}{ext}")):
            stem = p.stem
            frame = stem[: -len(suffix)] if suffix and stem.endswith(suffix) else stem
            d = load_depth_map(p)
            if d is None:
                continue
            depth_maps[frame] = d.astype(np.float32)
    return depth_maps


def analyze_point_cloud(points: np.ndarray):
    """Print a few simple statistics and warn about degenerate geometry."""
    print(f"\npoint cloud: {points.shape[0]} points")
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    print("bounds: min", mins, "max", maxs)

    # axis variances
    stds = points.std(axis=0)
    print("std dev on axes (x,y,z):", stds)
    if stds[2] < 1e-3:
        print("[WARNING] very low variance along z axis, cloud may be collapsed")

    # duplicates (rounded to millimetre precision)
    rounded = np.round(points * 1000.0) / 1000.0
    uniq = np.unique(rounded, axis=0)
    dup_cnt = points.shape[0] - uniq.shape[0]
    if dup_cnt > 0:
        print(f"[WARNING] found {dup_cnt} duplicate points after rounding")

    # simple histogram of z values
    try:
        counts, edges = np.histogram(points[:, 2], bins=50)
        peaks = np.where(counts > 0.5 * counts.max())[0]
        if len(peaks) <= 2:
            print("[INFO] z histogram looks very peaky; depth maps might all have
the same value or be badly scaled.")
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(description="Inspect depth maps / exported point cloud")
    parser.add_argument("--source_path", required=True,
                        help="Root of COLMAP dataset (contains sparse/0 or cameras.txt)")
    parser.add_argument("--depth_dir", required=True,
                        help="Directory containing depth maps (.npy, .png, etc.)")
    parser.add_argument("--depth_suffix", default="",
                        help="Suffix that was appended to the frame names in the map files")
    parser.add_argument("--depth_exts", default="auto",
                        help="Comma-separated list of extensions, or 'auto'")
    parser.add_argument("--mask_dir", default=None,
                        help="Optional masks to limit projection (used by exporter)")
    parser.add_argument("--ply", default=None,
                        help="If you already have an exported point cloud, skip export and analyze this")
    parser.add_argument("--output_dir", default=".",
                        help="Where to write temporary ply if --ply is not provided")
    args = parser.parse_args()

    exts = _normalize_exts(args.depth_exts)
    depth_maps = load_depth_maps(args.depth_dir, args.depth_suffix, exts)
    if not depth_maps:
        print("no depth maps found; please check the directory and suffix")
        return
    print(f"loaded {len(depth_maps)} depth maps, extensions {exts}")

    # load colmap model so that export routines can use its intrinsics/poses
    model_dir, cameras, images, points3D = load_colmap_model_with_fallback(
        args.source_path, log=print, context="Inspect"
    )
    if model_dir is None:
        print("failed to read COLMAP model; aborting")
        return

    if args.ply is None:
        # call exporter to produce a ply we can examine
        print("exporting temporary point cloud...")
        ply_path = export_depth_point_cloud_from_maps(
            depth_maps=depth_maps,
            dataset=argparse.Namespace(source_path=args.source_path, images="images"),
            output_dir=args.output_dir,
            mask_dir=args.mask_dir,
            depth_is_inverse=False,
            depth_scale_mode="median",
            depth_min_matches=50,
            output_name="inspect_depth.ply",
            log=print,
        )
        if ply_path is None or not os.path.exists(ply_path):
            print("export failed, cannot analyze point cloud")
            return
    else:
        ply_path = args.ply
    print(f"analyzing point cloud: {ply_path}")

    if fetchPly is None:
        print("scene.dataset_readers.fetchPly not available; cannot load PLY")
        return
    pcd = fetchPly(ply_path)
    if pcd is None or pcd.points is None:
        print("failed to load point cloud")
        return
    analyze_point_cloud(pcd.points)

    # as an additional check, compute per-frame centroid drift
    centroids = []
    for img in images.values():
        frame = Path(img.name).stem
        depth = depth_maps.get(frame)
        if depth is None:
            continue
        cam = cameras[img.camera_id]
        h, w = depth.shape[:2]
        sx = w / float(cam.width)
        sy = h / float(cam.height)
        R = img.qvec2rotmat()
        t = img.tvec.reshape(3, 1)
        ys, xs = np.indices((h, w))
        zs = depth
        xs = (xs.astype(np.float32) / sx - cam.ppx) * zs / cam.fx
        ys = (ys.astype(np.float32) / sy - cam.ppy) * zs / cam.fy
        pts = np.stack([xs, ys, zs], axis=-1).reshape(-1, 3)
        # filter invalid
        valid = np.isfinite(pts).all(axis=1) & (pts[:, 2] > 0)
        pts = pts[valid]
        if pts.size == 0:
            continue
        # move to world coordinates
        pts_w = (R @ pts.T + t).T
        centroids.append(np.mean(pts_w, axis=0))
    if centroids:
        cent = np.vstack(centroids)
        spread = cent.std(axis=0)
        print("centroid spread across frames (x,y,z):", spread)
        if spread[2] < 1e-3:
            print("[WARNING] frame centroids all have nearly identical z values")

    print("inspection complete")


if __name__ == "__main__":
    main()
