#!/usr/bin/env python3
import argparse
import os
from argparse import Namespace
from pathlib import Path

import numpy as np

from utils.depth_seed_runtime import export_depth_point_cloud_from_maps
from utils.depth_utils import DEFAULT_DEPTH_EXTS, load_depth_map


def _normalize_exts(exts_arg: str | None) -> list[str]:
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


def load_depth_maps(depth_dir: str, depth_suffix: str, exts: list[str]) -> dict:
    depth_maps = {}
    depth_dir_path = Path(depth_dir)
    seen = set()

    for ext in exts:
        pattern = f"*{depth_suffix}{ext}" if depth_suffix else f"*{ext}"
        for p in sorted(depth_dir_path.glob(pattern)):
            if p in seen:
                continue
            seen.add(p)
            stem = p.stem
            frame = stem[:-len(depth_suffix)] if depth_suffix and stem.endswith(depth_suffix) else stem
            depth = load_depth_map(p)
            if depth is None:
                continue
            depth_maps[frame] = depth.astype(np.float32)
    return depth_maps


def main():
    parser = argparse.ArgumentParser(description="Export depth maps to a COLMAP-aligned point cloud")
    parser.add_argument("--source_path", type=str, required=True, help="Dataset root with sparse/0 and images/")
    parser.add_argument("--depth_dir", type=str, required=True, help="Directory with depth .npy files")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save output .ply")
    parser.add_argument("--images", type=str, default="images", help="Images subdir under source_path")
    parser.add_argument(
        "--mask_dir",
        type=str,
        default=None,
        help="Optional mask directory (instance masks or per-frame masks) to limit projection",
    )
    parser.add_argument("--depth_suffix", type=str, default="", help="Optional suffix before .npy")
    parser.add_argument(
        "--depth_exts",
        type=str,
        default="auto",
        help="Comma-separated depth extensions (e.g. .png,.tif,.exr) or 'auto'",
    )
    parser.add_argument("--output_name", type=str, default="depth_points.ply", help="Output .ply filename")
    parser.add_argument("--depth_is_inverse", action="store_true", help="Interpret map values as inverse depth")
    parser.add_argument("--depth_scale_mode", type=str, default="median", choices=["median", "none", "global"])
    parser.add_argument("--depth_align_mode", type=str, default="affine", choices=["scale", "affine"])
    parser.add_argument("--depth_min_matches", type=int, default=80)
    parser.add_argument("--depth_scale_clamp", type=float, default=0.0)
    parser.add_argument("--skip_unscaled", action="store_true")
    parser.add_argument("--depth_ransac", action="store_true")
    parser.add_argument("--depth_ransac_thresh", type=float, default=0.1)
    parser.add_argument("--depth_ransac_iters", type=int, default=100)
    parser.add_argument("--depth_ransac_min_inliers", type=int, default=0)
    parser.add_argument("--depth_max_reproj_error", type=float, default=1.5)
    parser.add_argument("--sample_stride", type=int, default=3)
    parser.add_argument("--min_depth", type=float, default=0.2)
    parser.add_argument("--max_depth", type=float, default=8.0)
    parser.add_argument("--auto_clip_percentile_min", type=float, default=1.0)
    parser.add_argument("--auto_clip_percentile_max", type=float, default=99.0)
    args = parser.parse_args()

    exts = _normalize_exts(args.depth_exts)
    depth_maps = load_depth_maps(args.depth_dir, args.depth_suffix, exts)
    if not depth_maps:
        raise RuntimeError(
            f"No depth files found in: {args.depth_dir} with extensions: {', '.join(exts)}"
        )

    dataset = Namespace(source_path=os.path.abspath(args.source_path), images=args.images)
    ply_path = export_depth_point_cloud_from_maps(
        depth_maps=depth_maps,
        dataset=dataset,
        output_dir=args.output_dir,
        mask_dir=args.mask_dir,
        depth_is_inverse=args.depth_is_inverse,
        depth_scale_mode=args.depth_scale_mode,
        depth_min_matches=args.depth_min_matches,
        depth_scale_clamp=args.depth_scale_clamp,
        skip_unscaled=args.skip_unscaled,
        depth_ransac=args.depth_ransac,
        depth_ransac_thresh=args.depth_ransac_thresh,
        depth_ransac_iters=args.depth_ransac_iters,
        depth_ransac_min_inliers=args.depth_ransac_min_inliers,
        depth_align_mode=args.depth_align_mode,
        depth_max_reproj_error=args.depth_max_reproj_error,
        sample_stride=args.sample_stride,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        auto_clip_percentile_min=args.auto_clip_percentile_min,
        auto_clip_percentile_max=args.auto_clip_percentile_max,
        output_name=args.output_name,
        log=print,
    )
    print(f"Saved: {ply_path}")


if __name__ == "__main__":
    main()
