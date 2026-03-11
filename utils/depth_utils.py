from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import cv2
import imageio

from utils.masks_utils import list_masks_for_frame


DEFAULT_DEPTH_EXTS = [".npy", ".npz", ".exr", ".tiff", ".tif", ".png", ".jpg", ".jpeg"]


def get_colmap_intrinsics(camera):
    """Return fx, fy, cx, cy for a COLMAP camera (ignoring distortion)."""
    model = camera.model
    params = camera.params
    if model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL",
                 "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE"):
        f, cx, cy = params[0], params[1], params[2]
        fx = fy = f
    elif model in ("PINHOLE", "OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV",
                   "FOV", "THIN_PRISM_FISHEYE"):
        fx, fy, cx, cy = params[0], params[1], params[2], params[3]
    else:
        raise ValueError(f"Unsupported camera model for intrinsics: {model}")
    return fx, fy, cx, cy


def undistort_colmap_pixels(u, v, camera, sx=1.0, sy=1.0, iterations=6):
    """
    Convert image pixels to undistorted normalized camera coordinates.

    Returns:
        x_u, y_u where X_cam = [x_u * depth, y_u * depth, depth].
    """
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    fx, fy, cx, cy = get_colmap_intrinsics(camera)
    fx *= float(sx)
    fy *= float(sy)
    cx *= float(sx)
    cy *= float(sy)

    x_d = (u - cx) / fx
    y_d = (v - cy) / fy

    model = camera.model
    if model in ("SIMPLE_RADIAL", "RADIAL", "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE"):
        params = np.asarray(camera.params, dtype=np.float64)
        k1 = float(params[3]) if params.size > 3 else 0.0
        k2 = float(params[4]) if model in ("RADIAL", "RADIAL_FISHEYE") and params.size > 4 else 0.0

        x_u = x_d.copy()
        y_u = y_d.copy()
        n_iter = max(1, int(iterations))
        for _ in range(n_iter):
            r2 = x_u * x_u + y_u * y_u
            radial = 1.0 + k1 * r2 + k2 * r2 * r2
            radial = np.where(np.abs(radial) > 1e-8, radial, 1e-8)
            x_u = x_d / radial
            y_u = y_d / radial
        return x_u.astype(np.float32), y_u.astype(np.float32)

    return x_d.astype(np.float32), y_d.astype(np.float32)


def resolve_depth_path(depth_dir, frame_name, suffix="", exts=None):
    depth_dir = Path(depth_dir)
    exts = exts or DEFAULT_DEPTH_EXTS
    base = f"{frame_name}{suffix}"
    for ext in exts:
        candidate = depth_dir / f"{base}{ext}"
        if candidate.exists():
            return candidate
    return None


def load_depth_map(depth_path):
    if depth_path is None:
        return None
    depth_path = Path(depth_path)
    if depth_path.suffix.lower() == ".npy":
        depth = np.load(depth_path).astype(np.float32)
    elif depth_path.suffix.lower() == ".npz":
        data = np.load(depth_path)
        key = list(data.keys())[0]
        depth = data[key].astype(np.float32)
    else:
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if depth is None:
            return None
        if depth.ndim == 3:
            depth = depth[..., 0]
        depth = depth.astype(np.float32)
    return depth


def _sample_mask_points(mask, stride, max_points, rng):
    if stride > 1:
        sub = mask[::stride, ::stride]
        ys, xs = np.nonzero(sub)
        ys = ys * stride
        xs = xs * stride
    else:
        ys, xs = np.nonzero(mask)

    if max_points and len(ys) > max_points:
        idx = rng.choice(len(ys), size=max_points, replace=False)
        ys = ys[idx]
        xs = xs[idx]
    return ys, xs


def _estimate_scale_ransac(ratios, thresh=0.1, iters=100, min_inliers=50, rng=None):
    if ratios.size == 0:
        return None, 0
    rng = rng or np.random
    best_inliers = 0
    best_scale = None
    for _ in range(iters):
        s = float(ratios[rng.randint(0, ratios.size)])
        if not np.isfinite(s) or s <= 0:
            continue
        tol = max(1e-8, thresh * s)
        inliers = np.abs(ratios - s) <= tol
        count = int(inliers.sum())
        if count > best_inliers:
            best_inliers = count
            best_scale = float(np.median(ratios[inliers]))
            if best_inliers >= min_inliers and best_inliers == ratios.size:
                break
    return best_scale, best_inliers


def _infer_depth_inverse_from_matches(pred_raw, z, min_abs_log_corr=0.25):
    """
    Infer whether depth_map values behave like inverse depth.

    We use log-space correlation between sparse COLMAP z and sampled map values:
    - positive correlation  -> metric depth-like
    - negative correlation  -> inverse depth-like
    """
    pred_raw = np.asarray(pred_raw, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    ok = np.isfinite(pred_raw) & np.isfinite(z) & (pred_raw > 1e-8) & (z > 1e-8)
    if int(ok.sum()) < 10:
        return None

    pred_log = np.log(pred_raw[ok])
    z_log = np.log(z[ok])
    if np.std(pred_log) < 1e-8 or np.std(z_log) < 1e-8:
        return None

    corr = float(np.corrcoef(pred_log, z_log)[0, 1])
    if not np.isfinite(corr) or abs(corr) < float(min_abs_log_corr):
        return None
    return bool(corr < 0.0)


def _compute_scale_factor(image, camera, points3D, depth_map,
                          sx, sy, depth_is_inverse=False,
                          scale_mode="median", min_matches=50,
                          return_matches=False, use_ransac=False,
                          ransac_thresh=0.1, ransac_iters=100,
                          ransac_min_inliers=0, rng=None,
                          auto_infer_inverse=False,
                          return_inverse=False):
    if scale_mode == "none":
        if return_matches and return_inverse:
            return 1.0, 0, bool(depth_is_inverse)
        if return_matches:
            return 1.0, 0
        if return_inverse:
            return 1.0, bool(depth_is_inverse)
        return 1.0

    pids = image.point3D_ids
    valid = pids != -1
    if not np.any(valid):
        if return_matches and return_inverse:
            return 1.0, 0, bool(depth_is_inverse)
        if return_matches:
            return 1.0, 0
        if return_inverse:
            return 1.0, bool(depth_is_inverse)
        return 1.0

    pids_valid = pids[valid]
    xys_valid = image.xys[valid]

    # Filter out point IDs missing from points3D
    keep = [int(pid) in points3D for pid in pids_valid]
    if not any(keep):
        if return_matches and return_inverse:
            return 1.0, 0, bool(depth_is_inverse)
        if return_matches:
            return 1.0, 0
        if return_inverse:
            return 1.0, bool(depth_is_inverse)
        return 1.0
    pids_valid = pids_valid[keep]
    xys_valid = xys_valid[keep]
    u = np.round(xys_valid[:, 0] * sx).astype(np.int32)
    v = np.round(xys_valid[:, 1] * sy).astype(np.int32)

    h, w = depth_map.shape[:2]
    u = np.clip(u, 0, w - 1)
    v = np.clip(v, 0, h - 1)

    pred_raw = depth_map[v, u].astype(np.float32)

    R = image.qvec2rotmat()
    t = image.tvec.reshape(3, 1)
    xyz = np.array([points3D[int(pid)].xyz for pid in pids_valid], dtype=np.float32)
    cam_xyz = (R @ xyz.T) + t
    z = cam_xyz[2, :].reshape(-1)

    effective_inverse = bool(depth_is_inverse)
    if auto_infer_inverse:
        inferred = _infer_depth_inverse_from_matches(pred_raw, z)
        if inferred is not None:
            effective_inverse = bool(inferred)

    pred = pred_raw
    if effective_inverse:
        pos = pred[pred > 0]
        floor = max(float(pos.min()) * 0.1, 1e-2) if pos.size > 0 else 1e-2
        pred = 1.0 / np.clip(pred, floor, None)

    ok = np.isfinite(pred) & np.isfinite(z) & (pred > 1e-6) & (z > 1e-6)
    ok_count_raw = int(ok.sum())
    if ok_count_raw < min_matches:
        if return_matches and return_inverse:
            return 1.0, ok_count_raw, effective_inverse
        if return_matches:
            return 1.0, ok_count_raw
        if return_inverse:
            return 1.0, effective_inverse
        return 1.0

    pred_ok = pred[ok].astype(np.float32, copy=False)
    z_ok = z[ok].astype(np.float32, copy=False)

    # Robustly trim sparse match outliers (near-zero inverse depth and very far z tails).
    if pred_ok.size >= max(16, min_matches):
        p_lo, p_hi = np.percentile(pred_ok, [2.0, 98.0])
        z_lo, z_hi = np.percentile(z_ok, [2.0, 98.0])
        core = (
            (pred_ok >= float(p_lo))
            & (pred_ok <= float(p_hi))
            & (z_ok >= float(z_lo))
            & (z_ok <= float(z_hi))
        )
        if int(core.sum()) >= max(8, min_matches // 2):
            pred_ok = pred_ok[core]
            z_ok = z_ok[core]

    ok_count = int(pred_ok.size)
    if ok_count < min_matches:
        if return_matches and return_inverse:
            return 1.0, ok_count, effective_inverse
        if return_matches:
            return 1.0, ok_count
        if return_inverse:
            return 1.0, effective_inverse
        return 1.0

    ratios = z_ok / np.clip(pred_ok, 1e-6, None)
    if ratios.size >= max(16, min_matches):
        r_lo, r_hi = np.percentile(ratios, [5.0, 95.0])
        ratio_core = (ratios >= float(r_lo)) & (ratios <= float(r_hi))
        if int(ratio_core.sum()) >= max(8, min_matches // 2):
            ratios = ratios[ratio_core]

    if use_ransac:
        min_inliers = min_matches if ransac_min_inliers <= 0 else ransac_min_inliers
        scale, inliers = _estimate_scale_ransac(
            ratios,
            thresh=ransac_thresh,
            iters=ransac_iters,
            min_inliers=min_inliers,
            rng=rng
        )
        if scale is None or inliers < min_inliers:
            scale = float(np.median(ratios))
    else:
        scale = float(np.median(ratios))
    if not np.isfinite(scale) or scale <= 0:
        if return_matches and return_inverse:
            return 1.0, 0, effective_inverse
        if return_matches:
            return 1.0, 0
        if return_inverse:
            return 1.0, effective_inverse
        return 1.0
    if return_matches and return_inverse:
        return scale, ok_count, effective_inverse
    if return_matches:
        return scale, ok_count
    if return_inverse:
        return scale, effective_inverse
    return scale


def build_mask_to_cluster(point_to_masks, point_clusters):
    counts = defaultdict(Counter)
    for pid, masks in point_to_masks.items():
        cid = point_clusters.get(pid, -1)
        if cid is None or cid < 0:
            continue
        for m in masks:
            counts[tuple(m)][cid] += 1

    mask_to_cluster = {}
    for mask_key, counter in counts.items():
        cid = counter.most_common(1)[0][0]
        mask_to_cluster[mask_key] = cid
    return mask_to_cluster


def generate_depth_seed_points(images, cameras, points3D, mask_dir, depth_dir,
                               image_base_dir, depth_suffix="", depth_is_inverse=False,
                               depth_scale_mode="median", depth_min_matches=50,
                               mask_stride=4, max_points_per_mask=20000,
                               min_depth=0.0, max_depth=0.0, random_seed=42,
                               skip_unscaled=False, depth_scale_clamp=0.0,
                               scale_ransac=False, scale_ransac_thresh=0.1,
                               scale_ransac_iters=100, scale_ransac_min_inliers=0,
                               log=print):
    """
    Generate depth-based seed points from instance masks and monocular depth maps.
    Returns:
        xyz: (M,3) float32
        rgb: (M,3) float32 in [0,1]
        mask_keys: list of (frame_name, midx) for each point
    """
    rng = np.random.RandomState(random_seed)
    all_xyz = []
    all_rgb = []
    all_mask_keys = []

    stats = {
        "total_frames": len(images),
        "depth_found": 0,
        "masks_found": 0,
        "used_frames": 0,
        "skipped_no_depth": 0,
        "skipped_no_masks": 0,
        "skipped_unscaled": 0,
    }
    used_scales = []
    inferred_inverse_by_frame = {}
    auto_inverse_overrides = set()

    global_scale = None
    if depth_scale_mode == "global" or (depth_scale_clamp and depth_scale_clamp > 0):
        valid_scales = []
        for img in images.values():
            frame_name = Path(img.name).stem
            depth_path = resolve_depth_path(depth_dir, frame_name, suffix=depth_suffix)
            if depth_path is None:
                continue
            depth = load_depth_map(depth_path)
            if depth is None:
                continue

            cam = cameras[img.camera_id]
            H, W = depth.shape[:2]
            sx = W / float(cam.width)
            sy = H / float(cam.height)
            scale_mode_for_est = "median" if depth_scale_mode == "global" else depth_scale_mode
            scale, match_count, frame_inverse = _compute_scale_factor(
                img, cam, points3D, depth,
                sx=sx, sy=sy,
                depth_is_inverse=depth_is_inverse,
                scale_mode=scale_mode_for_est,
                min_matches=depth_min_matches,
                return_matches=True,
                use_ransac=scale_ransac,
                ransac_thresh=scale_ransac_thresh,
                ransac_iters=scale_ransac_iters,
                ransac_min_inliers=scale_ransac_min_inliers,
                rng=rng,
                auto_infer_inverse=False,
                return_inverse=True,
            )
            inferred_inverse_by_frame[frame_name] = frame_inverse
            if frame_inverse != bool(depth_is_inverse):
                auto_inverse_overrides.add(frame_name)
            if match_count >= depth_min_matches and scale > 0:
                valid_scales.append(scale)
        if valid_scales:
            scales_arr = np.asarray(valid_scales, dtype=np.float32)
            if scales_arr.size >= 5:
                q1, q3 = np.percentile(scales_arr, [25.0, 75.0])
                iqr = max(float(q3 - q1), 1e-6)
                lo = max(1e-8, float(q1 - 2.5 * iqr))
                hi = float(q3 + 2.5 * iqr)
                scales_trim = scales_arr[(scales_arr >= lo) & (scales_arr <= hi)]
                if scales_trim.size >= max(3, int(0.5 * scales_arr.size)):
                    scales_arr = scales_trim
            global_scale = float(np.median(scales_arr))
            log(f"[DepthSeeds] Global scale median from {len(valid_scales)} frames: {global_scale:.6f}")
        else:
            log("[DepthSeeds][WARN] No valid scales to compute global median.")

    image_base_dir = Path(image_base_dir)

    for img in images.values():
        frame_name = Path(img.name).stem
        depth_path = resolve_depth_path(depth_dir, frame_name, suffix=depth_suffix)
        if depth_path is None:
            stats["skipped_no_depth"] += 1
            continue
        depth = load_depth_map(depth_path)
        if depth is None:
            stats["skipped_no_depth"] += 1
            continue
        stats["depth_found"] += 1

        mask_files = list_masks_for_frame(mask_dir, frame_name, log=lambda *a, **k: None)
        if not mask_files:
            stats["skipped_no_masks"] += 1
            continue
        stats["masks_found"] += 1

        rgb_path = image_base_dir / img.name
        if not rgb_path.exists():
            continue
        rgb = imageio.imread(rgb_path)
        if rgb.ndim == 2:
            rgb = np.stack([rgb, rgb, rgb], axis=-1)
        if rgb.shape[2] > 3:
            rgb = rgb[..., :3]

        first_mask = cv2.imread(str(mask_files[0]), cv2.IMREAD_GRAYSCALE)
        if first_mask is None:
            continue
        H, W = first_mask.shape[:2]

        if depth.shape[0] != H or depth.shape[1] != W:
            depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_LINEAR)
        if rgb.shape[0] != H or rgb.shape[1] != W:
            rgb = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_LINEAR)

        cam = cameras[img.camera_id]
        sx = W / float(cam.width)
        sy = H / float(cam.height)

        scale_mode_for_est = "median" if depth_scale_mode == "global" else depth_scale_mode
        frame_inverse = inferred_inverse_by_frame.get(frame_name, bool(depth_is_inverse))
        scale, match_count, frame_inverse = _compute_scale_factor(
            img, cam, points3D, depth,
            sx=sx, sy=sy,
            depth_is_inverse=frame_inverse,
            scale_mode=scale_mode_for_est,
            min_matches=depth_min_matches,
            return_matches=True,
            use_ransac=scale_ransac,
            ransac_thresh=scale_ransac_thresh,
            ransac_iters=scale_ransac_iters,
            ransac_min_inliers=scale_ransac_min_inliers,
            rng=rng,
            auto_infer_inverse=False,
            return_inverse=True,
        )
        inferred_inverse_by_frame[frame_name] = frame_inverse
        if frame_inverse != bool(depth_is_inverse):
            auto_inverse_overrides.add(frame_name)
        if depth_scale_mode == "global" and global_scale is not None:
            # Keep one shared scale in global mode to avoid frame-to-frame drift.
            scale = global_scale
        if skip_unscaled and depth_scale_mode != "none" and match_count < depth_min_matches:
            stats["skipped_unscaled"] += 1
            continue
        if depth_scale_clamp and depth_scale_clamp > 0 and global_scale is not None:
            min_scale = global_scale / depth_scale_clamp
            max_scale = global_scale * depth_scale_clamp
            scale = float(np.clip(scale, min_scale, max_scale))
        used_scales.append(scale)

        R = img.qvec2rotmat()
        t = img.tvec.reshape(3, 1)

        for mask_path in mask_files:
            stem = Path(mask_path).stem
            try:
                frame_base, midx_str = stem.rsplit("_instance_", 1)
                midx = int(midx_str)
            except Exception:
                continue

            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
            if mask.shape[0] != H or mask.shape[1] != W:
                mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)

            ys, xs = _sample_mask_points(mask > 0, mask_stride, max_points_per_mask, rng)
            if len(ys) == 0:
                continue

            d = depth[ys, xs].astype(np.float32)
            if frame_inverse:
                pos = d[d > 0]
                floor = max(float(pos.min()) * 0.1, 1e-2) if pos.size > 0 else 1e-2
                d = 1.0 / np.clip(d, floor, None)
            d = d * scale

            valid = np.isfinite(d) & (d > 1e-6)
            if min_depth > 0:
                valid &= d >= min_depth
            if max_depth > 0:
                valid &= d <= max_depth
            if not np.any(valid):
                continue

            ys = ys[valid]
            xs = xs[valid]
            d = d[valid]

            x_norm, y_norm = undistort_colmap_pixels(xs, ys, cam, sx=sx, sy=sy)
            x_cam = x_norm * d
            y_cam = y_norm * d
            z_cam = d
            cam_pts = np.stack([x_cam, y_cam, z_cam], axis=0)

            world_pts = (R.T @ (cam_pts - t)).T
            colors = rgb[ys, xs, :].astype(np.float32) / 255.0

            all_xyz.append(world_pts)
            all_rgb.append(colors)
            all_mask_keys.extend([(frame_base, midx)] * world_pts.shape[0])
        stats["used_frames"] += 1

    if not all_xyz:
        log("[DepthSeeds] No depth seeds generated.")
        return None, None, []

    xyz = np.concatenate(all_xyz, axis=0).astype(np.float32)
    rgb = np.concatenate(all_rgb, axis=0).astype(np.float32)
    if used_scales:
        log(
            "[DepthSeeds] Scale stats: "
            f"median={np.median(used_scales):.6f}, "
            f"min={np.min(used_scales):.6f}, "
            f"max={np.max(used_scales):.6f}"
        )
    if auto_inverse_overrides:
        log(
            "[DepthSeeds][WARN] Auto-detected inverse-depth polarity override on "
            f"{len(auto_inverse_overrides)} frame(s)."
        )
    log(
        "[DepthSeeds] Frames: "
        f"total={stats['total_frames']}, depth={stats['depth_found']}, "
        f"masks={stats['masks_found']}, used={stats['used_frames']}, "
        f"skip_no_depth={stats['skipped_no_depth']}, "
        f"skip_no_masks={stats['skipped_no_masks']}, "
        f"skip_unscaled={stats['skipped_unscaled']}"
    )
    log(f"[DepthSeeds] Generated {xyz.shape[0]} depth seed points.")
    return xyz, rgb, all_mask_keys


def generate_depth_seed_points_from_maps(images, cameras, points3D, mask_dir, depth_maps,
                                         image_base_dir, depth_is_inverse=False,
                                         depth_scale_mode="median", depth_min_matches=50,
                                         mask_stride=4, max_points_per_mask=20000,
                                         min_depth=0.0, max_depth=0.0, random_seed=42,
                                         skip_unscaled=False, depth_scale_clamp=0.0,
                                         scale_ransac=False, scale_ransac_thresh=0.1,
                                         scale_ransac_iters=100, scale_ransac_min_inliers=0,
                                         log=print):
    """
    Generate depth-based seed points from instance masks and in-memory depth maps.
    depth_maps: dict mapping frame_name (stem) -> depth np.ndarray
    """
    rng = np.random.RandomState(random_seed)
    all_xyz = []
    all_rgb = []
    all_mask_keys = []

    stats = {
        "total_frames": len(images),
        "depth_found": 0,
        "masks_found": 0,
        "used_frames": 0,
        "skipped_no_depth": 0,
        "skipped_no_masks": 0,
        "skipped_unscaled": 0,
    }
    used_scales = []
    inferred_inverse_by_frame = {}
    auto_inverse_overrides = set()

    global_scale = None
    if depth_scale_mode == "global" or (depth_scale_clamp and depth_scale_clamp > 0):
        valid_scales = []
        for img in images.values():
            frame_name = Path(img.name).stem
            depth = depth_maps.get(frame_name)
            if depth is None:
                continue
            if hasattr(depth, "cpu"):
                depth = depth.detach().cpu().numpy()
            if depth.ndim == 3:
                if depth.shape[0] == 1:
                    depth = depth[0]
                else:
                    depth = depth[..., 0]
            cam = cameras[img.camera_id]
            H, W = depth.shape[:2]
            sx = W / float(cam.width)
            sy = H / float(cam.height)
            scale_mode_for_est = "median" if depth_scale_mode == "global" else depth_scale_mode
            scale, match_count, frame_inverse = _compute_scale_factor(
                img, cam, points3D, depth,
                sx=sx, sy=sy,
                depth_is_inverse=depth_is_inverse,
                scale_mode=scale_mode_for_est,
                min_matches=depth_min_matches,
                return_matches=True,
                use_ransac=scale_ransac,
                ransac_thresh=scale_ransac_thresh,
                ransac_iters=scale_ransac_iters,
                ransac_min_inliers=scale_ransac_min_inliers,
                rng=rng,
                auto_infer_inverse=False,
                return_inverse=True,
            )
            inferred_inverse_by_frame[frame_name] = frame_inverse
            if frame_inverse != bool(depth_is_inverse):
                auto_inverse_overrides.add(frame_name)
            if match_count >= depth_min_matches and scale > 0:
                valid_scales.append(scale)
        if valid_scales:
            scales_arr = np.asarray(valid_scales, dtype=np.float32)
            if scales_arr.size >= 5:
                q1, q3 = np.percentile(scales_arr, [25.0, 75.0])
                iqr = max(float(q3 - q1), 1e-6)
                lo = max(1e-8, float(q1 - 2.5 * iqr))
                hi = float(q3 + 2.5 * iqr)
                scales_trim = scales_arr[(scales_arr >= lo) & (scales_arr <= hi)]
                if scales_trim.size >= max(3, int(0.5 * scales_arr.size)):
                    scales_arr = scales_trim
            global_scale = float(np.median(scales_arr))
            log(f"[DepthSeeds] Global scale median from {len(valid_scales)} frames: {global_scale:.6f}")
        else:
            log("[DepthSeeds][WARN] No valid scales to compute global median.")

    image_base_dir = Path(image_base_dir)

    for img in images.values():
        frame_name = Path(img.name).stem
        depth = depth_maps.get(frame_name)
        if depth is None:
            log(f"[DEBUG] No depth map for frame: {frame_name}")
            stats["skipped_no_depth"] += 1
            continue
        if hasattr(depth, "cpu"):
            depth = depth.detach().cpu().numpy()
        if depth.ndim == 3:
            if depth.shape[0] == 1:
                depth = depth[0]
            else:
                depth = depth[..., 0]
        log(f"[DEBUG] Depth map for {frame_name}: shape={depth.shape}, min={np.min(depth):.4f}, max={np.max(depth):.4f}, valid={(np.isfinite(depth) & (depth > 1e-6)).sum()}/{depth.size}")
        stats["depth_found"] += 1

        mask_files = list_masks_for_frame(mask_dir, frame_name, log=lambda *a, **k: None)
        if not mask_files:
            log(f"[DEBUG] No mask files for frame: {frame_name}")
            stats["skipped_no_masks"] += 1
            continue
        stats["masks_found"] += 1

        rgb_path = image_base_dir / img.name
        if not rgb_path.exists():
            log(f"[DEBUG] RGB image not found for: {img.name}")
            continue
        rgb = imageio.imread(rgb_path)
        if rgb.ndim == 2:
            rgb = np.stack([rgb, rgb, rgb], axis=-1)
        if rgb.shape[2] > 3:
            rgb = rgb[..., :3]

        first_mask = cv2.imread(str(mask_files[0]), cv2.IMREAD_GRAYSCALE)
        if first_mask is None:
            log(f"[DEBUG] Could not read first mask: {mask_files[0]}")
            continue
        H, W = first_mask.shape[:2]

        if depth.shape[0] != H or depth.shape[1] != W:
            depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_LINEAR)
        if rgb.shape[0] != H or rgb.shape[1] != W:
            rgb = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_LINEAR)

        cam = cameras[img.camera_id]
        sx = W / float(cam.width)
        sy = H / float(cam.height)

        scale_mode_for_est = "median" if depth_scale_mode == "global" else depth_scale_mode
        frame_inverse = inferred_inverse_by_frame.get(frame_name, bool(depth_is_inverse))
        scale, match_count, frame_inverse = _compute_scale_factor(
            img, cam, points3D, depth,
            sx=sx, sy=sy,
            depth_is_inverse=frame_inverse,
            scale_mode=scale_mode_for_est,
            min_matches=depth_min_matches,
            return_matches=True,
            use_ransac=scale_ransac,
            ransac_thresh=scale_ransac_thresh,
            ransac_iters=scale_ransac_iters,
            ransac_min_inliers=scale_ransac_min_inliers,
            rng=rng,
            auto_infer_inverse=False,
            return_inverse=True,
        )
        inferred_inverse_by_frame[frame_name] = frame_inverse
        if frame_inverse != bool(depth_is_inverse):
            auto_inverse_overrides.add(frame_name)
        if depth_scale_mode == "global" and global_scale is not None:
            # Keep one shared scale in global mode to avoid frame-to-frame drift.
            scale = global_scale
        if skip_unscaled and depth_scale_mode != "none" and match_count < depth_min_matches:
            stats["skipped_unscaled"] += 1
            continue
        if depth_scale_clamp and depth_scale_clamp > 0 and global_scale is not None:
            min_scale = global_scale / depth_scale_clamp
            max_scale = global_scale * depth_scale_clamp
            scale = float(np.clip(scale, min_scale, max_scale))
        used_scales.append(scale)

        R = img.qvec2rotmat()
        t = img.tvec.reshape(3, 1)

        for mask_path in mask_files:
            stem = Path(mask_path).stem
            try:
                frame_base, midx_str = stem.rsplit("_instance_", 1)
                midx = int(midx_str)
            except Exception:
                log(f"[DEBUG] Could not parse mask instance from: {mask_path}")
                continue

            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                log(f"[DEBUG] Could not read mask: {mask_path}")
                continue
            if mask.shape[0] != H or mask.shape[1] != W:
                mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)

            ys, xs = _sample_mask_points(mask > 0, mask_stride, max_points_per_mask, rng)
            log(f"[DEBUG] Mask {mask_path}: {len(ys)} points sampled.")
            if len(ys) == 0:
                continue

            d = depth[ys, xs].astype(np.float32)
            if frame_inverse:
                pos = d[d > 0]
                floor = max(float(pos.min()) * 0.1, 1e-2) if pos.size > 0 else 1e-2
                d = 1.0 / np.clip(d, floor, None)
            d = d * scale

            valid = np.isfinite(d) & (d > 1e-6)
            log(f"[DEBUG] Valid depth after filtering: {valid.sum()}/{len(d)}")
            if min_depth > 0:
                valid &= d >= min_depth
            if max_depth > 0:
                valid &= d <= max_depth
            log(f"[DEBUG] Valid depth after min/max: {valid.sum()}")
            if not np.any(valid):
                continue

            ys = ys[valid]
            xs = xs[valid]
            d = d[valid]

            x_norm, y_norm = undistort_colmap_pixels(xs, ys, cam, sx=sx, sy=sy)
            x_cam = x_norm * d
            y_cam = y_norm * d
            z_cam = d
            cam_pts = np.stack([x_cam, y_cam, z_cam], axis=0)

            world_pts = (R.T @ (cam_pts - t)).T
            colors = rgb[ys, xs, :].astype(np.float32) / 255.0

            all_xyz.append(world_pts)
            all_rgb.append(colors)
            all_mask_keys.extend([(frame_base, midx)] * world_pts.shape[0])
        stats["used_frames"] += 1

    if not all_xyz:
        log("[DepthSeeds] No depth seeds generated.")
        return None, None, []

    xyz = np.concatenate(all_xyz, axis=0).astype(np.float32)
    rgb = np.concatenate(all_rgb, axis=0).astype(np.float32)
    if used_scales:
        log(
            "[DepthSeeds] Scale stats: "
            f"median={np.median(used_scales):.6f}, "
            f"min={np.min(used_scales):.6f}, "
            f"max={np.max(used_scales):.6f}"
        )
    if auto_inverse_overrides:
        log(
            "[DepthSeeds][WARN] Auto-detected inverse-depth polarity override on "
            f"{len(auto_inverse_overrides)} frame(s)."
        )
    log(
        "[DepthSeeds] Frames: "
        f"total={stats['total_frames']}, depth={stats['depth_found']}, "
        f"masks={stats['masks_found']}, used={stats['used_frames']}, "
        f"skip_no_depth={stats['skipped_no_depth']}, "
        f"skip_no_masks={stats['skipped_no_masks']}, "
        f"skip_unscaled={stats['skipped_unscaled']}"
    )
    log(f"[DepthSeeds] Generated {xyz.shape[0]} depth seed points.")
    return xyz, rgb, all_mask_keys
