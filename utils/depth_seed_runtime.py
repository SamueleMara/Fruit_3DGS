import os
from pathlib import Path
import numpy as np
import cv2
import imageio

from scene.dataset_readers import fetchPly, storePly
from utils.graphics_utils import BasicPointCloud
from utils.read_write_model import read_model
from utils import depth_utils


def _is_colmap_model_dir(model_dir):
    p = Path(model_dir)
    if not p.is_dir():
        return False
    required = ("cameras", "images", "points3D")
    for stem in required:
        if not ((p / f"{stem}.txt").exists() or (p / f"{stem}.bin").exists()):
            return False
    return True


def load_colmap_model_with_fallback(source_path, log=print, context="DepthPCL"):
    """
    Resolve COLMAP model with strict priority:
    1) custom deep-COLMAP extractor output (if present)
    2) <source_path>/sparse/0
    3) <source_path> (legacy layout)

    This intentionally does NOT consider sparse/0_original.
    """
    src = Path(source_path)
    candidates = [
        src / "deep_colmap" / "source_for_dps" / "sparse" / "0",
        src / "sparse" / "0",
        src,
    ]

    for model_dir in candidates:
        if not _is_colmap_model_dir(model_dir):
            continue
        try:
            cameras, images, points3D = read_model(str(model_dir), ext="")
        except Exception:
            continue
        if cameras is None or images is None or points3D is None:
            continue
        log(
            f"[{context}] Using COLMAP model: {model_dir} "
            f"(cameras={len(cameras)}, images={len(images)}, points3D={len(points3D)})"
        )
        return str(model_dir), cameras, images, points3D

    log(f"[{context}][WARN] Could not read COLMAP model under: {source_path}")
    return None, None, None, None


def _as_depth_2d(depth):
    if hasattr(depth, "detach"):
        depth = depth.detach().cpu().numpy()
    elif hasattr(depth, "cpu"):
        depth = depth.cpu().numpy()
    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim == 3:
        if depth.shape[0] == 1:
            depth = depth[0]
        elif depth.shape[-1] == 1:
            depth = depth[..., 0]
        else:
            depth = depth[..., 0]
    return depth.astype(np.float32, copy=False)


def _adapt_min_matches(requested_min_matches, total_sparse_points, depth_align_mode="scale"):
    requested = max(1, int(requested_min_matches))
    # Very low match counts are numerically unstable for depth alignment and
    # can produce per-frame scale/offset outliers that duplicate geometry.
    if str(depth_align_mode).lower() == "affine":
        requested = max(requested, 24)
    else:
        requested = max(requested, 8)
    total = int(total_sparse_points)
    if total <= 0 or total >= requested:
        return requested
    # For very sparse COLMAP models (e.g. < 50 points), avoid disabling alignment entirely.
    return max(4, min(requested, max(4, total // 6)))


def _filter_points3d_for_alignment(points3D, max_reproj_error=1.5, log=print, context="DepthPCL"):
    if points3D is None:
        return points3D
    total = int(len(points3D))
    if total == 0:
        return points3D
    if max_reproj_error is None or float(max_reproj_error) <= 0:
        return points3D

    thresh = float(max_reproj_error)
    kept = {}
    finite_err = []
    for pid, p3 in points3D.items():
        err = float(getattr(p3, "error", np.nan))
        if not np.isfinite(err):
            continue
        finite_err.append(err)
        if err <= thresh:
            kept[pid] = p3

    if not kept:
        log(
            f"[{context}][WARN] Reprojection-error filtering removed all sparse points "
            f"(threshold={thresh:.2f}px). Using unfiltered points."
        )
        return points3D

    min_keep = max(24, min(512, total // 10))
    min_ratio_count = max(24, int(0.15 * total))
    if len(kept) < min_keep and len(kept) < min_ratio_count and finite_err:
        relaxed_thresh = max(thresh, float(np.percentile(np.asarray(finite_err, dtype=np.float32), 80.0)))
        relaxed = {
            pid: p3
            for pid, p3 in points3D.items()
            if np.isfinite(float(getattr(p3, "error", np.nan))) and float(getattr(p3, "error", np.nan)) <= relaxed_thresh
        }
        if len(relaxed) > len(kept):
            kept = relaxed
            thresh = relaxed_thresh

    if len(kept) < max(12, total // 50):
        log(
            f"[{context}][WARN] Too few sparse points after reprojection filtering "
            f"({len(kept)}/{total}). Using unfiltered points."
        )
        return points3D

    log(
        f"[{context}] Reprojection-error filtered sparse anchors: "
        f"{len(kept)}/{total} points (max_error={thresh:.2f}px)"
    )
    return kept


def _collect_sparse_depth_matches(img, cam, points3D, depth_map):
    pids = img.point3D_ids
    valid = pids != -1
    if not np.any(valid):
        return np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.float32)

    pids_valid = pids[valid]
    xys_valid = img.xys[valid]
    keep = np.array([int(pid) in points3D for pid in pids_valid], dtype=bool)
    if not np.any(keep):
        return np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.float32)

    pids_valid = pids_valid[keep]
    xys_valid = xys_valid[keep]
    h, w = depth_map.shape[:2]
    sx = w / float(cam.width)
    sy = h / float(cam.height)
    u = np.clip(np.round(xys_valid[:, 0] * sx).astype(np.int32), 0, w - 1)
    v = np.clip(np.round(xys_valid[:, 1] * sy).astype(np.int32), 0, h - 1)

    pred_raw = depth_map[v, u].astype(np.float32)
    R = img.qvec2rotmat()
    t = img.tvec.reshape(3, 1)
    xyz = np.array([points3D[int(pid)].xyz for pid in pids_valid], dtype=np.float32)
    cam_xyz = (R @ xyz.T) + t
    z = cam_xyz[2, :].reshape(-1).astype(np.float32)

    ok = np.isfinite(pred_raw) & np.isfinite(z) & (pred_raw > 1e-8) & (z > 1e-8)
    return pred_raw[ok], z[ok]


def _compute_pooled_global_scale(
    images, cameras, points3D, depth_maps, depth_is_inverse=False, log=print,
):
    """
    Pool sparse depth matches from ALL frames to compute one robust global
    scale factor.  This is critical when the COLMAP model is very sparse
    (< 100 points) and no single frame has enough matches on its own.

    Returns (scale, total_matches) or (None, 0) on failure.
    """
    all_pred = []
    all_z = []
    for img in images.values():
        frame_name = Path(img.name).stem
        depth = depth_maps.get(frame_name)
        if depth is None:
            continue
        depth = _as_depth_2d(depth)
        cam = cameras.get(img.camera_id)
        if cam is None:
            continue
        pred_raw, z = _collect_sparse_depth_matches(img, cam, points3D, depth)
        if pred_raw.size == 0:
            continue
        all_pred.append(pred_raw)
        all_z.append(z)

    if not all_pred:
        return None, 0

    pred = np.concatenate(all_pred)
    z = np.concatenate(all_z)

    if bool(depth_is_inverse):
        pos = pred[pred > 0]
        floor = max(float(pos.min()) * 0.1, 1e-2) if pos.size > 0 else 1e-2
        pred = 1.0 / np.clip(pred, floor, None)

    ok = np.isfinite(pred) & np.isfinite(z) & (pred > 1e-6) & (z > 1e-6)
    pred = pred[ok]
    z = z[ok]
    if pred.size < 3:
        return None, 0

    ratios = z / np.clip(pred, 1e-6, None)
    if ratios.size >= 8:
        r_lo, r_hi = np.percentile(ratios, [5.0, 95.0])
        core = (ratios >= float(r_lo)) & (ratios <= float(r_hi))
        if int(core.sum()) >= max(3, ratios.size // 3):
            ratios = ratios[core]

    scale = float(np.median(ratios))
    if not np.isfinite(scale) or scale <= 0:
        return None, 0

    log(
        f"[DepthPCL] Pooled global scale from {int(pred.size)} cross-frame matches: "
        f"scale={scale:.6f} (ratio IQR: {float(np.percentile(ratios, 25)):.4f} – "
        f"{float(np.percentile(ratios, 75)):.4f})"
    )
    return scale, int(pred.size)


def _compute_affine_alignment(
    img,
    cam,
    points3D,
    depth_map,
    depth_is_inverse=False,
    min_matches=50,
    auto_infer_inverse=False,
):
    pred_raw, z = _collect_sparse_depth_matches(img, cam, points3D, depth_map)
    match_count = int(pred_raw.size)

    frame_inverse = bool(depth_is_inverse)
    if auto_infer_inverse and match_count >= 10:
        inferred = depth_utils._infer_depth_inverse_from_matches(pred_raw, z)
        if inferred is not None:
            frame_inverse = bool(inferred)

    if match_count == 0:
        return 1.0, 0.0, 0, frame_inverse

    pred = pred_raw
    if frame_inverse:
        pos = pred_raw[pred_raw > 0]
        floor = max(float(pos.min()) * 0.1, 1e-2) if pos.size > 0 else 1e-2
        pred = 1.0 / np.clip(pred_raw, floor, None)

    ok = np.isfinite(pred) & np.isfinite(z) & (pred > 1e-8) & (z > 1e-8)
    pred = pred[ok]
    z = z[ok]
    ok_count = int(pred.size)
    if ok_count < min_matches:
        return 1.0, 0.0, ok_count, frame_inverse

    pred_med = float(np.median(pred))
    z_med = float(np.median(z))
    pred_mad = float(np.median(np.abs(pred - pred_med)))
    z_mad = float(np.median(np.abs(z - z_med)))

    if pred_mad > 1e-8 and np.isfinite(pred_mad):
        scale = z_mad / pred_mad if z_mad > 1e-8 else float(np.median(z / np.clip(pred, 1e-8, None)))
    else:
        scale = float(np.median(z / np.clip(pred, 1e-8, None)))
    offset = z_med - scale * pred_med

    # Robust linear refinement on inliers.
    residual = z - (scale * pred + offset)
    res_med = float(np.median(residual))
    res_mad = float(np.median(np.abs(residual - res_med)))
    if np.isfinite(res_mad):
        thresh = max(1e-4, 3.0 * res_mad)
        inliers = np.abs(residual - res_med) <= thresh
        if int(inliers.sum()) >= max(4, min_matches // 2):
            A = np.stack([pred[inliers], np.ones(int(inliers.sum()), dtype=np.float32)], axis=1)
            try:
                coeffs, _, _, _ = np.linalg.lstsq(A, z[inliers], rcond=None)
                scale_ls = float(coeffs[0])
                offset_ls = float(coeffs[1])
                if np.isfinite(scale_ls) and np.isfinite(offset_ls) and scale_ls > 1e-8:
                    scale = scale_ls
                    offset = offset_ls
            except Exception:
                pass

    if not np.isfinite(scale) or scale <= 1e-8:
        return 1.0, 0.0, ok_count, frame_inverse
    if not np.isfinite(offset):
        offset = 0.0
    return float(scale), float(offset), ok_count, frame_inverse


def _compute_depth_alignment(
    img,
    cam,
    points3D,
    depth_map,
    depth_is_inverse=False,
    depth_scale_mode="median",
    depth_min_matches=50,
    depth_ransac=False,
    depth_ransac_thresh=0.1,
    depth_ransac_iters=100,
    depth_ransac_min_inliers=0,
    rng=None,
    depth_align_mode="scale",
    auto_infer_inverse=False,
):
    if depth_align_mode == "scale":
        h, w = depth_map.shape[:2]
        sx = w / float(cam.width)
        sy = h / float(cam.height)
        scale, match_count, frame_inverse = depth_utils._compute_scale_factor(
            img,
            cam,
            points3D,
            depth_map,
            sx=sx,
            sy=sy,
            depth_is_inverse=depth_is_inverse,
            scale_mode=depth_scale_mode,
            min_matches=depth_min_matches,
            return_matches=True,
            use_ransac=depth_ransac,
            ransac_thresh=depth_ransac_thresh,
            ransac_iters=depth_ransac_iters,
            ransac_min_inliers=depth_ransac_min_inliers,
            rng=rng,
            auto_infer_inverse=auto_infer_inverse,
            return_inverse=True,
        )
        return float(scale), 0.0, int(match_count), bool(frame_inverse)

    # Affine mode estimates z ~= scale * depth + offset (or inverse-depth equivalent).
    return _compute_affine_alignment(
        img,
        cam,
        points3D,
        depth_map,
        depth_is_inverse=depth_is_inverse,
        min_matches=depth_min_matches,
        auto_infer_inverse=auto_infer_inverse,
    )


def _apply_depth_alignment(depth_values, frame_inverse, scale, offset):
    d = np.asarray(depth_values, dtype=np.float32)
    if frame_inverse:
        valid = np.isfinite(d) & (d > 0)
        if valid.any():
            positive_min = float(np.min(d[valid]))
            floor = max(positive_min * 0.1, 1e-2)
        else:
            floor = 1e-2
        d = 1.0 / np.clip(d, floor, None)
    return d * float(scale) + float(offset)


def _alignment_residual_stats(img, cam, points3D, depth_map, frame_inverse, scale, offset):
    """
    Compare aligned depth predictions against sparse COLMAP z for one frame.
    Returns (median_rel_err, p90_rel_err, match_count) or (None, None, count).
    """
    pred_raw, z = _collect_sparse_depth_matches(img, cam, points3D, depth_map)
    count = int(pred_raw.size)
    if count < 8:
        return None, None, count

    pred_aligned = _apply_depth_alignment(pred_raw, frame_inverse, scale, offset)
    ok = np.isfinite(pred_aligned) & np.isfinite(z) & (pred_aligned > 1e-8) & (z > 1e-8)
    if int(ok.sum()) < 8:
        return None, None, int(ok.sum())

    rel = np.abs(pred_aligned[ok] - z[ok]) / np.clip(z[ok], 1e-6, None)
    if rel.size < 8:
        return None, None, int(rel.size)
    return float(np.median(rel)), float(np.percentile(rel, 90.0)), int(rel.size)


def _load_frame_projection_mask(mask_dir, frame_name):
    """
    Load a binary projection mask for one frame.
    Supports:
      1) instance masks: <frame_name>_instance_*.png (merged by union)
      2) single-frame masks: <frame_name>.(png|jpg|jpeg|bmp|tif|tiff|webp)
    """
    if mask_dir is None:
        return None
    root = Path(mask_dir)
    if not root.is_dir():
        return None

    merged = None
    instance_files = sorted(root.glob(f"{frame_name}_instance_*.png"))
    for mask_path in instance_files:
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        mask_bin = (mask > 0)
        if merged is None:
            merged = mask_bin
        else:
            if merged.shape != mask_bin.shape:
                mask_bin = cv2.resize(
                    mask_bin.astype(np.uint8),
                    (merged.shape[1], merged.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                ) > 0
            merged |= mask_bin

    if merged is not None:
        return merged

    for ext in ("png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp"):
        candidate = root / f"{frame_name}.{ext}"
        if not candidate.exists():
            continue
        mask = cv2.imread(str(candidate), cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            return (mask > 0)

    return None


def _build_consistency_grid_index(u_s, v_s, z_s, cell_size):
    if u_s is None or v_s is None or z_s is None:
        return None
    if int(len(u_s)) == 0:
        return None
    grid = {}
    cu_s = (u_s // cell_size).astype(np.int32)
    cv_s = (v_s // cell_size).astype(np.int32)
    for idx in range(u_s.shape[0]):
        key = (int(cu_s[idx]), int(cv_s[idx]))
        grid.setdefault(key, []).append(idx)
    return (u_s.astype(np.int32), v_s.astype(np.int32), z_s.astype(np.float32), grid, int(cell_size))


def _collect_observed_colmap_support(img, points3D, sx, sy, w, h):
    pids = np.asarray(img.point3D_ids)
    valid_obs = (pids != -1)
    if not np.any(valid_obs):
        return (
            np.empty((0,), dtype=np.int32),
            np.empty((0,), dtype=np.int32),
            np.empty((0,), dtype=np.float32),
        )

    pids_v = pids[valid_obs]
    xys_v = np.asarray(img.xys)[valid_obs]
    u_s = np.round(xys_v[:, 0] * float(sx)).astype(np.int32)
    v_s = np.round(xys_v[:, 1] * float(sy)).astype(np.int32)
    inb = (u_s >= 0) & (u_s < int(w)) & (v_s >= 0) & (v_s < int(h))
    u_s = u_s[inb]
    v_s = v_s[inb]
    pids_v = pids_v[inb]
    if u_s.size == 0:
        return (
            np.empty((0,), dtype=np.int32),
            np.empty((0,), dtype=np.int32),
            np.empty((0,), dtype=np.float32),
        )

    Rm = img.qvec2rotmat()
    tm = img.tvec.reshape(3, 1)
    z_s = np.empty((u_s.shape[0],), dtype=np.float32)
    keep_s = np.ones((u_s.shape[0],), dtype=bool)
    for i, pid in enumerate(pids_v):
        p3 = points3D.get(int(pid))
        if p3 is None:
            keep_s[i] = False
            continue
        X = np.asarray(p3.xyz, dtype=np.float32).reshape(3, 1)
        z_s[i] = float(((Rm @ X) + tm)[2, 0])
    keep_s &= np.isfinite(z_s) & (z_s > 1e-6)
    return u_s[keep_s], v_s[keep_s], z_s[keep_s]


def _collect_projected_colmap_support(img, cam, world_xyz, sx, sy, w, h):
    if world_xyz is None or int(world_xyz.shape[0]) == 0:
        return (
            np.empty((0,), dtype=np.int32),
            np.empty((0,), dtype=np.int32),
            np.empty((0,), dtype=np.float32),
        )

    Rm = img.qvec2rotmat().astype(np.float32)
    tm = img.tvec.reshape(3, 1).astype(np.float32)
    cam_xyz = (Rm @ world_xyz.T) + tm
    z = cam_xyz[2, :]
    valid = np.isfinite(z) & (z > 1e-6)
    if not np.any(valid):
        return (
            np.empty((0,), dtype=np.int32),
            np.empty((0,), dtype=np.int32),
            np.empty((0,), dtype=np.float32),
        )

    x = cam_xyz[0, valid] / z[valid]
    y = cam_xyz[1, valid] / z[valid]
    z = z[valid].astype(np.float32)

    fx, fy, cx, cy = depth_utils.get_colmap_intrinsics(cam)
    fx *= float(sx)
    fy *= float(sy)
    cx *= float(sx)
    cy *= float(sy)
    u = np.round(fx * x + cx).astype(np.int32)
    v = np.round(fy * y + cy).astype(np.int32)
    inb = (u >= 0) & (u < int(w)) & (v >= 0) & (v < int(h))
    u = u[inb]
    v = v[inb]
    z = z[inb]
    if u.size == 0:
        return (
            np.empty((0,), dtype=np.int32),
            np.empty((0,), dtype=np.int32),
            np.empty((0,), dtype=np.float32),
        )

    # Keep nearest projected support per pixel.
    best = {}
    for ui, vi, zi in zip(u.tolist(), v.tolist(), z.tolist()):
        key = (int(ui), int(vi))
        prev = best.get(key)
        if prev is None or zi < prev:
            best[key] = zi
    if not best:
        return (
            np.empty((0,), dtype=np.int32),
            np.empty((0,), dtype=np.int32),
            np.empty((0,), dtype=np.float32),
        )

    u_out = np.fromiter((k[0] for k in best.keys()), dtype=np.int32, count=len(best))
    v_out = np.fromiter((k[1] for k in best.keys()), dtype=np.int32, count=len(best))
    z_out = np.fromiter(best.values(), dtype=np.float32, count=len(best))
    return u_out, v_out, z_out


def _merge_colmap_support_points(u_a, v_a, z_a, u_b, v_b, z_b):
    if u_a.size == 0 and u_b.size == 0:
        return (
            np.empty((0,), dtype=np.int32),
            np.empty((0,), dtype=np.int32),
            np.empty((0,), dtype=np.float32),
        )
    if u_a.size == 0:
        return u_b.astype(np.int32), v_b.astype(np.int32), z_b.astype(np.float32)
    if u_b.size == 0:
        return u_a.astype(np.int32), v_a.astype(np.int32), z_a.astype(np.float32)

    u = np.concatenate([u_a, u_b], axis=0).astype(np.int32)
    v = np.concatenate([v_a, v_b], axis=0).astype(np.int32)
    z = np.concatenate([z_a, z_b], axis=0).astype(np.float32)
    best = {}
    for ui, vi, zi in zip(u.tolist(), v.tolist(), z.tolist()):
        key = (int(ui), int(vi))
        prev = best.get(key)
        if prev is None or zi < prev:
            best[key] = zi

    u_out = np.fromiter((k[0] for k in best.keys()), dtype=np.int32, count=len(best))
    v_out = np.fromiter((k[1] for k in best.keys()), dtype=np.int32, count=len(best))
    z_out = np.fromiter(best.values(), dtype=np.float32, count=len(best))
    return u_out, v_out, z_out


def _infer_global_depth_inverse(
    images,
    cameras,
    points3D,
    depth_maps,
    default_inverse=False,
    log=print,
):
    metric_votes = 0
    inverse_votes = 0
    unknown = 0
    for img in images.values():
        frame_name = Path(img.name).stem
        depth = depth_maps.get(frame_name)
        if depth is None:
            continue
        depth = _as_depth_2d(depth)
        cam = cameras.get(img.camera_id)
        if cam is None:
            continue
        pred_raw, z = _collect_sparse_depth_matches(img, cam, points3D, depth)
        if pred_raw.size < 10:
            unknown += 1
            continue
        inferred = depth_utils._infer_depth_inverse_from_matches(pred_raw, z)
        if inferred is None:
            unknown += 1
            continue
        if bool(inferred):
            inverse_votes += 1
        else:
            metric_votes += 1

    total_votes = metric_votes + inverse_votes
    if total_votes < 3:
        log(
            "[DepthPCL][Audit] Global depth polarity inference inconclusive "
            f"(votes={total_votes}, unknown={unknown}); using provided depth_is_inverse={bool(default_inverse)}"
        )
        return bool(default_inverse)

    vote_gap = abs(metric_votes - inverse_votes)
    majority_ratio = max(metric_votes, inverse_votes) / float(total_votes)
    if vote_gap < 2 or majority_ratio < 0.7:
        log(
            "[DepthPCL][Audit] Global depth polarity weak consensus "
            f"(metric={metric_votes}, inverse={inverse_votes}, unknown={unknown}); "
            f"using provided depth_is_inverse={bool(default_inverse)}"
        )
        return bool(default_inverse)

    decided_inverse = bool(inverse_votes > metric_votes)
    log(
        "[DepthPCL][Audit] Global depth polarity inferred: "
        f"depth_is_inverse={decided_inverse} (metric={metric_votes}, inverse={inverse_votes}, unknown={unknown})"
    )
    return decided_inverse


def _audit_frame_alignment(images, depth_maps, log=print):
    image_stems = {Path(img.name).stem for img in images.values()}
    depth_stems = {Path(str(k)).stem for k in depth_maps.keys()}
    missing_depth = sorted(image_stems - depth_stems)
    extra_depth = sorted(depth_stems - image_stems)
    log(
        "[DepthPCL][Audit] frame alignment: "
        f"images={len(image_stems)}, depth_maps={len(depth_stems)}, "
        f"missing_depth={len(missing_depth)}, extra_depth={len(extra_depth)}"
    )
    if missing_depth:
        log(f"[DepthPCL][Audit] missing depth for first frames: {missing_depth[:5]}")
    if extra_depth:
        log(f"[DepthPCL][Audit] extra depth maps without poses: {extra_depth[:5]}")


def _audit_pose_convention(images, cameras, points3D, log=print, max_images=8, max_points_per_image=128):
    errs_colmap = []
    errs_alt = []
    for img in sorted(images.values(), key=lambda im: im.id)[:max_images]:
        cam = cameras.get(img.camera_id)
        if cam is None:
            continue
        fx, fy, cx, cy = depth_utils.get_colmap_intrinsics(cam)
        R = img.qvec2rotmat()
        t = img.tvec.reshape(3, 1)
        pids = img.point3D_ids
        valid_idx = np.where(pids != -1)[0]
        if valid_idx.size == 0:
            continue
        step = max(1, int(np.ceil(valid_idx.size / float(max_points_per_image))))
        for idx in valid_idx[::step]:
            pid = int(pids[idx])
            if pid not in points3D:
                continue
            world_gt = np.asarray(points3D[pid].xyz, dtype=np.float32).reshape(3, 1)
            cam_pt = (R @ world_gt) + t
            z = float(cam_pt[2, 0])
            if z <= 1e-8:
                continue

            u = fx * float(cam_pt[0, 0]) / z + cx
            v = fy * float(cam_pt[1, 0]) / z + cy
            x_cam = (u - cx) * z / fx
            y_cam = (v - cy) * z / fy
            rec_cam = np.array([[x_cam], [y_cam], [z]], dtype=np.float32)

            world_colmap = (R.T @ (rec_cam - t)).reshape(3)
            world_alt = (R @ rec_cam + t).reshape(3)
            gt = world_gt.reshape(3)
            errs_colmap.append(float(np.linalg.norm(world_colmap - gt)))
            errs_alt.append(float(np.linalg.norm(world_alt - gt)))

    if not errs_colmap:
        log("[DepthPCL][Audit] Pose convention audit skipped: no valid sparse correspondences.")
        return

    med_colmap = float(np.median(errs_colmap))
    med_alt = float(np.median(errs_alt))
    log(
        "[DepthPCL][Audit] Pose convention check: "
        f"COLMAP formula median error={med_colmap:.6e}, alt formula median error={med_alt:.6e}"
    )
    if med_colmap > med_alt:
        log("[DepthPCL][WARN] Alternate pose convention fits sparse points better than expected.")


def add_depth_seed_points_from_maps(
    scene,
    dataset,
    depth_maps,
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
    depth_max_reproj_error=1.5,
):
    if depth_maps is None or len(depth_maps) == 0:
        print("[WARNING] depth_maps is empty; depth seeds skipped.")
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
    points3D = _filter_points3d_for_alignment(
        points3D,
        max_reproj_error=depth_max_reproj_error,
        log=print,
        context="DepthSeeds",
    )

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

    depth_xyz, depth_rgb, _ = depth_utils.generate_depth_seed_points_from_maps(
        images=images,
        cameras=cameras,
        points3D=points3D,
        mask_dir=depth_seed_mask_dir,
        depth_maps=depth_maps,
        image_base_dir=image_base_dir,
        depth_is_inverse=depth_seed_inverse,
        depth_scale_mode=depth_seed_scale_mode,
        depth_min_matches=depth_seed_min_matches,
        mask_stride=depth_seed_stride,
        max_points_per_mask=depth_seed_max_points,
        min_depth=depth_seed_min_depth,
        max_depth=depth_seed_max_depth,
        random_seed=depth_seed_random_seed,
        skip_unscaled=depth_seed_skip_unscaled,
        depth_scale_clamp=depth_seed_scale_clamp,
        scale_ransac=depth_seed_ransac,
        scale_ransac_thresh=depth_seed_ransac_thresh,
        scale_ransac_iters=depth_seed_ransac_iters,
        scale_ransac_min_inliers=depth_seed_ransac_min_inliers,
        log=print,
    )

    if depth_xyz is None or depth_rgb is None or depth_xyz.shape[0] == 0:
        print("[WARNING] No depth seed points generated.")
        return False

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
    return True


def scale_depth_maps_to_colmap(
    depth_maps_by_stem,
    dataset,
    depth_is_inverse=False,
    depth_scale_mode="median",
    depth_min_matches=50,
    depth_scale_clamp=0.0,
    skip_unscaled=False,
    depth_ransac=False,
    depth_ransac_thresh=0.1,
    depth_ransac_iters=100,
    depth_ransac_min_inliers=0,
    depth_align_mode="scale",
    depth_max_reproj_error=1.5,
):
    """
    Scale depth maps (in-memory) to COLMAP scale using sparse points.

    Returns:
        scaled_by_stem: dict[stem] -> depth map
        scaled_by_name: dict[image_name] -> depth map
    """
    _, cameras, images, points3D = load_colmap_model_with_fallback(
        dataset.source_path,
        log=print,
        context="DepthSeeds",
    )
    if cameras is None or images is None or points3D is None:
        print(f"[WARNING] Unable to read COLMAP model under: {dataset.source_path}")
        return {}, {}
    points3D = _filter_points3d_for_alignment(
        points3D,
        max_reproj_error=depth_max_reproj_error,
        log=print,
        context="DepthSeeds",
    )

    _audit_frame_alignment(images, depth_maps_by_stem, log=print)
    rng = np.random.RandomState(0)
    effective_min_matches = _adapt_min_matches(
        depth_min_matches,
        len(points3D),
        depth_align_mode=depth_align_mode,
    )
    if effective_min_matches != int(depth_min_matches):
        if effective_min_matches > int(depth_min_matches):
            print(
                "[DepthSeeds][WARN] Requested depth_min_matches="
                f"{int(depth_min_matches)} is too low for stable {depth_align_mode} alignment. "
                f"Using {effective_min_matches}."
            )
        else:
            print(
                "[DepthSeeds][WARN] Requested depth_min_matches="
                f"{int(depth_min_matches)} is too high for sparse COLMAP ({len(points3D)} points). "
                f"Using {effective_min_matches} for depth alignment."
            )
    effective_depth_scale_mode = depth_scale_mode
    if depth_scale_mode == "median" and len(points3D) < 100:
        effective_depth_scale_mode = "global"
        print(
            "[DepthSeeds][WARN] Sparse COLMAP model detected; switching depth_scale_mode "
            "from 'median' to 'global' for stable multi-view alignment."
        )
    global_inverse = _infer_global_depth_inverse(
        images=images,
        cameras=cameras,
        points3D=points3D,
        depth_maps=depth_maps_by_stem,
        default_inverse=depth_is_inverse,
        log=print,
    )

    global_scale = None
    global_offset = 0.0
    if effective_depth_scale_mode == "global" or (depth_scale_clamp and depth_scale_clamp > 0):
        valid_scales = []
        valid_offsets = []
        for img in images.values():
            frame_name = Path(img.name).stem
            depth = depth_maps_by_stem.get(frame_name)
            if depth is None:
                continue
            depth = _as_depth_2d(depth)
            cam = cameras[img.camera_id]
            scale_mode_for_est = "median" if effective_depth_scale_mode == "global" else effective_depth_scale_mode
            scale, offset, match_count, frame_inverse = _compute_depth_alignment(
                img=img,
                cam=cam,
                points3D=points3D,
                depth_map=depth,
                depth_is_inverse=global_inverse,
                depth_scale_mode=scale_mode_for_est,
                depth_min_matches=effective_min_matches,
                depth_ransac=depth_ransac,
                depth_ransac_thresh=depth_ransac_thresh,
                depth_ransac_iters=depth_ransac_iters,
                depth_ransac_min_inliers=depth_ransac_min_inliers,
                rng=rng,
                depth_align_mode=depth_align_mode,
                auto_infer_inverse=False,
            )
            if match_count >= effective_min_matches and scale > 0:
                valid_scales.append(scale)
                if depth_align_mode == "affine":
                    valid_offsets.append(offset)
        if valid_scales:
            scales_arr = np.asarray(valid_scales, dtype=np.float32)
            # Trim extreme frame scales before global aggregation.
            if scales_arr.size >= 5:
                q1, q3 = np.percentile(scales_arr, [25.0, 75.0])
                iqr = max(float(q3 - q1), 1e-6)
                lo = max(1e-8, float(q1 - 2.5 * iqr))
                hi = float(q3 + 2.5 * iqr)
                scales_trim = scales_arr[(scales_arr >= lo) & (scales_arr <= hi)]
                if scales_trim.size >= max(3, int(0.5 * scales_arr.size)):
                    scales_arr = scales_trim
            global_scale = float(np.median(scales_arr))
            if depth_align_mode == "affine" and valid_offsets:
                offsets_arr = np.asarray(valid_offsets, dtype=np.float32)
                global_offset = float(np.median(offsets_arr))
            else:
                global_offset = 0.0
            print(
                f"[DepthSeeds] Global alignment median from {len(valid_scales)} frames: "
                f"scale={global_scale:.6f}, offset={global_offset:.6f}"
            )
        else:
            print("[DepthSeeds][WARN] No per-frame alignments passed threshold. Trying pooled cross-frame alignment...")
            pooled_scale, pooled_n = _compute_pooled_global_scale(
                images, cameras, points3D, depth_maps_by_stem,
                depth_is_inverse=global_inverse, log=print,
            )
            if pooled_scale is not None:
                global_scale = pooled_scale
                global_offset = 0.0
            else:
                print("[DepthSeeds][WARN] Pooled alignment also failed.")

    scaled_by_stem = {}
    scaled_by_name = {}
    for img in images.values():
        frame_name = Path(img.name).stem
        depth = depth_maps_by_stem.get(frame_name)
        if depth is None:
            continue
        depth = _as_depth_2d(depth)

        if effective_depth_scale_mode == "none":
            scale = 1.0
            offset = 0.0
            match_count = effective_min_matches
            frame_inverse = bool(global_inverse)
        else:
            cam = cameras[img.camera_id]
            scale_mode_for_est = "median" if effective_depth_scale_mode == "global" else effective_depth_scale_mode
            scale, offset, match_count, frame_inverse = _compute_depth_alignment(
                img=img,
                cam=cam,
                points3D=points3D,
                depth_map=depth,
                depth_is_inverse=global_inverse,
                depth_scale_mode=scale_mode_for_est,
                depth_min_matches=effective_min_matches,
                depth_ransac=depth_ransac,
                depth_ransac_thresh=depth_ransac_thresh,
                depth_ransac_iters=depth_ransac_iters,
                depth_ransac_min_inliers=depth_ransac_min_inliers,
                rng=rng,
                depth_align_mode=depth_align_mode,
                auto_infer_inverse=False,
            )

        if effective_depth_scale_mode == "global" and global_scale is not None:
            scale = global_scale
            offset = global_offset if depth_align_mode == "affine" else 0.0
        elif skip_unscaled and effective_depth_scale_mode != "none" and match_count < effective_min_matches:
            continue
        if depth_scale_clamp and depth_scale_clamp > 0 and global_scale is not None:
            min_scale = global_scale / depth_scale_clamp
            max_scale = global_scale * depth_scale_clamp
            scale = float(np.clip(scale, min_scale, max_scale))

        depth_scaled = _apply_depth_alignment(depth, frame_inverse, scale, offset)
        scaled_by_stem[frame_name] = depth_scaled
        scaled_by_name[img.name] = depth_scaled

    return scaled_by_stem, scaled_by_name


def export_depth_point_cloud_from_maps(
    depth_maps,
    dataset,
    output_dir,
    mask_dir=None,
    depth_is_inverse=False,
    depth_scale_mode="median",
    depth_min_matches=50,
    depth_scale_clamp=0.0,
    skip_unscaled=False,
    depth_ransac=False,
    depth_ransac_thresh=0.1,
    depth_ransac_iters=100,
    depth_ransac_min_inliers=0,
    depth_align_mode="scale",
    depth_max_reproj_error=1.5,
    colmap_consistency: bool = True,
    colmap_consistency_radius_px: int = 6,
    colmap_consistency_rel_depth: float = 0.20,
    colmap_consistency_require_support: bool = False,
    sample_stride=4,
    min_depth=0.0,
    max_depth=0.0,
    auto_clip_percentile_min=1.0,
    auto_clip_percentile_max=99.0,
    output_name="depth_anything_points.ply",
    save_debug_clouds=True,
    log=print,
):
    """
    Export a point cloud from in-memory depth maps (frame stem -> depth map).
    Depths are aligned to COLMAP scale using sparse points before backprojection.
    """
    if depth_maps is None or len(depth_maps) == 0:
        log("[WARNING] Depth point cloud export skipped: no depth maps available.")
        return None

    _, cameras, images, points3D = load_colmap_model_with_fallback(
        dataset.source_path,
        log=log,
        context="DepthPCL",
    )
    if cameras is None or images is None or points3D is None:
        log(f"[WARNING] Depth point cloud export skipped: could not read a COLMAP model under {dataset.source_path}")
        return None
    points3D = _filter_points3d_for_alignment(
        points3D,
        max_reproj_error=depth_max_reproj_error,
        log=log,
        context="DepthPCL",
    )
    projected_support_xyz = None
    if bool(colmap_consistency) and points3D:
        try:
            projected_support_xyz = np.asarray(
                [np.asarray(p3.xyz, dtype=np.float32) for p3 in points3D.values()],
                dtype=np.float32,
            )
            log(
                f"[DepthPCL] Using projected COLMAP supports from "
                f"{int(projected_support_xyz.shape[0])} sparse points."
            )
        except Exception:
            projected_support_xyz = None

    image_base_dir = Path(dataset.source_path) / dataset.images
    if not image_base_dir.exists():
        log(f"[WARNING] Depth point cloud export skipped: image dir not found at {image_base_dir}")
        return None

    stride = max(1, int(sample_stride))
    rng = np.random.RandomState(0)
    global_inverse = bool(depth_is_inverse)
    used_scales = []
    used_offsets = []
    use_projection_masks = False
    mask_root = None
    mask_cache = {}
    skipped_missing_mask = 0
    if mask_dir is not None:
        mask_root = Path(mask_dir)
        if mask_root.is_dir():
            use_projection_masks = True
            log(f"[DepthPCL] Applying projection masks from: {mask_root}")
        else:
            log(f"[DepthPCL][WARN] mask_dir does not exist, disabling masked projection: {mask_root}")
    effective_require_support = bool(colmap_consistency_require_support)
    if bool(colmap_consistency) and not use_projection_masks and not effective_require_support:
        effective_require_support = True
        log(
            "[DepthPCL][WARN] No projection masks provided; enabling strict COLMAP support "
            "requirement to suppress background-sheet artifacts."
        )

    effective_min_matches = _adapt_min_matches(
        depth_min_matches,
        len(points3D),
        depth_align_mode=depth_align_mode,
    )
    if effective_min_matches != int(depth_min_matches):
        if effective_min_matches > int(depth_min_matches):
            log(
                "[DepthPCL][WARN] Requested depth_min_matches="
                f"{int(depth_min_matches)} is too low for stable {depth_align_mode} alignment. "
                f"Using {effective_min_matches}."
            )
        else:
            log(
                "[DepthPCL][WARN] Requested depth_min_matches="
                f"{int(depth_min_matches)} is too high for sparse COLMAP ({len(points3D)} points). "
                f"Using {effective_min_matches} for alignment."
            )
    effective_depth_scale_mode = depth_scale_mode
    if depth_scale_mode == "median" and len(points3D) < 100:
        effective_depth_scale_mode = "global"
        log(
            "[DepthPCL][WARN] Sparse COLMAP model detected; switching depth_scale_mode "
            "from 'median' to 'global' for stable multi-view alignment."
        )
    log(
        f"[DepthPCL] Alignment config: scale_mode={effective_depth_scale_mode}, "
        f"align_mode={depth_align_mode}, min_matches={effective_min_matches}"
    )

    _audit_frame_alignment(images, depth_maps, log=log)
    _audit_pose_convention(images, cameras, points3D, log=log)
    global_inverse = _infer_global_depth_inverse(
        images=images,
        cameras=cameras,
        points3D=points3D,
        depth_maps=depth_maps,
        default_inverse=depth_is_inverse,
        log=log,
    )

    global_scale = None
    global_offset = 0.0
    if effective_depth_scale_mode == "global" or (depth_scale_clamp and depth_scale_clamp > 0):
        valid_scales = []
        valid_offsets = []
        for img in images.values():
            frame_name = Path(img.name).stem
            depth = depth_maps.get(frame_name)
            if depth is None:
                continue
            depth = _as_depth_2d(depth)
            cam = cameras[img.camera_id]
            mode_for_est = "median" if effective_depth_scale_mode == "global" else effective_depth_scale_mode
            scale, offset, match_count, frame_inverse = _compute_depth_alignment(
                img=img,
                cam=cam,
                points3D=points3D,
                depth_map=depth,
                depth_is_inverse=global_inverse,
                depth_scale_mode=mode_for_est,
                depth_min_matches=effective_min_matches,
                depth_ransac=depth_ransac,
                depth_ransac_thresh=depth_ransac_thresh,
                depth_ransac_iters=depth_ransac_iters,
                depth_ransac_min_inliers=depth_ransac_min_inliers,
                rng=rng,
                depth_align_mode=depth_align_mode,
                auto_infer_inverse=False,
            )
            if match_count >= effective_min_matches and scale > 0:
                valid_scales.append(scale)
                if depth_align_mode == "affine":
                    valid_offsets.append(offset)
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
            if depth_align_mode == "affine" and valid_offsets:
                offsets_arr = np.asarray(valid_offsets, dtype=np.float32)
                global_offset = float(np.median(offsets_arr))
            else:
                global_offset = 0.0
            log(
                f"[DepthPCL] Global alignment from {len(valid_scales)} frames: "
                f"scale={global_scale:.6f}, offset={global_offset:.6f}"
            )
        else:
            log("[DepthPCL][WARN] No per-frame alignments passed threshold. Trying pooled cross-frame alignment...")
            pooled_scale, pooled_n = _compute_pooled_global_scale(
                images, cameras, points3D, depth_maps,
                depth_is_inverse=global_inverse, log=log,
            )
            if pooled_scale is not None:
                global_scale = pooled_scale
                global_offset = 0.0
            else:
                log("[DepthPCL][WARN] Pooled alignment also failed. Export will proceed with scale=1 (unaligned).")

    all_xyz = []
    all_rgb = []
    per_frame_clouds = []
    used_frames = 0
    skipped_suspicious = 0
    dropped_inconsistent = 0
    dropped_no_support = 0

    ordered_images = sorted(images.values(), key=lambda im: im.id)
    for img in ordered_images:
        frame_name = Path(img.name).stem
        depth = depth_maps.get(frame_name)
        if depth is None:
            continue
        depth = _as_depth_2d(depth)

        rgb_path = image_base_dir / img.name
        if not rgb_path.exists():
            continue
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

        h, w = rgb.shape[:2]
        if depth.shape[:2] != (h, w):
            depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)

        cam = cameras[img.camera_id]
        sx = w / float(cam.width)
        sy = h / float(cam.height)

        # Build a COLMAP z lookup for consistency filtering.
        # We merge:
        # 1) observed sparse correspondences in this image
        # 2) projected supports from all filtered COLMAP points (denser anchors)
        sparse_index = None
        if bool(colmap_consistency):
            cell_size = max(2, int(colmap_consistency_radius_px))
            u_obs, v_obs, z_obs = _collect_observed_colmap_support(
                img=img,
                points3D=points3D,
                sx=sx,
                sy=sy,
                w=w,
                h=h,
            )
            u_proj, v_proj, z_proj = _collect_projected_colmap_support(
                img=img,
                cam=cam,
                world_xyz=projected_support_xyz,
                sx=sx,
                sy=sy,
                w=w,
                h=h,
            )
            u_s, v_s, z_s = _merge_colmap_support_points(
                u_obs, v_obs, z_obs, u_proj, v_proj, z_proj
            )
            sparse_index = _build_consistency_grid_index(u_s, v_s, z_s, cell_size=cell_size)

        if sparse_index is None and bool(colmap_consistency) and bool(effective_require_support):
            dropped_no_support += 1
            continue

        mode_for_est = "median" if effective_depth_scale_mode == "global" else effective_depth_scale_mode
        frame_inverse = bool(global_inverse)
        if effective_depth_scale_mode == "none":
            scale = 1.0
            offset = 0.0
            match_count = effective_min_matches
        else:
            scale, offset, match_count, frame_inverse = _compute_depth_alignment(
                img=img,
                cam=cam,
                points3D=points3D,
                depth_map=depth,
                depth_is_inverse=global_inverse,
                depth_scale_mode=mode_for_est,
                depth_min_matches=effective_min_matches,
                depth_ransac=depth_ransac,
                depth_ransac_thresh=depth_ransac_thresh,
                depth_ransac_iters=depth_ransac_iters,
                depth_ransac_min_inliers=depth_ransac_min_inliers,
                rng=rng,
                depth_align_mode=depth_align_mode,
                auto_infer_inverse=False,
            )

        if effective_depth_scale_mode == "global" and global_scale is not None:
            # Keep per-frame affine estimates when enough sparse support exists.
            # Fall back to global values on weak frames.
            if depth_align_mode == "affine":
                if match_count < effective_min_matches:
                    scale = global_scale
                    offset = global_offset
            else:
                scale = global_scale
                offset = 0.0
        elif skip_unscaled and effective_depth_scale_mode != "none" and match_count < effective_min_matches:
            continue
        if depth_scale_clamp and depth_scale_clamp > 0 and global_scale is not None:
            min_scale = global_scale / depth_scale_clamp
            max_scale = global_scale * depth_scale_clamp
            scale = float(np.clip(scale, min_scale, max_scale))
        used_scales.append(float(scale))
        used_offsets.append(float(offset))

        # Drop frames whose aligned sparse-depth residuals are still too high.
        if effective_depth_scale_mode != "none":
            rel_med, rel_p90, rel_n = _alignment_residual_stats(
                img=img,
                cam=cam,
                points3D=points3D,
                depth_map=depth,
                frame_inverse=frame_inverse,
                scale=scale,
                offset=offset,
            )
            if rel_med is not None and rel_n >= max(12, effective_min_matches):
                if rel_med > 0.45 and rel_p90 > 0.95:
                    skipped_suspicious += 1
                    log(
                        f"[DepthPCL][WARN] Skipping high-residual frame '{frame_name}' "
                        f"(rel_med={rel_med:.4f}, rel_p90={rel_p90:.4f}, matches={rel_n}, "
                        f"scale={float(scale):.6f}, offset={float(offset):.6f})."
                    )
                    continue

        ys = np.arange(0, h, stride, dtype=np.int32)
        xs = np.arange(0, w, stride, dtype=np.int32)
        if ys.size == 0 or xs.size == 0:
            continue
        yy, xx = np.meshgrid(ys, xs, indexing="ij")
        yy = yy.reshape(-1)
        xx = xx.reshape(-1)

        if use_projection_masks:
            frame_mask = mask_cache.get(frame_name, None)
            if frame_name not in mask_cache:
                frame_mask = _load_frame_projection_mask(mask_root, frame_name)
                mask_cache[frame_name] = frame_mask
            if frame_mask is None:
                skipped_missing_mask += 1
                log(f"[DepthPCL][WARN] Skipping frame '{frame_name}': no projection mask found.")
                continue
            if frame_mask.shape[:2] != (h, w):
                frame_mask = cv2.resize(
                    frame_mask.astype(np.uint8),
                    (w, h),
                    interpolation=cv2.INTER_NEAREST,
                ) > 0
            keep = frame_mask[yy, xx]
            if not np.any(keep):
                continue
            yy = yy[keep]
            xx = xx[keep]

        d = _apply_depth_alignment(depth[yy, xx], frame_inverse, scale, offset)
        valid = np.isfinite(d) & (d > 1e-6)
        if min_depth > 0:
            valid &= d >= float(min_depth)
        if max_depth > 0:
            valid &= d <= float(max_depth)
        if (
            auto_clip_percentile_min is not None
            and auto_clip_percentile_max is not None
            and 0.0 <= float(auto_clip_percentile_min) < float(auto_clip_percentile_max) <= 100.0
        ):
            d_valid = d[valid]
            if d_valid.size >= 32:
                pmin = float(np.percentile(d_valid, auto_clip_percentile_min))
                pmax = float(np.percentile(d_valid, auto_clip_percentile_max))
                valid &= d >= pmin
                valid &= d <= pmax
        # If a frame is dominated by points hugging the max-depth cap, it usually creates
        # long "curtain" artifacts from low-confidence background regions.
        if max_depth > 0:
            near_cap = valid & (d >= 0.985 * float(max_depth))
            valid_count = int(valid.sum())
            near_cap_count = int(near_cap.sum())
            if valid_count >= 64 and near_cap_count >= max(128, int(0.20 * valid_count)):
                valid &= d < 0.985 * float(max_depth)
                log(
                    f"[DepthPCL][WARN] Dropping far-cap saturated samples in frame '{frame_name}' "
                    f"({near_cap_count}/{valid_count} near max_depth={float(max_depth):.3f})."
                )
        # Guard against extreme per-frame depth outliers that create ghost planes.
        d_valid = d[valid]
        if d_valid.size >= 32:
            d_med = float(np.median(d_valid))
            d_p99 = float(np.percentile(d_valid, 99.0))
            if d_med <= 1e-6 or d_p99 > max(10.0, 30.0 * d_med):
                skipped_suspicious += 1
                log(
                    f"[DepthPCL][WARN] Skipping suspicious frame '{frame_name}' "
                    f"(aligned depth median={d_med:.6f}, p99={d_p99:.6f}, "
                    f"scale={float(scale):.6f}, offset={float(offset):.6f}, matches={int(match_count)})."
                )
                continue
        if not np.any(valid):
            continue

        yy = yy[valid]
        xx = xx[valid]
        d = d[valid]

        # COLMAP consistency check: drop dense depth points that disagree with nearby sparse z.
        if sparse_index is not None and bool(colmap_consistency):
            u_s, v_s, z_s, grid, cell_size = sparse_index
            r = float(max(1, int(colmap_consistency_radius_px)))
            rel_thr = float(max(0.01, colmap_consistency_rel_depth))

            cu = (xx // cell_size).astype(np.int32)
            cv = (yy // cell_size).astype(np.int32)
            keys = cu.astype(np.int64) * 1000000 + cv.astype(np.int64)
            order = np.argsort(keys)

            xx_o = xx[order]
            yy_o = yy[order]
            d_o = d[order]
            cu_o = cu[order]
            cv_o = cv[order]

            keep_o = np.zeros((xx_o.shape[0],), dtype=bool)
            start = 0
            while start < xx_o.shape[0]:
                end = start + 1
                while end < xx_o.shape[0] and cu_o[end] == cu_o[start] and cv_o[end] == cv_o[start]:
                    end += 1

                cell = (int(cu_o[start]), int(cv_o[start]))
                cand = []
                for du_cell in (-1, 0, 1):
                    for dv_cell in (-1, 0, 1):
                        cand.extend(grid.get((cell[0] + du_cell, cell[1] + dv_cell), []))

                if cand:
                    cand = np.asarray(cand, dtype=np.int32)
                    u_c = u_s[cand].astype(np.float32)
                    v_c = v_s[cand].astype(np.float32)
                    z_c = z_s[cand].astype(np.float32)

                    pts_x = xx_o[start:end].astype(np.float32)
                    pts_y = yy_o[start:end].astype(np.float32)
                    dx = pts_x[:, None] - u_c[None, :]
                    dy = pts_y[:, None] - v_c[None, :]
                    dist2 = dx * dx + dy * dy
                    nn = np.argmin(dist2, axis=1)
                    d2 = dist2[np.arange(dist2.shape[0]), nn]
                    within = d2 <= (r * r)
                    if within.any():
                        z_nn = z_c[nn]
                        rel = np.abs(d_o[start:end] - z_nn) / np.clip(z_nn, 1e-6, None)
                        keep_o[start:end] = within & (rel <= rel_thr)
                    else:
                        if not bool(effective_require_support):
                            keep_o[start:end] = True
                else:
                    if not bool(effective_require_support):
                        keep_o[start:end] = True

                start = end

            keep = np.zeros_like(keep_o)
            keep[order] = keep_o
            dropped = int((~keep).sum())
            if bool(effective_require_support):
                dropped_no_support += dropped
            else:
                dropped_inconsistent += dropped

            if not np.any(keep):
                continue
            yy = yy[keep]
            xx = xx[keep]
            d = d[keep]

        x_norm, y_norm = depth_utils.undistort_colmap_pixels(xx, yy, cam, sx=sx, sy=sy)
        x_cam = x_norm * d
        y_cam = y_norm * d
        z_cam = d
        cam_pts = np.stack([x_cam, y_cam, z_cam], axis=0)

        r = img.qvec2rotmat()
        t = img.tvec.reshape(3, 1)
        world_pts = (r.T @ (cam_pts - t)).T.astype(np.float32)
        colors = rgb[yy, xx, :].astype(np.uint8)

        all_xyz.append(world_pts)
        all_rgb.append(colors)
        if save_debug_clouds:
            per_frame_clouds.append((frame_name, world_pts, colors))
        used_frames += 1

    if not all_xyz:
        log("[WARNING] Depth point cloud export skipped: no valid 3D points generated.")
        return None

    xyz = np.concatenate(all_xyz, axis=0).astype(np.float32)
    rgb = np.concatenate(all_rgb, axis=0).astype(np.uint8)

    os.makedirs(output_dir, exist_ok=True)
    ply_path = os.path.join(output_dir, output_name)
    storePly(ply_path, xyz, rgb)

    if save_debug_clouds and per_frame_clouds:
        first_xyz = per_frame_clouds[0][1]
        first_rgb = per_frame_clouds[0][2]
        storePly(os.path.join(output_dir, "debug_single_frame_points.ply"), first_xyz, first_rgb)
        if len(per_frame_clouds) >= 2:
            two_xyz = np.concatenate([per_frame_clouds[0][1], per_frame_clouds[1][1]], axis=0).astype(np.float32)
            two_rgb = np.concatenate([per_frame_clouds[0][2], per_frame_clouds[1][2]], axis=0).astype(np.uint8)
            storePly(os.path.join(output_dir, "debug_two_frame_points.ply"), two_xyz, two_rgb)

        cam_centers = []
        for img in ordered_images:
            r = img.qvec2rotmat()
            t = img.tvec.reshape(3, 1)
            cam_centers.append((-r.T @ t).reshape(3))
        if cam_centers:
            cam_xyz = np.asarray(cam_centers, dtype=np.float32)
            cam_rgb = np.tile(np.array([[255, 0, 0]], dtype=np.uint8), (cam_xyz.shape[0], 1))
            storePly(os.path.join(output_dir, "debug_camera_centers.ply"), cam_xyz, cam_rgb)

    if used_scales:
        log(
            "[DepthPCL] Alignment stats: "
            f"scale median={np.median(used_scales):.6f}, "
            f"min={np.min(used_scales):.6f}, max={np.max(used_scales):.6f}; "
            f"offset median={np.median(used_offsets):.6f}"
        )
    if skipped_suspicious > 0:
        log(f"[DepthPCL][WARN] Skipped {skipped_suspicious} frame(s) due to unstable depth alignment.")
    if bool(colmap_consistency) and (dropped_inconsistent > 0 or dropped_no_support > 0):
        log(
            "[DepthPCL][Audit] COLMAP consistency filtering: "
            f"dropped_inconsistent={int(dropped_inconsistent)}, dropped_no_support={int(dropped_no_support)}, "
            f"radius_px={int(colmap_consistency_radius_px)}, rel_depth={float(colmap_consistency_rel_depth):.3f}, "
            f"require_support={bool(effective_require_support)}"
        )
    log(f"[INFO] Saved depth point cloud: {ply_path} ({xyz.shape[0]} points, {used_frames} frames, stride={stride})")
    if save_debug_clouds:
        log(
            "[INFO] Saved debug point clouds: "
            f"{os.path.join(output_dir, 'debug_single_frame_points.ply')}, "
            f"{os.path.join(output_dir, 'debug_two_frame_points.ply')}, "
            f"{os.path.join(output_dir, 'debug_camera_centers.ply')}"
        )
    return ply_path
