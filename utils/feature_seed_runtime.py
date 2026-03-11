import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import imageio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.depth_utils import undistort_colmap_pixels
from utils.depth_seed_runtime import load_colmap_model_with_fallback
from utils.graphics_utils import BasicPointCloud
from utils.masks_utils import list_masks_for_frame
from utils.read_write_model import Image, Point3D, qvec2rotmat, read_model, write_model
from utils.sh_utils import SH2RGB


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)


@dataclass
class _FrameFeatures:
    image_id: int
    image_name: str
    camera_id: int
    qvec: np.ndarray
    tvec: np.ndarray
    width: int
    height: int
    rgb: np.ndarray
    depth: np.ndarray
    uv: np.ndarray
    desc: np.ndarray


def _to_numpy_depth(depth):
    if hasattr(depth, "detach"):
        depth = depth.detach().cpu().numpy()
    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim == 3:
        if depth.shape[0] == 1:
            depth = depth[0]
        else:
            depth = depth[..., 0]
    return np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)


def _load_union_mask(mask_dir, frame_name, width, height):
    if mask_dir is None:
        return None
    mask_dir = Path(mask_dir)
    if not mask_dir.exists():
        return None

    exts = ("png", "jpg", "jpeg", "bmp", "tif", "tiff")
    single = None
    for ext in exts:
        p = mask_dir / f"{frame_name}.{ext}"
        if p.exists():
            single = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            break

    if single is not None:
        if single.shape[:2] != (height, width):
            single = cv2.resize(single, (width, height), interpolation=cv2.INTER_NEAREST)
        return single > 0

    instance_files = list_masks_for_frame(mask_dir, frame_name, log=lambda *a, **k: None)
    if not instance_files:
        return None

    merged = np.zeros((height, width), dtype=np.uint8)
    for p in instance_files:
        m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        if m.shape[:2] != (height, width):
            m = cv2.resize(m, (width, height), interpolation=cv2.INTER_NEAREST)
        merged = np.maximum(merged, m)
    return merged > 0


def _extract_current_point_cloud(scene):
    g = scene.gaussians
    xyz = g.get_xyz.detach().cpu().numpy().astype(np.float32)
    if xyz.shape[0] == 0:
        return xyz, np.zeros((0, 3), dtype=np.float32)

    try:
        fdc = g.get_features_dc.detach().cpu().numpy()
        if fdc.ndim == 3:
            fdc = fdc[:, 0, :]
        rgb = np.clip(SH2RGB(fdc), 0.0, 1.0).astype(np.float32)
    except Exception:
        rgb = np.full((xyz.shape[0], 3), 0.5, dtype=np.float32)
    return xyz, rgb


def _build_deep_backbone(backbone: str, device: torch.device) -> Optional[nn.Module]:
    try:
        import torchvision
    except Exception as exc:
        print(f"[WARNING] torchvision is required for deep feature seeds but is unavailable: {exc}")
        return None

    backbone = (backbone or "resnet18").lower()
    supported = {"resnet18", "resnet50"}
    if backbone not in supported:
        print(f"[WARNING] Unsupported deep feature backbone '{backbone}', falling back to resnet18.")
        backbone = "resnet18"

    try:
        if backbone == "resnet50":
            ctor = torchvision.models.resnet50
            weights_enum = getattr(torchvision.models, "ResNet50_Weights", None)
        else:
            ctor = torchvision.models.resnet18
            weights_enum = getattr(torchvision.models, "ResNet18_Weights", None)

        try:
            # torchvision>=0.13
            if weights_enum is not None:
                model = ctor(weights=weights_enum.DEFAULT)
            else:
                # torchvision<0.13 fallback path
                model = ctor(pretrained=True)
        except Exception as exc:
            print(f"[WARNING] Could not load pretrained {backbone} weights ({exc}); using random init.")
            try:
                model = ctor(weights=None)
            except TypeError:
                model = ctor(pretrained=False)
        trunk = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
        )
        trunk.eval().to(device)
        for p in trunk.parameters():
            p.requires_grad_(False)
        return trunk
    except Exception as exc:
        print(f"[WARNING] Failed to initialize deep backbone: {exc}")
        return None


def _extract_deep_keypoints_and_descriptors(
    rgb: np.ndarray,
    max_points: int,
    device: torch.device,
    backbone: nn.Module,
    mask: Optional[np.ndarray] = None,
    nms_kernel: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    max_points = max(1, int(max_points))
    img = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
    img = (img - IMAGENET_MEAN.to(device)) / IMAGENET_STD.to(device)

    with torch.no_grad():
        feat = backbone(img)  # [1, C, Hf, Wf]

    feat_raw = feat[0]
    desc = F.normalize(feat_raw, dim=0)
    saliency = feat_raw.norm(dim=0)  # [Hf, Wf]

    if mask is not None:
        mask_t = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
        mask_f = F.interpolate(mask_t, size=saliency.shape, mode="nearest")[0, 0] > 0.5
        saliency = torch.where(mask_f, saliency, torch.full_like(saliency, -1e6))

    nms_kernel = max(3, int(nms_kernel))
    if nms_kernel % 2 == 0:
        nms_kernel += 1
    pooled = F.max_pool2d(saliency.unsqueeze(0).unsqueeze(0), kernel_size=nms_kernel, stride=1, padding=nms_kernel // 2)[0, 0]
    keep = (saliency == pooled) & (saliency > -1e5)
    coords = torch.nonzero(keep, as_tuple=False)
    if coords.numel() == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0, desc.shape[0]), dtype=np.float32)

    scores = saliency[coords[:, 0], coords[:, 1]]
    k = min(max_points, int(coords.shape[0]))
    top_idx = torch.topk(scores, k=k, largest=True).indices
    coords = coords[top_idx]

    h, w = rgb.shape[:2]
    hf, wf = saliency.shape
    sx = float(w) / float(wf)
    sy = float(h) / float(hf)

    ys = coords[:, 0].float()
    xs = coords[:, 1].float()
    u = (xs + 0.5) * sx - 0.5
    v = (ys + 0.5) * sy - 0.5
    uv = torch.stack([u, v], dim=1).cpu().numpy().astype(np.float32)

    d = desc[:, coords[:, 0], coords[:, 1]].t().contiguous().cpu().numpy().astype(np.float32)
    d /= np.linalg.norm(d, axis=1, keepdims=True).clip(min=1e-8)
    return uv, d


def _match_mutual_nn(desc_a: np.ndarray, desc_b: np.ndarray, sim_thresh: float) -> Tuple[np.ndarray, np.ndarray]:
    if desc_a.shape[0] == 0 or desc_b.shape[0] == 0:
        return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.int64)

    sim = np.matmul(desc_a.astype(np.float32), desc_b.astype(np.float32).T)
    best_b = np.argmax(sim, axis=1)
    best_a = np.argmax(sim, axis=0)

    idx_a = np.arange(desc_a.shape[0], dtype=np.int64)
    mutual = best_a[best_b] == idx_a
    score_ok = sim[idx_a, best_b] >= float(sim_thresh)
    keep = mutual & score_ok

    if not np.any(keep):
        return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.int64)

    return idx_a[keep], best_b[keep].astype(np.int64)


def _backproject_world(uv: np.ndarray, depth: np.ndarray, cam, qvec: np.ndarray, tvec: np.ndarray) -> Optional[np.ndarray]:
    width = depth.shape[1]
    height = depth.shape[0]
    u = int(np.clip(np.round(float(uv[0])), 0, width - 1))
    v = int(np.clip(np.round(float(uv[1])), 0, height - 1))
    d = float(depth[v, u])
    if not np.isfinite(d) or d <= 1e-6:
        return None

    sx = width / float(cam.width)
    sy = height / float(cam.height)
    x_norm, y_norm = undistort_colmap_pixels(np.array([u]), np.array([v]), cam, sx=sx, sy=sy)
    x_cam = float(x_norm[0]) * d
    y_cam = float(y_norm[0]) * d
    cam_pt = np.array([x_cam, y_cam, d], dtype=np.float32)

    r = qvec2rotmat(qvec.astype(np.float64))
    t = tvec.reshape(3).astype(np.float32)
    world = (r.T @ (cam_pt - t)).astype(np.float32)
    return world


def _export_colmap_text_model(
    export_dir: Path,
    cameras: Dict[int, object],
    images: Dict[int, object],
    tracks: Dict[int, Dict[str, object]],
) -> bool:
    export_dir.mkdir(parents=True, exist_ok=True)

    obs_by_image: Dict[int, List[Tuple[float, float, int]]] = {iid: [] for iid in images.keys()}
    for pid, track in tracks.items():
        for obs in track["obs"]:
            obs_by_image[obs["image_id"]].append((obs["u"], obs["v"], pid))

    new_images: Dict[int, Image] = {}
    obs_index: Dict[Tuple[int, int], int] = {}
    for image_id, img in images.items():
        obs = obs_by_image.get(image_id, [])
        if len(obs) > 0:
            xys = np.array([[o[0], o[1]] for o in obs], dtype=np.float64)
            pids = np.array([o[2] for o in obs], dtype=np.int64)
            for k, (_, _, pid) in enumerate(obs):
                obs_index[(image_id, pid)] = k
        else:
            xys = np.empty((0, 2), dtype=np.float64)
            pids = np.empty((0,), dtype=np.int64)

        new_images[image_id] = Image(
            id=img.id,
            qvec=np.asarray(img.qvec, dtype=np.float64),
            tvec=np.asarray(img.tvec, dtype=np.float64),
            camera_id=img.camera_id,
            name=img.name,
            xys=xys,
            point3D_ids=pids,
        )

    new_points3d: Dict[int, Point3D] = {}
    for pid, track in tracks.items():
        image_ids = np.array([o["image_id"] for o in track["obs"]], dtype=np.int32)
        point2d_idxs = np.array([obs_index[(o["image_id"], pid)] for o in track["obs"]], dtype=np.int32)
        rgb = np.clip(np.asarray(track["rgb"], dtype=np.float32), 0.0, 255.0).astype(np.uint8)
        new_points3d[pid] = Point3D(
            id=int(pid),
            xyz=np.asarray(track["xyz"], dtype=np.float64),
            rgb=rgb,
            error=float(track["error"]),
            image_ids=image_ids,
            point2D_idxs=point2d_idxs,
        )

    write_model(cameras, new_images, new_points3d, str(export_dir), ext=".txt")
    return True


def add_feature_seed_points_from_maps(
    scene,
    dataset,
    depth_maps,
    depth_is_inverse=False,
    feature_mask_dir=None,
    feature_type="deep",
    feature_max_points_per_image=1200,
    feature_min_depth=0.0,
    feature_max_depth=0.0,
    feature_dedup_voxel=0.0,
    feature_pair_window=1,
    feature_match_sim_thresh=0.75,
    feature_match_depth_consistency=0.15,
    feature_deep_device=None,
    feature_deep_backbone="resnet18",
    feature_export_colmap=False,
    feature_export_colmap_dir=None,
):
    """
    Augment initialization by extracting deep local descriptors and matching
    them across nearby frames, then backprojecting the matched points to 3D
    with depth priors + COLMAP camera poses.
    """
    if depth_maps is None or len(depth_maps) == 0:
        print("[WARNING] Feature seeds skipped: no depth maps available.")
        return False

    if feature_type not in ("deep", "orb", "gftt"):
        print(f"[WARNING] Unsupported feature_seed_type='{feature_type}', using 'deep'.")
        feature_type = "deep"

    if feature_type != "deep":
        print("[WARNING] Only deep feature path is recommended. Forcing --feature_seed_type deep.")
        feature_type = "deep"

    _, cameras, images, _ = load_colmap_model_with_fallback(
        dataset.source_path,
        log=print,
        context="FeatureSeeds",
    )
    if cameras is None or images is None or len(images) == 0:
        print(f"[WARNING] Feature seeds skipped: failed to read COLMAP model under {dataset.source_path}")
        return False

    image_base_dir = Path(dataset.source_path) / dataset.images
    if not image_base_dir.exists():
        print(f"[WARNING] Feature seeds skipped: image dir not found: {image_base_dir}")
        return False

    if feature_mask_dir is not None and not os.path.isdir(feature_mask_dir):
        print(f"[WARNING] feature_seed_mask_dir not found: {feature_mask_dir}; using full images.")
        feature_mask_dir = None

    if feature_deep_device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(feature_deep_device)

    backbone = _build_deep_backbone(feature_deep_backbone, device=device)
    if backbone is None:
        return False

    frames: List[_FrameFeatures] = []
    ordered_images = sorted(images.values(), key=lambda x: x.id)
    for img in ordered_images:
        frame_name = Path(img.name).stem
        depth = depth_maps.get(frame_name)
        if depth is None:
            continue
        depth = _to_numpy_depth(depth)

        rgb_path = image_base_dir / img.name
        if not rgb_path.exists():
            continue

        rgb = imageio.imread(rgb_path)
        if rgb.ndim == 2:
            rgb = np.stack([rgb, rgb, rgb], axis=-1)
        if rgb.shape[2] > 3:
            rgb = rgb[..., :3]
        height, width = rgb.shape[:2]

        if depth.shape[:2] != (height, width):
            depth = cv2.resize(depth, (width, height), interpolation=cv2.INTER_LINEAR)

        if depth_is_inverse:
            depth = 1.0 / np.clip(depth, 1e-6, None)

        if feature_min_depth > 0:
            depth = np.where(depth >= float(feature_min_depth), depth, 0.0).astype(np.float32)
        if feature_max_depth > 0:
            depth = np.where(depth <= float(feature_max_depth), depth, 0.0).astype(np.float32)

        mask = _load_union_mask(feature_mask_dir, frame_name, width, height)
        uv, desc = _extract_deep_keypoints_and_descriptors(
            rgb=rgb,
            max_points=feature_max_points_per_image,
            device=device,
            backbone=backbone,
            mask=mask,
            nms_kernel=5,
        )
        if uv.shape[0] == 0:
            continue

        frames.append(
            _FrameFeatures(
                image_id=img.id,
                image_name=img.name,
                camera_id=img.camera_id,
                qvec=np.asarray(img.qvec, dtype=np.float64),
                tvec=np.asarray(img.tvec, dtype=np.float64),
                width=width,
                height=height,
                rgb=rgb,
                depth=depth.astype(np.float32),
                uv=uv.astype(np.float32),
                desc=desc.astype(np.float32),
            )
        )

    if len(frames) < 2:
        print("[WARNING] Feature seeds skipped: not enough frames with deep features.")
        return False

    tracks: Dict[int, Dict[str, object]] = {}
    feat_xyz = []
    feat_rgb = []
    point_id = 1
    pair_window = max(1, int(feature_pair_window))
    sim_thresh = float(feature_match_sim_thresh)
    depth_consistency = max(0.0, float(feature_match_depth_consistency))
    pair_count = 0
    match_count = 0

    for i in range(len(frames)):
        fa = frames[i]
        cam_a = cameras[fa.camera_id]
        for j in range(i + 1, min(len(frames), i + 1 + pair_window)):
            fb = frames[j]
            cam_b = cameras[fb.camera_id]
            ia, ib = _match_mutual_nn(fa.desc, fb.desc, sim_thresh=sim_thresh)
            if ia.shape[0] == 0:
                continue
            pair_count += 1

            for ka, kb in zip(ia.tolist(), ib.tolist()):
                uv_a = fa.uv[ka]
                uv_b = fb.uv[kb]
                xa = _backproject_world(uv_a, fa.depth, cam_a, fa.qvec, fa.tvec)
                xb = _backproject_world(uv_b, fb.depth, cam_b, fb.qvec, fb.tvec)
                if xa is None or xb is None:
                    continue

                da = float(np.linalg.norm(xa))
                db = float(np.linalg.norm(xb))
                dist = float(np.linalg.norm(xa - xb))
                max_d = max(da, db, 1e-6)
                if depth_consistency > 0.0 and dist > depth_consistency * max_d:
                    continue

                xyz = ((xa + xb) * 0.5).astype(np.float32)
                ua = int(np.clip(np.round(float(uv_a[0])), 0, fa.width - 1))
                va = int(np.clip(np.round(float(uv_a[1])), 0, fa.height - 1))
                rgb01 = (fa.rgb[va, ua, :].astype(np.float32) / 255.0).clip(0.0, 1.0)

                feat_xyz.append(xyz)
                feat_rgb.append(rgb01)
                match_count += 1

                tracks[point_id] = {
                    "xyz": xyz.astype(np.float64),
                    "rgb": (rgb01 * 255.0).astype(np.float32),
                    "error": dist,
                    "obs": [
                        {"image_id": fa.image_id, "u": float(uv_a[0]), "v": float(uv_a[1])},
                        {"image_id": fb.image_id, "u": float(uv_b[0]), "v": float(uv_b[1])},
                    ],
                }
                point_id += 1

    if len(feat_xyz) == 0:
        print("[WARNING] Feature seeds skipped: no deep matches survived geometry checks.")
        return False

    feat_xyz = np.asarray(feat_xyz, dtype=np.float32)
    feat_rgb = np.asarray(feat_rgb, dtype=np.float32)
    before_dedup = int(feat_xyz.shape[0])

    if feature_dedup_voxel and feature_dedup_voxel > 0:
        vox = np.floor(feat_xyz / float(feature_dedup_voxel)).astype(np.int64)
        _, unique_idx = np.unique(vox, axis=0, return_index=True)
        unique_idx = np.sort(unique_idx)
        feat_xyz = feat_xyz[unique_idx]
        feat_rgb = feat_rgb[unique_idx]

    base_xyz, base_rgb = _extract_current_point_cloud(scene)
    if base_xyz.shape[0] > 0:
        points = np.concatenate([base_xyz, feat_xyz], axis=0)
        colors = np.concatenate([base_rgb, feat_rgb], axis=0)
    else:
        points = feat_xyz
        colors = feat_rgb
    normals = np.zeros_like(points, dtype=np.float32)

    combined_pcd = BasicPointCloud(points=points, colors=colors, normals=normals)
    scene.gaussians.create_from_pcd(
        combined_pcd,
        scene.getTrainCameras(),
        scene.cameras_extent,
    )

    if feature_export_colmap:
        if feature_export_colmap_dir is None:
            feature_export_colmap_dir = os.path.join(dataset.model_path, "feature_seed_colmap")
        export_ok = _export_colmap_text_model(
            export_dir=Path(feature_export_colmap_dir),
            cameras=cameras,
            images=images,
            tracks=tracks,
        )
        if export_ok:
            print(f"[INFO] Exported deep feature COLMAP text model to: {feature_export_colmap_dir}")

    print(
        f"[INFO] Added {feat_xyz.shape[0]} deep feature seed points "
        f"(raw={before_dedup}, matched={match_count}, pairs={pair_count}, backbone={feature_deep_backbone}). "
        f"Total points: {points.shape[0]}"
    )
    return True
