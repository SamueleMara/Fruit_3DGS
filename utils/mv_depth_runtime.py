import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.graphics_utils import fov2focal, getWorld2View2


def _import_multiview_depth_model(depthsplat_root: Path | None = None):
    try:
        from src.model.encoder.unimatch.mv_unimatch import MultiViewUniMatch
        return MultiViewUniMatch
    except Exception:
        if depthsplat_root is None:
            depthsplat_root = Path(__file__).resolve().parents[1] / "depthsplat"
        if depthsplat_root.exists():
            sys.path.insert(0, str(depthsplat_root))
        from src.model.encoder.unimatch.mv_unimatch import MultiViewUniMatch
        return MultiViewUniMatch


def _get_camera_c2w(cam) -> np.ndarray:
    w2c = getWorld2View2(cam.R, cam.T, cam.trans, cam.scale)
    return np.linalg.inv(w2c)


def _get_camera_intrinsics_normalized(cam) -> np.ndarray:
    fx = fov2focal(cam.FoVx, cam.image_width)
    fy = fov2focal(cam.FoVy, cam.image_height)
    cx = 0.5 * cam.image_width
    cy = 0.5 * cam.image_height
    K = np.array(
        [[fx, 0.0, cx],
         [0.0, fy, cy],
         [0.0, 0.0, 1.0]],
        dtype=np.float32
    )
    K[0, 0] /= float(cam.image_width)
    K[0, 2] /= float(cam.image_width)
    K[1, 1] /= float(cam.image_height)
    K[1, 2] /= float(cam.image_height)
    return K


def _compute_neighbor_indices(cameras, num_views: int) -> np.ndarray:
    centers = []
    for cam in cameras:
        centers.append(cam.camera_center.detach().cpu().numpy())
    centers = np.stack(centers, axis=0)
    diffs = centers[:, None, :] - centers[None, :, :]
    dists = np.linalg.norm(diffs, axis=-1)
    return np.argsort(dists, axis=1)[:, :num_views]


def _load_weights(model, weights_path: str | None, log=print):
    if not weights_path:
        log("[MVDepth][WARN] No weights provided; using random initialization.")
        return
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Multi-view depth weights not found: {weights_path}")
    state = torch.load(weights_path, map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        log(f"[MVDepth][WARN] Missing keys in weights: {len(missing)}")
    if unexpected:
        log(f"[MVDepth][WARN] Unexpected keys in weights: {len(unexpected)}")


@torch.no_grad()
def build_multiview_depth_priors(
    cameras,
    num_views: int = 3,
    weights_path: str | None = None,
    device: str | None = None,
    min_depth: float = 0.1,
    max_depth: float = 100.0,
    downscale: int = 1,
    cache_dir: str | None = None,
    use_fp16: bool = False,
    vit_type: str = "vits",
    num_scales: int = 1,
    upsample_factor: int = 4,
    lowest_feature_resolution: int = 4,
    depth_unet_channels: int = 128,
    num_depth_candidates: int = 128,
    grid_sample_disable_cudnn: bool = False,
    log=print,
):
    if cameras is None or len(cameras) == 0:
        log("[MVDepth][WARN] No cameras provided; skipping multi-view depth.")
        return None
    if num_views < 2:
        log("[MVDepth][WARN] num_views < 2; skipping multi-view depth.")
        return None

    num_views = min(num_views, len(cameras))
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if not isinstance(device, str):
        device = str(device)

    MultiViewUniMatch = _import_multiview_depth_model()
    model = MultiViewUniMatch(
        num_scales=num_scales,
        upsample_factor=upsample_factor,
        lowest_feature_resolution=lowest_feature_resolution,
        vit_type=vit_type,
        unet_channels=depth_unet_channels,
        num_depth_candidates=num_depth_candidates,
        grid_sample_disable_cudnn=grid_sample_disable_cudnn,
    )
    _load_weights(model, weights_path, log=log)
    model.to(device).eval()

    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)

    neighbor_indices = _compute_neighbor_indices(cameras, num_views)
    priors = [None] * len(cameras)

    min_depth = float(min_depth)
    max_depth = float(max_depth)
    if min_depth <= 0 or max_depth <= 0 or max_depth <= min_depth:
        raise ValueError(
            f"Invalid min/max depth: min_depth={min_depth}, max_depth={max_depth}"
        )
    min_inv = 1.0 / max_depth
    max_inv = 1.0 / min_depth

    for idx in tqdm(range(len(cameras)), desc="Building multi-view depth priors"):
        cam = cameras[idx]
        cache_path = None
        if cache_dir:
            cache_path = os.path.join(cache_dir, f"{Path(cam.image_name).stem}.npy")
            if os.path.exists(cache_path):
                depth = np.load(cache_path).astype(np.float32)
                priors[idx] = torch.from_numpy(depth[None])
                continue

        view_ids = neighbor_indices[idx].tolist()
        cams = [cameras[j] for j in view_ids]

        target_h = cams[0].image_height
        target_w = cams[0].image_width
        if downscale > 1:
            target_h = max(32, int(target_h // downscale))
            target_w = max(32, int(target_w // downscale))

        images = []
        intrinsics = []
        extrinsics = []
        for c in cams:
            img = c.original_image[:3].to(device)
            if img.shape[-2:] != (target_h, target_w):
                img = F.interpolate(
                    img.unsqueeze(0),
                    size=(target_h, target_w),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
            images.append(img)
            K = torch.from_numpy(_get_camera_intrinsics_normalized(c)).to(device)
            intrinsics.append(K)
            c2w = torch.from_numpy(_get_camera_c2w(c)).to(device)
            extrinsics.append(c2w)

        images = torch.stack(images, dim=0).unsqueeze(0)
        intrinsics = torch.stack(intrinsics, dim=0).unsqueeze(0)
        extrinsics = torch.stack(extrinsics, dim=0).unsqueeze(0)
        min_inv_t = torch.full((1, len(cams)), min_inv, device=device)
        max_inv_t = torch.full((1, len(cams)), max_inv, device=device)

        if use_fp16 and device.startswith("cuda"):
            with torch.cuda.amp.autocast(dtype=torch.float16):
                results = model(
                    images,
                    intrinsics=intrinsics,
                    extrinsics=extrinsics,
                    min_depth=min_inv_t,
                    max_depth=max_inv_t,
                )
        else:
            results = model(
                images,
                intrinsics=intrinsics,
                extrinsics=extrinsics,
                min_depth=min_inv_t,
                max_depth=max_inv_t,
            )

        depth = results["depth_preds"][-1][0, 0].float()
        if depth.shape[-2:] != (cam.image_height, cam.image_width):
            depth = F.interpolate(
                depth.unsqueeze(0).unsqueeze(0),
                size=(cam.image_height, cam.image_width),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0).squeeze(0)

        depth = depth.clamp(min=1e-6)
        priors[idx] = depth.detach().cpu().unsqueeze(0)
        if cache_path:
            np.save(cache_path, priors[idx].squeeze(0).numpy())

    return priors
