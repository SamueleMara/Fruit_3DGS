import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from gaussian_renderer import render


def filter_and_save(scene, mask_dir, iteration, K=2, semantic_threshold=0.3):
    """
    Filter Gaussians based on:
      1. Contribution maps (which splats contribute to white mask pixels)
      2. Learned semantic field (gaussian.semantic_mask > threshold)
    
    Saves filtered point cloud and per-camera images showing:
      - Binary mask
      - Contribution overlay
    """
    gaussians = scene.gaussians
    all_xyz = gaussians.get_xyz
    N = all_xyz.shape[0]
    device = all_xyz.device
    keep_mask = torch.zeros(N, dtype=torch.bool, device=device)

    cameras = scene.getTrainCameras()
    output_dir = os.path.join(scene.model_path, "point_cloud", f"iteration_{iteration}")
    vis_dir = os.path.join(output_dir, "filter_visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Dummy rendering pipeline
    class DummyPipe:
        debug = False
        antialiasing = False
        compute_cov3D_python = False
        convert_SHs_python = False

    dummy_pipe = DummyPipe()
    bg_color = torch.tensor([0, 0, 0], dtype=torch.float32, device=device)

    # Semantic mask
    if hasattr(gaussians, "semantic_mask") and isinstance(gaussians.semantic_mask, torch.Tensor):
        semantic_mask = torch.sigmoid(gaussians.semantic_mask.squeeze())
        print(f"[INFO] Semantic mask found (range: {semantic_mask.min().item():.3f}–{semantic_mask.max().item():.3f})")
    else:
        print("[WARN] No semantic_mask field found in Gaussians — using only 2D contribution filtering.")
        semantic_mask = torch.zeros(N, device=device)

    # Iterate over cameras
    for cam in tqdm(cameras, desc="Filtering cameras"):

        # ---- NEW: Use already-loaded mask ----
        if cam.mask is None:
            print(f"[WARN] No mask in camera {cam.image_name}, skipping.")
            continue

        # Ensure mask is on correct device & shape
        mask_tensor = cam.mask.squeeze().detach().cpu().numpy()
        H, W = mask_tensor.shape

        # Convert mask to boolean (white = True)
        mask_white = mask_tensor > 0.5

        render_out = render(
            cam,
            gaussians,
            dummy_pipe,
            bg_color,
            contrib=True,
            K=K
        )

        if not render_out or "contrib_indices" not in render_out or render_out["contrib_indices"] is None:
            print(f"[WARN] No contributor data for {cam.image_name}, skipping.")
            continue

        contrib_indices = render_out["contrib_indices"].detach().cpu().numpy().astype(np.int32)

        # Safety check: ensure sizes match
        if contrib_indices.shape[:2] != mask_white.shape:
            print(f"[WARN] Mismatch in mask size for {cam.image_name}, resizing.")
            mask_white = np.array(Image.fromarray(mask_white.astype(np.uint8)*255)
                                  .resize(contrib_indices.shape[1::-1], Image.NEAREST)) > 127

        white_y, white_x = np.where(mask_white)
        if len(white_y) == 0:
            continue

        # Gaussians contributing to masked pixels
        gauss_ids = contrib_indices[white_y, white_x, :].reshape(-1)
        gauss_ids = gauss_ids[gauss_ids >= 0]

        if len(gauss_ids) == 0:
            continue

        unique_ids = np.unique(gauss_ids)
        keep_mask[unique_ids] = True

        # Save visualization as before
        base = os.path.splitext(cam.image_name)[0]
        Image.fromarray(mask_white.astype(np.uint8) * 255).save(
            os.path.join(vis_dir, f"{base}_mask.png")
        )

        contrib_count = (contrib_indices >= 0).sum(axis=2)
        if contrib_count.max() > 0:
            contrib_norm = (contrib_count / contrib_count.max() * 255).astype(np.uint8)
        else:
            contrib_norm = contrib_count.astype(np.uint8)

        # Contribution overlay
        contrib_color = np.stack([contrib_norm]*3, axis=2)
        contrib_color[mask_white, :] = [255, 0, 0]  # red overlay

        Image.fromarray(contrib_color).save(
            os.path.join(vis_dir, f"{base}_contrib_overlay.png")
        )

    # Combine with semantic mask
    keep_mask &= (semantic_mask > semantic_threshold)

    kept_count = int(keep_mask.sum().item())
    print(f"[INFO] Keeping {kept_count}/{N} Gaussians after filtering")
    
    if kept_count == 0:
        print("[WARN] No Gaussians selected — saving original scene.")
        filtered_ply_path = os.path.join(output_dir, "scene_mask_filtered_renderer.ply")
        gaussians.save_ply(filtered_ply_path)
        return

    # Apply filtering to all per-Gaussian tensors
    for attr_name in list(vars(gaussians).keys()):
        attr = getattr(gaussians, attr_name)
        if isinstance(attr, torch.Tensor) and attr.shape[0] == keep_mask.shape[0]:
            setattr(gaussians, attr_name, attr[keep_mask])

    filtered_ply_path = os.path.join(output_dir, "scene_mask_filtered_renderer.ply")
    gaussians.save_ply(filtered_ply_path)
    print(f"[OK] Filtered scene saved to: {filtered_ply_path}")
    print(f"[OK] Saved mask and contribution images to: {vis_dir}")

