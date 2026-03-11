"""
Enhanced depth seeding with multi-view fusion support.

This script adds multi-view depth fusion capability to the training pipeline.
"""

import os
import numpy as np
from utils import depth_utils
from utils.depth_fusion import fuse_multi_view_depth
from utils.depth_seed_runtime import load_colmap_model_with_fallback
from scene.dataset_readers import fetchPly
from utils.graphics_utils import BasicPointCloud


def add_depth_seed_points_with_fusion(
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
    # NEW: Multi-view fusion parameters
    use_depth_fusion=True,
    fusion_overlap_thresh=0.5,
    fusion_consistency_thresh=0.01,
    fusion_min_views=2,
    fusion_stride=4,
):
    """
    Add depth seed points with optional multi-view fusion.
    
    Args:
        use_depth_fusion: Enable multi-view depth fusion
        fusion_overlap_thresh: Minimum overlap for neighbor selection
        fusion_consistency_thresh: Max 3D error for consistency (meters)
        fusion_min_views: Minimum views for fusion
        fusion_stride: Pixel sampling stride for fusion
    """
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

    _, cameras_colmap, images, points3D = load_colmap_model_with_fallback(
        dataset.source_path,
        log=print,
        context="DepthSeedFusion",
    )
    if cameras_colmap is None or images is None or points3D is None:
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

    # NEW: Multi-view depth fusion
    if use_depth_fusion:
        print("[INFO] Performing multi-view depth fusion...")
        
        # Load all depth maps
        depth_maps = {}
        instance_masks = {}
        
        train_cameras = scene.getTrainCameras()
        
        for cam in train_cameras:
            if hasattr(cam, 'invdepthmap') and cam.invdepthmap is not None:
                # Convert inverse depth to depth
                depth_map = 1.0 / (cam.invdepthmap + 1e-6)
                depth_map = depth_map.cpu().numpy() if hasattr(depth_map, 'cpu') else depth_map
                if depth_map.ndim == 3:
                    if depth_map.shape[0] == 1:
                        depth_map = depth_map[0]
                    else:
                        depth_map = depth_map[..., 0]
                depth_maps[cam.image_name] = depth_map.astype(np.float32)
                
                # Load instance masks if available
                mask_path = os.path.join(depth_seed_mask_dir, f"{cam.image_name}.png")
                if os.path.exists(mask_path):
                    import cv2
                    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    instance_masks[cam.image_name] = mask
        
        if len(depth_maps) < fusion_min_views:
            print(f"[WARNING] Not enough depth maps ({len(depth_maps)}) for fusion. Skipping fusion.")
            use_depth_fusion = False
        else:
            # Perform fusion
            fused_results = fuse_multi_view_depth(
                cameras=train_cameras,
                depth_maps=depth_maps,
                instance_masks=instance_masks if instance_masks else None,
                overlap_thresh=fusion_overlap_thresh,
                consistency_thresh=fusion_consistency_thresh,
                min_views=fusion_min_views,
                stride=fusion_stride,
                verbose=True
            )
            
            # Update camera depth maps with fused versions
            for cam in train_cameras:
                if cam.image_name in fused_results:
                    fused_depth, confidence = fused_results[cam.image_name]
                    
                    # Convert back to inverse depth
                    fused_invdepth = 1.0 / (fused_depth + 1e-6)
                    
                    # Update camera depth map
                    import torch
                    cam.invdepthmap = torch.from_numpy(fused_invdepth).float()
                    
                    # Store confidence for later use
                    cam.depth_confidence = torch.from_numpy(confidence).float()
                    
                    print(f"[INFO] Fused depth for {cam.image_name}: "
                          f"mean confidence = {confidence.mean():.3f}")
            
            print(f"[INFO] Multi-view depth fusion complete for {len(fused_results)} cameras")

    # Generate depth seed points (using fused depths if enabled)
    depth_xyz, depth_rgb, _ = depth_utils.generate_depth_seed_points(
        images=images,
        cameras=cameras_colmap,
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
    
    fusion_status = " (with multi-view fusion)" if use_depth_fusion else ""
    print(f"[INFO] Added {depth_xyz.shape[0]} depth seed points{fusion_status}. "
          f"Total points: {combined_pcd.points.shape[0]}")
    return True
