"""
Multi-View Depth Fusion for 3D Gaussian Splatting

This module implements geometric consistency-based depth fusion across multiple views
to reduce noise and improve depth quality in occluded regions.

Author: Antigravity AI
Date: January 2026
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import cv2


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """
    Compute weighted median of values.
    
    Args:
        values: Array of values [N]
        weights: Array of weights [N]
    
    Returns:
        Weighted median value
    """
    if len(values) == 0:
        return 0.0
    
    if len(values) == 1:
        return values[0]
    
    # Sort by values
    sorted_indices = np.argsort(values)
    sorted_values = values[sorted_indices]
    sorted_weights = weights[sorted_indices]
    
    # Normalize weights
    sorted_weights = sorted_weights / sorted_weights.sum()
    
    # Find median
    cumsum = np.cumsum(sorted_weights)
    median_idx = np.searchsorted(cumsum, 0.5)
    
    return sorted_values[median_idx]


def unproject_pixel(u: int, v: int, depth: float, camera) -> np.ndarray:
    """
    Unproject a pixel to 3D using depth and camera parameters.
    
    Args:
        u, v: Pixel coordinates
        depth: Depth value at pixel
        camera: Camera object with intrinsics and extrinsics
    
    Returns:
        xyz: 3D point in world coordinates [3]
    """
    # Get camera intrinsics
    fx = camera.FoVx_to_focal(camera.FoVx, camera.image_width)
    fy = camera.FoVy_to_focal(camera.FoVy, camera.image_height)
    cx = camera.image_width / 2.0
    cy = camera.image_height / 2.0
    
    # Unproject to camera coordinates
    x_cam = (u - cx) * depth / fx
    y_cam = (v - cy) * depth / fy
    z_cam = depth
    
    xyz_cam = np.array([x_cam, y_cam, z_cam])
    
    # Transform to world coordinates
    # camera.world_view_transform is world-to-camera, so we need its inverse
    R = camera.R.T  # Transpose for camera-to-world rotation
    t = camera.T
    
    xyz_world = R @ xyz_cam - R @ t
    
    return xyz_world


def project_point(xyz_world: np.ndarray, camera) -> Tuple[int, int, float]:
    """
    Project a 3D point to pixel coordinates.
    
    Args:
        xyz_world: 3D point in world coordinates [3]
        camera: Camera object
    
    Returns:
        u, v: Pixel coordinates
        depth: Depth in camera frame
    """
    # Transform to camera coordinates
    R = camera.R
    t = camera.T
    
    xyz_cam = R @ xyz_world + t
    
    # Get camera intrinsics
    fx = camera.FoVx_to_focal(camera.FoVx, camera.image_width)
    fy = camera.FoVy_to_focal(camera.FoVy, camera.image_height)
    cx = camera.image_width / 2.0
    cy = camera.image_height / 2.0
    
    # Project to image
    depth = xyz_cam[2]
    
    if depth <= 0:
        return -1, -1, -1
    
    u = int(fx * xyz_cam[0] / depth + cx)
    v = int(fy * xyz_cam[1] / depth + cy)
    
    return u, v, depth


def in_bounds(u: int, v: int, height: int, width: int) -> bool:
    """Check if pixel coordinates are within image bounds."""
    return 0 <= u < width and 0 <= v < height


def find_overlapping_views(ref_camera, all_cameras, overlap_thresh: float = 0.5) -> List[int]:
    """
    Find cameras with significant overlap with reference camera.
    
    Args:
        ref_camera: Reference camera
        all_cameras: List of all cameras
        overlap_thresh: Minimum overlap ratio (0-1)
    
    Returns:
        List of camera indices with sufficient overlap
    """
    overlapping = []
    
    # Simple heuristic: cameras within angular threshold
    ref_center = -ref_camera.R.T @ ref_camera.T  # Camera center in world coords
    ref_direction = ref_camera.R[2, :]  # Camera viewing direction (z-axis)
    
    for idx, cam in enumerate(all_cameras):
        if cam == ref_camera:
            continue
        
        cam_center = -cam.R.T @ cam.T
        cam_direction = cam.R[2, :]
        
        # Check viewing direction similarity (dot product)
        direction_similarity = np.dot(ref_direction, cam_direction)
        
        # Check distance between camera centers
        distance = np.linalg.norm(ref_center - cam_center)
        
        # Heuristic: similar viewing direction and reasonable distance
        if direction_similarity > 0.5 and distance < 2.0:  # Adjust thresholds as needed
            overlapping.append(idx)
    
    return overlapping


def get_mask_pixels(mask: np.ndarray, stride: int = 1) -> List[Tuple[int, int]]:
    """
    Get pixel coordinates where mask is non-zero.
    
    Args:
        mask: Binary mask [H, W]
        stride: Sampling stride (default: 1 for all pixels)
    
    Returns:
        List of (u, v) pixel coordinates
    """
    coords = np.where(mask > 0)
    pixels = [(int(coords[1][i]), int(coords[0][i])) 
              for i in range(0, len(coords[0]), stride)]
    return pixels


def fuse_multi_view_depth(
    cameras: List,
    depth_maps: Dict[str, np.ndarray],
    instance_masks: Optional[Dict[str, np.ndarray]] = None,
    overlap_thresh: float = 0.5,
    consistency_thresh: float = 0.01,  # 1cm
    min_views: int = 2,
    stride: int = 4,
    verbose: bool = True
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Fuse depth from multiple views using geometric consistency.
    
    Algorithm:
    1. For each pixel in reference view:
       - Project to 3D using depth
       - Reproject to neighboring views
       - Check depth consistency
       - Fuse consistent depths (weighted by confidence)
    
    Args:
        cameras: List of camera objects
        depth_maps: Dict mapping camera image_name to depth map [H, W]
        instance_masks: Optional dict mapping camera image_name to mask [H, W]
        overlap_thresh: Minimum overlap ratio for neighbor selection
        consistency_thresh: Max 3D error for consistency (meters)
        min_views: Minimum views for fusion
        stride: Pixel sampling stride (for efficiency)
        verbose: Print progress
    
    Returns:
        Dict mapping camera image_name to (fused_depth, confidence)
        - fused_depth: Fused depth map [H, W]
        - confidence: Confidence map [H, W] (0-1, based on # consistent views)
    """
    fused_depths = {}
    
    # Create camera name to index mapping
    cam_name_to_idx = {cam.image_name: idx for idx, cam in enumerate(cameras)}
    
    iterator = tqdm(cameras, desc="Fusing depth maps") if verbose else cameras
    
    for ref_cam in iterator:
        ref_name = ref_cam.image_name
        
        if ref_name not in depth_maps:
            continue
        
        ref_depth = depth_maps[ref_name]
        if ref_depth.ndim == 3:
            if ref_depth.shape[0] == 1:
                ref_depth = ref_depth[0]
            else:
                ref_depth = ref_depth[..., 0]
            ref_depth = ref_depth.astype(np.float32)
        H, W = ref_depth.shape
        
        # Get mask for this view (if provided)
        if instance_masks is not None and ref_name in instance_masks:
            ref_mask = instance_masks[ref_name]
        else:
            ref_mask = np.ones((H, W), dtype=np.uint8) * 255
        
        # Find neighboring views
        neighbor_indices = find_overlapping_views(ref_cam, cameras, overlap_thresh)
        
        if len(neighbor_indices) < min_views - 1:
            # Not enough neighbors, use original depth
            fused_depths[ref_name] = (ref_depth.copy(), np.zeros((H, W)))
            continue
        
        # Initialize fused depth and confidence
        fused = np.zeros((H, W), dtype=np.float32)
        confidence = np.zeros((H, W), dtype=np.float32)
        
        # Get pixels to process
        pixels = get_mask_pixels(ref_mask, stride=stride)
        
        for u, v in pixels:
            ref_depth_val = ref_depth[v, u]
            
            if ref_depth_val <= 0:
                continue
            
            # Unproject to 3D
            try:
                xyz_ref = unproject_pixel(u, v, ref_depth_val, ref_cam)
            except:
                continue
            
            # Collect depth estimates from neighbors
            depth_estimates = [ref_depth_val]
            weights = [1.0]
            
            for neighbor_idx in neighbor_indices:
                neighbor_cam = cameras[neighbor_idx]
                neighbor_name = neighbor_cam.image_name
                
                if neighbor_name not in depth_maps:
                    continue
                
                neighbor_depth = depth_maps[neighbor_name]
                if neighbor_depth.ndim == 3:
                    if neighbor_depth.shape[0] == 1:
                        neighbor_depth = neighbor_depth[0]
                    else:
                        neighbor_depth = neighbor_depth[..., 0]
                    neighbor_depth = neighbor_depth.astype(np.float32)
                
                # Reproject to neighbor view
                try:
                    u_n, v_n, depth_n = project_point(xyz_ref, neighbor_cam)
                except:
                    continue
                
                if not in_bounds(u_n, v_n, neighbor_depth.shape[0], neighbor_depth.shape[1]):
                    continue
                
                # Get depth at reprojected location
                neighbor_depth_val = neighbor_depth[v_n, u_n]
                
                if neighbor_depth_val <= 0:
                    continue
                
                # Unproject neighbor depth
                try:
                    xyz_neighbor = unproject_pixel(u_n, v_n, neighbor_depth_val, neighbor_cam)
                except:
                    continue
                
                # Check geometric consistency
                error = np.linalg.norm(xyz_ref - xyz_neighbor)
                
                if error < consistency_thresh:
                    depth_estimates.append(neighbor_depth_val)
                    # Gaussian weighting based on error
                    weight = np.exp(-error / (consistency_thresh / 2.0))
                    weights.append(weight)
            
            # Fuse depths (weighted median for robustness)
            if len(depth_estimates) >= min_views:
                fused[v, u] = weighted_median(
                    np.array(depth_estimates), 
                    np.array(weights)
                )
                # Confidence based on number of consistent views
                confidence[v, u] = min(1.0, len(depth_estimates) / (len(neighbor_indices) + 1))
            else:
                # Not enough consistent views, use original
                fused[v, u] = ref_depth_val
                confidence[v, u] = 0.0
        
        # Interpolate missing values (where stride > 1)
        if stride > 1:
            # Create mask of valid pixels
            valid_mask = (fused > 0).astype(np.uint8)
            
            # Inpaint missing regions
            if valid_mask.sum() > 0:
                fused = cv2.inpaint(
                    fused.astype(np.float32),
                    (1 - valid_mask).astype(np.uint8),
                    inpaintRadius=stride * 2,
                    flags=cv2.INPAINT_NS
                )
                confidence = cv2.inpaint(
                    confidence.astype(np.float32),
                    (1 - valid_mask).astype(np.uint8),
                    inpaintRadius=stride * 2,
                    flags=cv2.INPAINT_NS
                )
        
        fused_depths[ref_name] = (fused, confidence)
    
    return fused_depths


def visualize_depth_fusion(
    original_depth: np.ndarray,
    fused_depth: np.ndarray,
    confidence: np.ndarray,
    save_path: Optional[str] = None
):
    """
    Visualize depth fusion results.
    
    Args:
        original_depth: Original depth map [H, W]
        fused_depth: Fused depth map [H, W]
        confidence: Confidence map [H, W]
        save_path: Optional path to save visualization
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Original depth
    im0 = axes[0, 0].imshow(original_depth, cmap='viridis')
    axes[0, 0].set_title('Original Depth')
    axes[0, 0].axis('off')
    plt.colorbar(im0, ax=axes[0, 0])
    
    # Fused depth
    im1 = axes[0, 1].imshow(fused_depth, cmap='viridis')
    axes[0, 1].set_title('Fused Depth')
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1])
    
    # Confidence
    im2 = axes[1, 0].imshow(confidence, cmap='hot', vmin=0, vmax=1)
    axes[1, 0].set_title('Fusion Confidence')
    axes[1, 0].axis('off')
    plt.colorbar(im2, ax=axes[1, 0])
    
    # Difference
    diff = np.abs(fused_depth - original_depth)
    im3 = axes[1, 1].imshow(diff, cmap='coolwarm')
    axes[1, 1].set_title('Absolute Difference')
    axes[1, 1].axis('off')
    plt.colorbar(im3, ax=axes[1, 1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()


if __name__ == "__main__":
    print("Multi-View Depth Fusion Module")
    print("Usage: Import and call fuse_multi_view_depth() with your cameras and depth maps")
