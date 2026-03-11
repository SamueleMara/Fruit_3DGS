"""
Advanced Depth Scaling Methods Implementation

Provides multiple robust scaling methods for depth map alignment with COLMAP reconstructions.
"""

import numpy as np
from typing import Tuple, Optional, Dict
from scipy.optimize import least_squares


def huber_scale(ratios: np.ndarray, k: float = 1.345, max_iters: int = 10) -> float:
    """
    Huber regression for robust scaling.
    
    Args:
        ratios: Array of depth ratios (ground_truth / predicted)
        k: Huber parameter (transition point, default 1.345 for ~95% asymptotic efficiency)
        max_iters: Maximum iterations for least squares
    
    Returns:
        Robust scale factor
    """
    if len(ratios) == 0:
        return 1.0
    
    valid = np.isfinite(ratios) & (ratios > 0)
    if not np.any(valid):
        return 1.0
    
    ratios = ratios[valid]
    
    def huber_loss(scale):
        residuals = ratios - scale
        loss = np.where(
            np.abs(residuals) <= k,
            0.5 * residuals**2,
            k * (np.abs(residuals) - 0.5*k)
        )
        return np.sqrt(loss)
    
    try:
        result = least_squares(
            huber_loss,
            x0=[np.median(ratios)],
            ftol=1e-8,
            max_nfev=max_iters
        )
        scale = float(result.x[0])
        if np.isfinite(scale) and scale > 0:
            return scale
    except Exception as e:
        print(f"[WARNING] Huber regression failed: {e}")
    
    return float(np.median(ratios))


def m_estimator_scale(ratios: np.ndarray, max_iters: int = 5, 
                      threshold_mult: float = 1.5, use_tukey: bool = True) -> float:
    """
    M-Estimator with iterative reweighting (Tukey's biweight by default).
    
    Args:
        ratios: Array of depth ratios
        max_iters: Number of reweighting iterations
        threshold_mult: Multiplier for threshold (1.5 = remove ~5% outliers per iteration)
        use_tukey: Use Tukey biweight (True) or Bisquare (False)
    
    Returns:
        Robust scale factor
    """
    if len(ratios) == 0:
        return 1.0
    
    valid = np.isfinite(ratios) & (ratios > 0)
    if not np.any(valid):
        return 1.0
    
    ratios = ratios[valid]
    scale = np.median(ratios)
    
    for iteration in range(max_iters):
        residuals = ratios - scale
        
        # Median Absolute Deviation (MAD) for robust std estimation
        mad = np.median(np.abs(residuals - np.median(residuals)))
        
        # Scale factor: 1.4826 converts MAD to standard deviation equivalent
        c = threshold_mult * 1.4826 * mad + 1e-8
        
        # Normalized residuals
        u = np.abs(residuals) / c
        
        if use_tukey:
            # Tukey's biweight: smooth down-weighting
            # weight = (1 - u^2)^2 for |u| < 1, else 0
            weights = np.where(u < 1, (1 - u**2)**2, 0)
        else:
            # Bisquare: similar but simpler
            # weight = (1 - u^2)^2 for |u| < 1, else 0
            weights = np.where(u < 1, (1 - u**2)**2, 0)
        
        # Weighted median
        if np.sum(weights) > 0:
            sorted_indices = np.argsort(ratios)
            cumsum = np.cumsum(weights[sorted_indices])
            if cumsum[-1] > 0:
                scale_new = ratios[sorted_indices[np.searchsorted(cumsum, cumsum[-1]/2)]]
                if np.isfinite(scale_new) and scale_new > 0:
                    scale = scale_new
    
    return float(scale) if np.isfinite(scale) and scale > 0 else 1.0


def ransac_scale(ratios: np.ndarray, thresh: float = 0.1, iters: int = 100,
                 min_inliers: int = 50, rng: Optional[np.random.Generator] = None) -> Tuple[Optional[float], int]:
    """
    RANSAC-based robust scaling.
    
    Args:
        ratios: Array of depth ratios
        thresh: Relative threshold for inlier (10% = 0.1)
        iters: Number of RANSAC iterations
        min_inliers: Minimum required inliers
        rng: Random number generator
    
    Returns:
        (scale, inlier_count)
    """
    if len(ratios) == 0:
        return None, 0
    
    valid = np.isfinite(ratios) & (ratios > 0)
    if not np.any(valid):
        return None, 0
    
    ratios = ratios[valid]
    
    rng = rng or np.random.default_rng()
    best_inliers = 0
    best_scale = None
    
    for _ in range(iters):
        # Sample random scale hypothesis
        idx = rng.integers(0, len(ratios))
        s = float(ratios[idx])
        
        if not np.isfinite(s) or s <= 0:
            continue
        
        # Check inliers with relative threshold
        tol = max(1e-8, thresh * s)
        inliers_mask = np.abs(ratios - s) <= tol
        inlier_count = int(inliers_mask.sum())
        
        if inlier_count > best_inliers:
            best_inliers = inlier_count
            # Use median of inliers as estimate
            best_scale = float(np.median(ratios[inliers_mask]))
            
            # Early exit if all inliers found
            if best_inliers >= min_inliers and best_inliers == len(ratios):
                break
    
    return best_scale, best_inliers


def ransac_then_m_estimator(ratios: np.ndarray, ransac_thresh: float = 0.1,
                            ransac_iters: int = 100, ransac_min_inliers: int = 50,
                            inlier_threshold: float = 0.2,
                            m_estimator_iters: int = 5) -> float:
    """
    Two-stage robust scaling: RANSAC for outlier rejection, then M-estimator on inliers.
    
    This is the best overall method combining speed and robustness.
    
    Args:
        ratios: Array of depth ratios
        ransac_thresh: RANSAC inlier threshold
        ransac_iters: RANSAC iterations
        ransac_min_inliers: Minimum RANSAC inliers
        inlier_threshold: Relative threshold for extracting inliers (0.2 = 20%)
        m_estimator_iters: M-estimator iterations
    
    Returns:
        Robust scale factor
    """
    if len(ratios) == 0:
        return 1.0
    
    # Stage 1: RANSAC to get rough estimate
    scale_ransac, inlier_count = ransac_scale(
        ratios,
        thresh=ransac_thresh,
        iters=ransac_iters,
        min_inliers=ransac_min_inliers
    )
    
    if scale_ransac is None:
        return float(np.median(ratios))
    
    # Stage 2: Extract inliers and refine with M-estimator
    inlier_mask = np.abs(ratios - scale_ransac) < inlier_threshold * scale_ransac
    inlier_ratios = ratios[inlier_mask]
    
    if len(inlier_ratios) >= 10:
        # Refine estimate using M-estimator on inliers only
        scale_refined = m_estimator_scale(inlier_ratios, max_iters=m_estimator_iters)
        return scale_refined
    else:
        return scale_ransac


def depth_weighted_scale(gt_depth: np.ndarray, pred_depth: np.ndarray,
                        weight_by_depth: bool = True,
                        mask: Optional[np.ndarray] = None) -> float:
    """
    Weighted scaling prioritizing certain depth ranges.
    
    Args:
        gt_depth: Ground truth depth values
        pred_depth: Predicted depth values
        weight_by_depth: If True, weight by GT depth magnitude; if False, uniform weighting
        mask: Optional binary mask for valid pixels
    
    Returns:
        Weighted scale factor
    """
    if mask is not None:
        valid = mask & np.isfinite(gt_depth) & np.isfinite(pred_depth)
    else:
        valid = np.isfinite(gt_depth) & np.isfinite(pred_depth)
    
    gt = gt_depth[valid]
    pred = pred_depth[valid]
    
    if len(gt) == 0:
        return 1.0
    
    # Avoid division by zero
    pred = np.maximum(pred, 1e-6)
    
    ratios = gt / pred
    valid_ratios = ratios[np.isfinite(ratios) & (ratios > 0)]
    
    if len(valid_ratios) == 0:
        return 1.0
    
    if weight_by_depth:
        # Weight by ground truth depth (more weight to far points)
        weights = gt[np.isfinite(ratios) & (ratios > 0)]
        weights = weights / np.mean(weights)
    else:
        weights = np.ones_like(valid_ratios)
    
    scale = np.sum(weights * valid_ratios) / np.sum(weights)
    
    return float(scale) if np.isfinite(scale) and scale > 0 else 1.0


def per_view_scales(scales_dict: Dict[int, float]) -> Tuple[float, Dict[int, float]]:
    """
    Aggregate per-view scales robustly.
    
    Args:
        scales_dict: Dictionary mapping view_id to scale factor
    
    Returns:
        (global_scale, filtered_scales)
    """
    scales = np.array([s for s in scales_dict.values() if 0.1 < s < 100])
    
    if len(scales) == 0:
        return 1.0, {}
    
    # Use median with outlier removal
    median_scale = np.median(scales)
    mad = np.median(np.abs(scales - median_scale))
    
    # Keep scales within 3*MAD of median
    outlier_threshold = 3 * max(1e-8, 1.4826 * mad)
    filtered = scales[np.abs(scales - median_scale) <= outlier_threshold]
    
    global_scale = np.median(filtered) if len(filtered) > 0 else median_scale
    
    # Create filtered dict
    filtered_dict = {
        k: v for k, v in scales_dict.items()
        if np.abs(v - median_scale) <= outlier_threshold
    }
    
    return float(global_scale), filtered_dict


def compute_scale_advanced(image_id: int, ratios: np.ndarray,
                          method: str = "ransac_robust",
                          verbose: bool = False,
                          **kwargs) -> float:
    """
    Main entry point for advanced depth scaling.
    
    Args:
        image_id: Image identifier (for logging)
        ratios: Array of depth ratios (ground_truth / predicted)
        method: Scaling method to use
            - "median": Simple median (baseline)
            - "mean": Simple mean
            - "ransac": RANSAC only
            - "huber": Huber regression
            - "m_estimator": M-estimator only
            - "ransac_robust": RANSAC + M-estimator (BEST)
        verbose: Print debug info
        **kwargs: Additional parameters for specific methods
    
    Returns:
        Scale factor
    """
    valid = np.isfinite(ratios) & (ratios > 0)
    if not np.any(valid):
        if verbose:
            print(f"[Image {image_id}] No valid ratios")
        return 1.0
    
    ratios = ratios[valid]
    n_valid = len(ratios)
    
    if method == "median":
        scale = float(np.median(ratios))
    
    elif method == "mean":
        scale = float(np.mean(ratios))
    
    elif method == "ransac":
        scale, inliers = ransac_scale(ratios, **kwargs)
        if scale is None:
            scale = float(np.median(ratios))
        if verbose:
            print(f"[Image {image_id}] RANSAC: scale={scale:.4f}, inliers={inliers}/{n_valid}")
    
    elif method == "huber":
        scale = huber_scale(ratios, **kwargs)
        if verbose:
            print(f"[Image {image_id}] Huber: scale={scale:.4f}")
    
    elif method == "m_estimator":
        scale = m_estimator_scale(ratios, **kwargs)
        if verbose:
            print(f"[Image {image_id}] M-Estimator: scale={scale:.4f}")
    
    elif method == "ransac_robust":
        scale = ransac_then_m_estimator(ratios, **kwargs)
        if verbose:
            print(f"[Image {image_id}] RANSAC+M-Est: scale={scale:.4f}")
    
    else:
        raise ValueError(f"Unknown scaling method: {method}")
    
    if not np.isfinite(scale) or scale <= 0:
        return 1.0
    
    return scale


# Presets for common scenarios
SCALING_PRESETS = {
    "default": {
        "method": "ransac_robust",
        "ransac_thresh": 0.1,
        "ransac_iters": 100,
        "ransac_min_inliers": 50,
    },
    "aggressive": {
        "method": "ransac_robust",
        "ransac_thresh": 0.05,  # More aggressive outlier rejection
        "ransac_iters": 200,
        "ransac_min_inliers": 100,
        "m_estimator_iters": 10,
    },
    "conservative": {
        "method": "ransac",  # Less aggressive
        "ransac_thresh": 0.2,
        "ransac_iters": 50,
        "ransac_min_inliers": 20,
    },
    "fast": {
        "method": "median",  # Fastest but least robust
    },
    "robust": {
        "method": "m_estimator",
        "max_iters": 10,
        "threshold_mult": 2.0,
    },
}


if __name__ == "__main__":
    # Test the implementations
    print("Testing depth scaling methods...\n")
    
    # Create synthetic test data with outliers
    np.random.seed(42)
    true_scale = 2.5
    ratios = true_scale + np.random.normal(0, 0.15, 100)  # Gaussian noise
    
    # Add outliers (20%)
    outlier_indices = np.random.choice(100, 20, replace=False)
    ratios[outlier_indices] = np.random.uniform(0.5, 5.0, 20)
    
    print(f"True scale: {true_scale}")
    print(f"Data: mean={np.mean(ratios):.4f}, std={np.std(ratios):.4f}\n")
    
    # Test methods
    methods = ["median", "ransac", "huber", "m_estimator", "ransac_robust"]
    
    for method in methods:
        scale = compute_scale_advanced(0, ratios, method=method, verbose=True)
        error = abs(scale - true_scale) / true_scale * 100
        print(f"  Error: {error:.2f}%\n")
