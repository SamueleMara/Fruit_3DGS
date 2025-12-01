#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import random
import json
import numpy as np
import math
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from utils.read_write_model import read_model
from utils.cluster_utils import compute_mask_centroids,refine_clusters_with_metric
from utils.visualize_clusters import visualize_colmap_clusters
from utils import masks_utils
from scene.colmap_masker import ColmapMaskFilter

from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement

from pathlib import Path
from tqdm import tqdm
import glob, json
from collections import defaultdict
import torch
from PIL import Image

from scipy.spatial import cKDTree
from sklearn.neighbors import KDTree
from skopt import gp_minimize

HAVE_CUML = False
try:
    from cuml.cluster import DBSCAN as cuDBSCAN
    HAVE_CUML = True
except Exception:
    HAVE_CUML = False

try:
    from scipy.spatial import cKDTree as KDTree
    from sklearn.neighbors import KDTree
    HAVE_KDTREE = True
except Exception:
    KDTree = None
    HAVE_KDTREE = False



class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0], mask_dir=None):
        """
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.depths, args.eval, args.train_test_exp)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.depths, args.eval)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(
                scene_info.train_cameras, resolution_scale, args, scene_info.is_nerf_synthetic, False, mask_dir=mask_dir
            )
            gs_camera_list = self.train_cameras[resolution_scale] 
            self.gs_cameras = {Path(cam.image_name).stem: cam for cam in gs_camera_list}
            
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(
                scene_info.test_cameras, resolution_scale, args, scene_info.is_nerf_synthetic, True, mask_dir=mask_dir
            )

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path, 
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"), args.train_test_exp)
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, scene_info.train_cameras, self.cameras_extent)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        exposure_dict = {
            image_name: self.gaussians.get_exposure_from_name(image_name).detach().cpu().numpy().tolist()
            for image_name in self.gaussians.exposure_mapping
        }

        with open(os.path.join(self.model_path, "exposure.json"), "w") as f:
            json.dump(exposure_dict, f, indent=2)

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]

    # --------------------------------------------------------------------------
    # NEW FUNCTION: load both trained model and COLMAP-seed Gaussian model
    # --------------------------------------------------------------------------
    def load_with_colmap_seed(self, args: ModelParams, load_iteration=None, mask_dir=None,num_its_BO=20,bo_optimize=False):
        """
        Loads both:
        - The trained Gaussian model (from .ply)
        - A new Gaussian model created from COLMAP point cloud as cluster seed
        with instance-aware clustering using mask instances.

        Returns:
            trained_gaussians: GaussianModel or None if not found
            seed_gaussians: GaussianModel created from COLMAP point cloud
            scene_info: metadata (train/test cameras, normalization, etc.)
        """
        model_path = self.model_path
        source_path = args.source_path

        # 1. Determine which iteration to load
        if load_iteration is not None and load_iteration == -1:
            load_iteration = searchForMaxIteration(os.path.join(model_path, "point_cloud"))
            print(f"[Scene] Using trained model from iteration {load_iteration}")

        # 2. Load COLMAP or Blender dataset
        if os.path.exists(os.path.join(source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](
                source_path, args.images, args.depths, args.eval, args.train_test_exp)
        elif os.path.exists(os.path.join(source_path, "transforms_train.json")):
            print("[Scene] Found Blender transforms JSON — loading Blender dataset...")
            scene_info = sceneLoadTypeCallbacks["Blender"](
                source_path, args.white_background, args.depths, args.eval)
        else:
            raise RuntimeError("Could not recognize scene type: no sparse/ or transforms_train.json found.")

        # 3. Load COLMAP model if needed and build 3D→2D mapping
        if not hasattr(self, "point_to_pixels"):
            if not hasattr(self, "points3D") or not hasattr(self, "images"):
                self.colmap_model_dir = os.path.join(source_path, "sparse", "0")
                self.load_colmap()
            self.build_point_pixel_mapping()

        # 4. Bayesian Optimization for clustering parameters
        if mask_dir is None:
            raise RuntimeError("[Scene] mask_dir must be provided for instance-aware clustering")

        if bo_optimize:
            print("[Scene] Running Bayesian Optimization for instance clustering...")

            # Precompute masks / overlaps / propagation ONCE
            print("[Scene] Precomputing mask mappings and overlaps for BO (once)...")
            mask_instances, point_to_masks, mask_to_points = masks_utils.compute_full_point_to_mask_instance_mapping(
                self.points3D, self.images, mask_dir
            )
            overlaps = masks_utils.compute_mask_overlaps(point_to_masks, mask_to_points, min_shared=1, top_k=10**9,
                                                        log=(print if False else (lambda *a, **k: None)))
            multi_view_links = masks_utils.propagate_all_masks_gpu(
                points3D=self.points3D,
                images=self.images,
                mask_instances=mask_instances,
                gs_cameras=self.gs_cameras,
                device="cuda"
            )
            precomputed = {
                "mask_instances": mask_instances,
                "point_to_masks": point_to_masks,
                "mask_to_points": mask_to_points,
                "overlaps": overlaps,
                "multi_view_links": multi_view_links
            }

            # Wrap the BO objective with tqdm
            progress_bar = tqdm(total=num_its_BO, desc="[Scene] BO iterations", leave=True)

            def bo_objective(params):
                jaccard_threshold, spatial_weight = params
                clusters = self.build_instance_seed_clusters(
                    mask_dir,
                    device="cuda",
                    jaccard_threshold=jaccard_threshold,
                    spatial_consistency_weight=spatial_weight,
                    refine_iters=1,
                    verbose=False,
                    precomputed=precomputed,
                    reuse_precomputed=True
                )

                def get_xyz(pid):
                    return np.asarray(self.points3D[pid].xyz, dtype=np.float32)[:3]

                pids = [pid for pid in clusters.keys() if clusters[pid] != -1]
                if len(pids) == 0:
                    coherence = 1.0
                else:
                    X = np.vstack([get_xyz(pid) for pid in pids]).astype(np.float32)
                    labels = np.array([clusters[pid] for pid in pids], dtype=int)
                    from sklearn.neighbors import NearestNeighbors
                    n_neighbors = min(8 + 1, X.shape[0])
                    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='kd_tree').fit(X)
                    _, indices = nbrs.kneighbors(X)
                    coherences = []
                    for i, neigh_idx in enumerate(indices):
                        neigh_idx = neigh_idx[1:]
                        same = (labels[neigh_idx] == labels[i])
                        coherences.append(np.mean(same))
                    coherence = float(np.mean(coherences))

                # Geometric compactness penalty (avg cluster radius)
                clusters_xyz = defaultdict(list)
                pid_to_idx = {pid:i for i,pid in enumerate(pids)}
                for pid, cid in clusters.items():
                    if cid == -1: continue
                    clusters_xyz[cid].append(X[pid_to_idx[pid]])
                if len(clusters_xyz) == 0:
                    avg_radius = 0.0
                else:
                    avg_radius = np.mean([np.mean(np.linalg.norm(np.stack(pts) - np.mean(np.stack(pts), axis=0), axis=1)) for pts in clusters_xyz.values()])

                score = coherence  # or coherence - 0.2*avg_radius
                progress_bar.update(1)  # update tqdm bar for each BO iteration
                return -score

            res = gp_minimize(
                bo_objective,
                dimensions=[(0.01, 0.1), (0.5, 2.0)],  # jaccard_threshold, spatial_weight
                n_calls=num_its_BO,
                random_state=42
            )

            progress_bar.close()

            best_jaccard, best_spatial = res.x
            print(f"[Scene] BO result: jaccard_threshold={best_jaccard:.3f}, spatial_weight={best_spatial:.3f}")

            # Build clusters with best parameters
            point_clusters = self.build_instance_seed_clusters(
                mask_dir,
                device="cuda",
                jaccard_threshold=best_jaccard,
                spatial_consistency_weight=best_spatial,
                refine_iters=100,
                verbose=True
            )
        else:
            point_clusters = self.build_instance_seed_clusters(mask_dir)

        self.point_clusters = point_clusters
        # print(f"[DEBUG] COLMAP clusters: {point_clusters}")

        # 5. Create GaussianModel from COLMAP point cloud
        print("[Scene] Creating GaussianModel from COLMAP point cloud (cluster seed)...")
        seed_gaussians = GaussianModel(sh_degree=args.sh_degree)
        seed_gaussians.create_from_pcd(
            scene_info.point_cloud,
            scene_info.train_cameras,
            scene_info.nerf_normalization["radius"]
        )

        # 6. Map COLMAP points to corresponding Gaussians
        # print("[DEBUG] Mapping COLMAP point IDs to Gaussian indices...")

        colmap_pids = list(point_clusters.keys())  # point_clusters computed earlier
        colmap_xyz = torch.stack([
            torch.tensor(self.points3D[pid].xyz, dtype=torch.float32)
            for pid in colmap_pids
        ], dim=0).to(seed_gaussians._xyz.device)  # [M,3]

        colmap_cids = torch.tensor([point_clusters[pid] for pid in colmap_pids],
                                dtype=torch.long, device=seed_gaussians._xyz.device)  # [M]

        gaussian_xyz = seed_gaussians._xyz  # [N,3]

        # Compute squared distances [N, M]
        dists = torch.cdist(gaussian_xyz, colmap_xyz, p=2)  # [N, M]

        # Find nearest COLMAP point for each Gaussian
        nearest_idx = torch.argmin(dists, dim=1)  # [N]

        # Assign Gaussian cluster IDs
        gaussian_clusters = colmap_cids[nearest_idx]  # [N]

        # Attach cluster IDs to Gaussian model
        seed_gaussians.cluster_ids = gaussian_clusters
        self.seed_gaussian_clusters = gaussian_clusters

        # Debug info
        num_assigned = (gaussian_clusters >= 0).sum().item()
        num_unique = len(torch.unique(gaussian_clusters))
        print(f"[Scene] Assigned all {gaussian_xyz.shape[0]} Gaussians using nearest COLMAP point")
        print(f"[Scene] Total {num_unique} unique clusters: {torch.unique(gaussian_clusters).tolist()}")

        # 7. Load trained Gaussian model if available
        trained_gaussians = None
        if load_iteration is not None:
            trained_model_seg_path = os.path.join(
                model_path, "point_cloud", f"iteration_{load_iteration}", "scene_semantics_filtered.ply"
            )
            if os.path.exists(trained_model_seg_path):
                print(f"[Scene] Loading trained Gaussian model from {trained_model_seg_path}")
                trained_gaussians = GaussianModel(sh_degree=args.sh_degree)
                trained_gaussians.load_ply(trained_model_seg_path, args.train_test_exp)

        return trained_gaussians, seed_gaussians, scene_info

    # --------------------------------------------------------------------------
    # NEW FUNCTION: save the clustered ply with an additional cluster_id field
    # --------------------------------------------------------------------------
    def save_clustered_ply(self, path, gaussians, cluster_ids=None):
        """
            Save Gaussian points as PLY, optionally including cluster IDs for visualization.

            Args:
                path (str): output PLY path
                gaussians (GaussianModel): Gaussian model to save
                cluster_ids (Tensor[N], optional): cluster assignment per Gaussian
            """
        mkdir_p(os.path.dirname(path))
        xyz = gaussians._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = gaussians._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = gaussians._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = gaussians._opacity.detach().cpu().numpy()
        scale = gaussians._scaling.detach().cpu().numpy()
        rotation = gaussians._rotation.detach().cpu().numpy()

        semantics = None
        if gaussians.semantic_mask is not None:
            semantics = gaussians.semantic_mask.detach().cpu().numpy()[..., None]

        dtype_full = [(attribute, 'f4') for attribute in gaussians.construct_list_of_attributes()]
        if cluster_ids is not None:
            dtype_full.append(('cluster_id', 'i4'))

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attrs = [xyz, normals, f_dc, f_rest, opacities, scale, rotation]
        attributes = np.concatenate([a if a.ndim == 2 else a.reshape(a.shape[0], -1) for a in attrs], axis=1)
        if semantics is not None:
            attributes = np.concatenate((attributes, semantics), axis=1)
        if cluster_ids is not None:
            cluster_np = cluster_ids.detach().cpu().numpy()[..., None]
            attributes = np.concatenate((attributes, cluster_np), axis=1)

        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el], text=True).write(path)

        print(f"[OK] Clustered PLY saved at {path}")

    # ---------------------------------
    # NEW FUNCTION: load colmap model
    # ---------------------------------
    def load_colmap(self):
        """
        Load COLMAP model (cameras, images, and points3D) from self.colmap_model_dir.

        This wraps utils.read_write_model.read_model() and caches the loaded structures
        for later geometric reasoning, instance association, or cluster seeding.

        Expected directory structure:
            self.colmap_model_dir/
                cameras.txt
                images.txt
                points3D.txt

        Side effects:
            - Populates self.cameras, self.images, self.points3D
            - Resets camera graph and masks cache
        """
        print(f"[INFO] Loading COLMAP model from: {self.colmap_model_dir}")

        try:
            self.cameras, self.images, self.points3D = read_model(self.colmap_model_dir, ext=".txt")
        except Exception as e:
            print(f"[ERROR] Failed to load COLMAP model: {e}")
            raise

        print(f"[OK] Loaded {len(self.cameras)} cameras, {len(self.images)} images, {len(self.points3D)} 3D points")

        # Reset caches
        self.camera_graph = None
        self.camera_centers = {}
        if hasattr(self, "_masks_cache"):
            self._masks_cache.clear()

        return self.cameras, self.images, self.points3D

    # ---------------------------------
    # NEW FUNCTION: build 3D→2D point-pixel mapping
    # ---------------------------------
    def build_point_pixel_mapping(self, save_path=None):
        """
        Vectorized version: For each 3D COLMAP point, compute all 2D pixel coordinates in images where it is observed.

        Returns:
            dict[int, list[dict]]: Mapping point_id -> list of observations:
                {"image_id": int, "image_name": str, "xy": [float, float]}
        """

        if not hasattr(self, "points3D") or not hasattr(self, "images"):
            raise RuntimeError("[Scene] COLMAP model not loaded. Call `load_colmap()` first.")

        point_ids = []
        image_ids = []
        point2d_idxs = []

        # Collect all point observations
        for pid, pt in self.points3D.items():
            n_obs = len(pt.image_ids)
            if n_obs == 0:
                continue
            point_ids.append(np.full(n_obs, pid, dtype=int))
            image_ids.append(pt.image_ids)
            point2d_idxs.append(pt.point2D_idxs)

        if not point_ids:
            print("[Scene] No point observations found.")
            return {}

        point_ids = np.concatenate(point_ids)
        image_ids = np.concatenate(image_ids)
        point2d_idxs = np.concatenate(point2d_idxs)

        # Gather corresponding image names and pixel coordinates
        all_image_names = np.array([self.images[i].name for i in image_ids])
        all_xys = np.array([self.images[i].xys[idx] for i, idx in zip(image_ids, point2d_idxs)])

        # Build mapping dictionary
        point_to_pixels = {}
        for pid, img_name, xy, img_id in zip(point_ids, all_image_names, all_xys, image_ids):
            if pid not in point_to_pixels:
                point_to_pixels[pid] = []
            point_to_pixels[pid].append({
                "image_id": int(img_id),
                "image_name": img_name,
                "xy": xy.tolist()
            })

        print(f"[Scene] Built vectorized 3D→2D mapping for {len(point_to_pixels)} points")

        # Optionally save to JSON
        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "w") as f:
                json.dump(point_to_pixels, f, indent=2)
            print(f"[Scene] Saved 3D→2D mapping to {save_path}")

        self.point_to_pixels = point_to_pixels
        return point_to_pixels
        
    # ------------------------------------------------------------
    # NEW FUNCTION: build instance-aware COLMAP seed clusters with BO-tunable params
    # ------------------------------------------------------------
    # ---------- build_instance_seed_clusters (updated) ----------
    def build_instance_seed_clusters(
        self,
        mask_dir,
        device="cuda",
        jaccard_threshold=0.01,
        spatial_consistency_weight=0.5,
        min_shared=1,
        refine_iters=50,
        verbose=False,
        precomputed=None,
        reuse_precomputed=True
    ):
        """
        Build global instance clusters. If `precomputed` dict is provided and reuse_precomputed=True,
        expensive mappings/overlaps/propagation will be re-used (required for efficient BO).
        """
        # ---------- Step 0: mask-point mappings (cache-aware) ----------
        if precomputed is not None and reuse_precomputed:
            mask_instances = precomputed.get("mask_instances", None)
            point_to_masks = precomputed.get("point_to_masks", None)
            mask_to_points = precomputed.get("mask_to_points", None)
            overlaps = precomputed.get("overlaps", None)
            multi_view_links = precomputed.get("multi_view_links", None)
            if verbose:
                print("[Scene] Using precomputed mask mappings/overlaps.")
        else:
            mask_instances, point_to_masks, mask_to_points = masks_utils.compute_full_point_to_mask_instance_mapping(
                self.points3D, self.images, mask_dir
            )
            overlaps = None
            multi_view_links = None

        self.mask_instances = mask_instances
        self._cached_point_to_masks = point_to_masks
        self._cached_mask_to_points = mask_to_points

        # ---------- Step 1: overlaps (cache-aware) ----------
        if overlaps is None:
            overlaps = masks_utils.compute_mask_overlaps(point_to_masks, mask_to_points, min_shared=min_shared, top_k=10**9, log=(print if verbose else (lambda *a, **k: None)))
        # ---------- Step 2: derive merged groups under current jaccard_threshold ----------
        parent = {}
        def find_m(x):
            parent.setdefault(x, x)
            if parent[x] != x:
                parent[x] = find_m(parent[x])
            return parent[x]
        def union_m(a, b):
            ra, rb = find_m(a), find_m(b)
            if ra != rb:
                parent[rb] = ra

        for a, b, shared, jaccard_val, sA, sB in overlaps:
            if jaccard_val >= jaccard_threshold:
                union_m(a, b)

        all_masks = set(mask_to_points.keys())
        groups = defaultdict(list)
        for m in all_masks:
            groups[find_m(m)].append(m)
        merged_groups = [sorted(g) for g in groups.values() if len(g) > 1]
        if verbose:
            print(f"[INFO] Found {len(merged_groups)} merge candidate groups with jaccard >= {jaccard_threshold}")

        # ---------- Step 3: optionally run heavy propagation if not precomputed ----------
        if multi_view_links is None:
            multi_view_links = masks_utils.propagate_all_masks_gpu(
                points3D=self.points3D,
                images=self.images,
                mask_instances=self.mask_instances,
                gs_cameras=self.gs_cameras,
                device=device
            )
        self._cached_multi_view_links = multi_view_links

        # ---------- Step 4: initial point->instance assignment ----------
        point_to_instance = {}
        for pid, masks in point_to_masks.items():
            point_to_instance[pid] = masks[0] if masks else -1

        # ---------- Step 5: union merged_groups into root mapping ----------
        # Build union-find over instances using merged_groups (re-using parent map)
        # Note: parent already set by previous union_m calls, but ensure every mask key is present
        for m in mask_to_points.keys():
            parent.setdefault(m, m)
        # Now point->root -> cluster ids
        clusters = {}
        cluster_id_map = {}
        cid = 0
        def find_root_inst(inst):
            parent.setdefault(inst, inst)
            return find_m(inst)

        for pid, inst in point_to_instance.items():
            if inst == -1:
                clusters[pid] = -1
                continue
            root = find_root_inst(inst)
            if root not in cluster_id_map:
                cluster_id_map[root] = cid
                cid += 1
            clusters[pid] = cluster_id_map[root]

        self.point_clusters = clusters

        # Step 6: refine clusters using supplied spatial_consistency_weight
        refine_clusters_with_metric(
            self,
            max_iters=refine_iters,
            w_geom=spatial_consistency_weight,
            w_coh=1.0,
            w_size=0.5,
            w_count=0.5,
            alpha=1.0,
            spatial_consistency_weight=spatial_consistency_weight
        )

        return self.point_clusters


    def build_gaussian_instance_adjacency(
        self,
        topK_contrib_indices,   # [N, C, K]
        mask_images,            # list of C masks, each [I, H, W]
        num_gaussians,
        num_mask_instances,
        device="cuda"
    ):
        """
        Fully optimized + vectorized construction of adjacency:
            A[g, inst] = 1  if Gaussian g touches instance inst.
        """
        

        N, C, K = topK_contrib_indices.shape
        topK = topK_contrib_indices.to(device)

        # -----------------------------------------------
        # Precompute global gaussian index lookup
        # -----------------------------------------------
        # gaussian_ids_flat = [0,0,..0, 1,1,...1, ..., N-1,N-1,..N-1] of length N*K
        gaussian_ids_flat = (
            torch.arange(N, device=device)
            .unsqueeze(1)
            .expand(N, K)
            .reshape(-1)
        )  # [N*K]

        # Max total entries (upper bound, we filter invalid later)
        max_entries = C * N * K

        # Preallocate outputs (large, but avoids Python overhead)
        gaussian_buffer = torch.empty(max_entries, dtype=torch.long, device=device)
        instance_buffer = torch.empty(max_entries, dtype=torch.long, device=device)
        write_pos = 0

        # -----------------------------------------------
        # Process each camera
        # -----------------------------------------------
        for cam_idx in range(C):
            # ---- mask preprocessing (vectorized) ----
            mask_cam = mask_images[cam_idx].to(device)        # [I, H, W]
            I, H, W = mask_cam.shape

            # instance_map[y,x] = instance_id (0 = background)
            # argmax is OK and very fast in modern GPUs
            instance_map = mask_cam.argmax(dim=0).reshape(-1)     # [H*W]

            # ---- gather all K indices for this camera ----
            # topK_cam: [N,K]
            topK_cam = topK[:, cam_idx, :].reshape(-1)            # [N*K]

            # Mask: valid pixel indices ∈ [0, H*W)
            valid_pix = topK_cam >= 0
            if not valid_pix.any():
                continue

            topK_valid = topK_cam[valid_pix]                      # actual pixel IDs
            gauss_valid = gaussian_ids_flat[valid_pix]            # matching Gaussian IDs

            # ---- lookup instances at these pixel coords ----
            inst_ids = instance_map[topK_valid]                   # [M]

            # Filter background (0 is background)
            keep = inst_ids > 0
            if not keep.any():
                continue

            g_sel = gauss_valid[keep]
            i_sel = inst_ids[keep]

            # ---- write to preallocated buffer ----
            n_sel = g_sel.numel()
            gaussian_buffer[write_pos: write_pos+n_sel] = g_sel
            instance_buffer[write_pos: write_pos+n_sel] = i_sel
            write_pos += n_sel

        # -----------------------------------------------
        # Final slice
        # -----------------------------------------------
        gaussian_ids_final = gaussian_buffer[:write_pos]
        instance_ids_final = instance_buffer[:write_pos]

        # Clamp instance IDs
        instance_ids_final = instance_ids_final.clamp(0, num_mask_instances - 1)

        # -----------------------------------------------
        # Build sparse adjacency
        # -----------------------------------------------
        idx = torch.stack([gaussian_ids_final, instance_ids_final], dim=0)

        vals = torch.ones(idx.shape[1], dtype=torch.float32, device=device)

        A = torch.sparse_coo_tensor(
            idx,
            vals,
            size=(num_gaussians, num_mask_instances)
        ).coalesce()

        return A


