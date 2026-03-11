import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
from plyfile import PlyData

def visualize_clusters_from_ply(ply_path):
    """
    Load a PLY saved by GaussianModel.save_clustered_ply() and visualize clusters.
    Each cluster gets a unique color. All negative cluster IDs (background) are skipped.
    """
    # -------------------------
    # Read PLY
    # -------------------------
    ply = PlyData.read(ply_path)
    v = ply['vertex'].data

    # Extract xyz
    xyz = np.vstack([v['x'], v['y'], v['z']]).T

    # Check cluster field
    if 'cluster_id' not in v.dtype.names:
        raise ValueError("PLY has no 'cluster_id' field. Make sure you saved clustered PLY.")

    cluster_ids = v['cluster_id'].astype(np.int32)

    # -------------------------
    # Filter out all negative cluster points (background)
    # -------------------------
    mask = cluster_ids > 1
    xyz = xyz[mask]
    cluster_ids = cluster_ids[mask]

    # -------------------------
    # Create Open3D point cloud
    # -------------------------
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    # -------------------------
    # Assign colors per cluster
    # -------------------------
    unique_clusters = np.unique(cluster_ids)
    rng = np.random.default_rng(42)   # deterministic colors

    # Map cluster → color
    cluster_color_map = {cid: rng.random(3) for cid in unique_clusters}

    # Expand to per-point color array
    colors = np.array([cluster_color_map[cid] for cid in cluster_ids], dtype=np.float32)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    print(f"[INFO] Visualizing {xyz.shape[0]} points across {len(unique_clusters)} clusters (excluding background).")

    # -------------------------
    # Show
    # -------------------------
    o3d.visualization.draw_geometries([pcd])

def visualize_colmap_clusters(scene_info, scene, title="COLMAP Seed Clusters"):
    """
    Visualize COLMAP 3D points colored by cluster ID.
    Only COLMAP points, no Gaussians overlay.

    Inputs:
        scene_info: contains COLMAP point cloud info
        scene: Scene object containing `point_clusters` dict
    """

    # -------------------------------------
    # 1. Build arrays from point_clusters
    # -------------------------------------
    colmap_points_list = []
    colmap_cids_list = []

    # print(scene.points3D)

    unassigned = 0
    for pid, cid in scene.point_clusters.items():
        if pid in scene.points3D:
            # Extract xyz coordinates properly
            pt3d = scene.points3D[pid]
           
            try:
                pt = np.array(pt3d.xyz)
            except AttributeError:
                pt = np.array(pt3d[:3])

            colmap_points_list.append(pt)
            colmap_cids_list.append(cid)
        else:
            unassigned += 1

    if unassigned > 0:
        print(f"[DEBUG] {unassigned} points skipped (not found in points3D)")

    colmap_points = np.stack(colmap_points_list, axis=0)  # [Nc, 3]
    colmap_cids   = np.array(colmap_cids_list)            # [Nc]

    # -----------------------------
    # 2. Inspect clusters
    # -----------------------------
    unique_cids, counts = np.unique(colmap_cids, return_counts=True)
    # print(f"[DEBUG] Total points: {len(colmap_cids)}, unique clusters: {len(unique_cids)}")
    # for cid, cnt in zip(unique_cids, counts):
    #     print(f"  Cluster {cid}: {cnt} points")

    # -----------------------------
    # 3. Build unlimited color palette
    # -----------------------------
    K = len(unique_cids)
    base_maps = [plt.cm.get_cmap("tab20"), plt.cm.get_cmap("tab20b"), plt.cm.get_cmap("tab20c")]
    palette = []
    for cmap in base_maps:
        palette.extend([cmap(i) for i in range(cmap.N)])
    colors = [palette[i % len(palette)] for i in range(K)]

    # -----------------------------
    # 4. Plot
    # -----------------------------
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    for idx, cid in enumerate(unique_cids):
        mask = colmap_cids == cid
        cluster_pts = colmap_points[mask]
        ax.scatter(
            cluster_pts[:, 0],
            cluster_pts[:, 1],
            cluster_pts[:, 2],
            s=45,
            color=colors[idx],
            alpha=0.95,
            label=f"Cluster {cid}",
        )

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.show()
