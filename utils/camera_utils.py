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

from scene.cameras import Camera
import numpy as np
from utils.graphics_utils import fov2focal
from PIL import Image
import cv2

WARNED = False

def loadCam(args, id, cam_info, resolution_scale, is_nerf_synthetic, is_test_dataset, mask_dir=None, invdepthmap=None):
    image = Image.open(cam_info.image_path)

    # Use in-memory depth map if provided
    if invdepthmap is None:
        if cam_info.depth_path != "":
            try:
                depth_img = cv2.imread(cam_info.depth_path, -1)
                if depth_img is None:
                    print(f"[WARNING] cv2.imread failed for depth file: {cam_info.depth_path}. File missing or unreadable.")
                    invdepthmap = None
                else:
                    if is_nerf_synthetic:
                        invdepthmap = depth_img.astype(np.float32) / 512
                    else:
                        invdepthmap = depth_img.astype(np.float32) / float(2**16)
            except FileNotFoundError:
                print(f"Error: The depth file at path '{cam_info.depth_path}' was not found.")
                invdepthmap = None
            except IOError:
                print(f"Error: Unable to open the image file '{cam_info.depth_path}'. It may be corrupted or an unsupported format.")
                invdepthmap = None
            except Exception as e:
                print(f"An unexpected error occurred when trying to read depth at {cam_info.depth_path}: {e}")
                invdepthmap = None
        else:
            invdepthmap = None
    else:
        # Use provided in-memory depth map
        pass  # invdepthmap is already set

    # --- Resolution scaling code remains the same ---
    orig_w, orig_h = image.size
    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:
        if args.resolution == -1 and orig_w > 1600:
            global WARNED
            if not WARNED:
                print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\nIf this is not desired, please explicitly specify '--resolution/-r' as 1")
                WARNED = True
            global_down = orig_w / 1600
        else:
            global_down = 1
        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    cam = Camera(
        resolution,
        colmap_id=cam_info.uid,
        R=cam_info.R,
        T=cam_info.T,
        FoVx=cam_info.FovX,
        FoVy=cam_info.FovY,
        depth_params=cam_info.depth_params,
        image=image,
        invdepthmap=invdepthmap,
        image_name=cam_info.image_name,
        uid=id,
        data_device=args.data_device,
        train_test_exp=args.train_test_exp,
        is_test_dataset=is_test_dataset,
        is_test_view=cam_info.is_test,
        mask_dir=mask_dir
    )

    # Enforce mask presence
    if mask_dir is not None and not cam.has_valid_mask:
        return None

    return cam

def cameraList_from_camInfos(
    cam_infos,
    resolution_scale,
    args,
    is_nerf_synthetic,
    is_test_dataset,
    mask_dir=None,
    invdepthmaps=None
):
    camera_list = []

    for id, c in enumerate(cam_infos):
        invdepthmap = None
        if invdepthmaps is not None:
            # invdepthmaps can be a dict (by image_name or uid) or a list
            if isinstance(invdepthmaps, dict):
                invdepthmap = invdepthmaps.get(getattr(c, 'image_name', None))
                if invdepthmap is None:
                    invdepthmap = invdepthmaps.get(getattr(c, 'uid', None))
            elif isinstance(invdepthmaps, list) and id < len(invdepthmaps):
                invdepthmap = invdepthmaps[id]
        cam = loadCam(
            args,
            id,
            c,
            resolution_scale,
            is_nerf_synthetic,
            is_test_dataset,
            mask_dir=mask_dir,
            invdepthmap=invdepthmap
        )

        if cam is not None:
            camera_list.append(cam)

    print(
        f"[INFO] Cameras kept after mask filtering: "
        f"{len(camera_list)} / {len(cam_infos)}"
    )

    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry