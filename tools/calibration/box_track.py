import numpy as np
import pyk4a
from pyk4a import PyK4APlayback, CalibrationType
import cv2
import os
import pickle
import time
from utils import get_detections
from box_coords import ALL_TAG_COORDS, NUM_TAGS

class Camera:
    def __init__(self, intrinsics, extrinsics):
        self.intrinsics = intrinsics
        self.extrinsics = extrinsics

# assumes dir structure is dir/cam_extr/camera_{id}.pkl and dir/cam_intr/camera_{id}.pkl
def initialize_cameras(dir):
    cameras = {}
    extr_dir = os.path.join(dir, "cam_extr")
    intr_dir = os.path.join(dir, "cam_intr")
    
    for filename in os.listdir(extr_dir):
        if filename.startswith("camera_") and filename.endswith(".pkl"):
            cam_id = filename[len("camera_"):-len(".pkl")]
            extr_path = os.path.join(extr_dir, filename)
            intr_path = os.path.join(intr_dir, f"camera_{cam_id}.pkl")
            
            if not os.path.exists(intr_path):
                print(f"Warning: Intrinsics file not found for camera {cam_id}, skipping")
                continue
            
            with open(extr_path, "rb") as f:
                extrinsics = pickle.load(f)
            with open(intr_path, "rb") as f:
                intrinsics = pickle.load(f)
            
            cam_id = int(cam_id)
            cameras[cam_id] = Camera(intrinsics=intrinsics, extrinsics=extrinsics)
        else:
            raise Exception(f"Unexpected file in extrinsics directory: {filename}")
    
    return cameras

# TODO this is the function that will be passed in for optimization, all 
# the code below once tested needs to be placed in here and we need to calculate total residual error
def project_points(box_pose: np.ndarray, cameras: dict, tag_coords_3d: np.ndarray):
    calib_dir = "_DATA\\03-03-2026\\03-03-2026\\calib_2026-03-03"
    cameras = initialize_cameras(calib_dir)

    # TODO un hardcode this
    dir_zero = "_DATA\\03-03-2026\\03-03-2026\\k4a_0_000679600712"    
    dir_one = "_DATA\\03-03-2026\\03-03-2026\\k4a_1_000696700112"
    dir_two = "_DATA\\03-03-2026\\03-03-2026\\k4a_2_000706100112"
    dir_three = '_DATA\\03-03-2026\\03-03-2026\\k4a_3_000279301412'

    dirs = [dir_zero, dir_one, dir_two, dir_three]

    return True

if __name__ == "__main__":
    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--image-dir", type=str, required=True, help="Directory containing input images")
    # args = parser.parse_args()

    calib_dir = "_DATA\\03-03-2026\\03-03-2026\\calib_2026-03-03"
    cameras = initialize_cameras(calib_dir)

    # TODO un hardcode this
    dir_zero = "_DATA\\03-03-2026\\03-03-2026\\k4a_0_000679600712"    
    dir_one = "_DATA\\03-03-2026\\03-03-2026\\k4a_1_000696700112"
    dir_two = "_DATA\\03-03-2026\\03-03-2026\\k4a_2_000706100112"
    dir_three = '_DATA\\03-03-2026\\03-03-2026\\k4a_3_000279301412'

    dirs = [dir_zero, dir_one, dir_two, dir_three]

    # initial 6D pose for the box, [r, t]
    initial_box_pose = np.random.randn(6)
    

    for cam_id, cam in cameras.items():

        tags_found = np.zeros(NUM_TAGS, dtype=np.int32)
        tag_pixel_coords = np.zeros((NUM_TAGS, 4, 2), dtype=np.float64) # pixel coordinates of the 4 corners of each tag, in the same order as ALL_TAG_COORDS
        rvec = initial_box_pose[:3].reshape(3,1)
        tvec = initial_box_pose[3:]

        R, _ = cv2.Rodrigues(rvec)

        box_to_world = np.eye(4)
        box_to_world[:3, :3] = R
        box_to_world[:3, 3] = tvec
        
        extr = cam.extrinsics
        world_to_cam = np.linalg.inv(extr) 

        box_to_cam = world_to_cam @ box_to_world
        
        cam_dir = dirs[cam_id]
        
        video_path = os.path.join(cam_dir, "raw.mkv")

        # we don't want to be reading from disk every time
        pb = PyK4APlayback(video_path)
        pb.open()
        pb.get_next_capture()

        cap = pb.get_next_capture()
        color_image = cap.color
        bgr = cv2.imdecode(color_image, cv2.IMREAD_COLOR)
        detections = get_detections(bgr)
        if detections is None:
            print(f"No detections found for camera {cam_id}. Skipping...")
            continue
        for det in detections:
            tag_id = det.tag_id
            tags_found[det.tag_id] = 1
            tag_pixel_coords[det.tag_id] = det.corners

        tag_2d_found = tag_pixel_coords[tags_found == 1].reshape(-1, 2)

        # these are in terms of the box frame
        # Nx4
        tag_3d_found = ALL_TAG_COORDS[tags_found == 1].reshape(-1, 4)

        box_to_cam_cartesian = box_to_cam[:3, :] # drop homogeneous coordinate
        K = cam.intrinsics
        tag_3d_projected = K @ box_to_cam_cartesian @ tag_3d_found.T # 3x4 @ 4xN -> 3xN

        # treat z as the new homogenous coordinate, divide x and y by z to get pixel coordinates
        u = tag_3d_projected[0, :] / tag_3d_projected[2, :]
        v = tag_3d_projected[1, :] / tag_3d_projected[2, :]

        tag_2d_found = tag_2d_found.T # 2xN
        print(tag_2d_found.shape, tag_3d_projected.shape)
        error = np.sum(np.sqrt((u - tag_2d_found[0, :]) ** 2 + (v - tag_2d_found[1, :]) ** 2))
        print(f"Total reprojection error for camera {cam_id}: {error}")
        





            







        


