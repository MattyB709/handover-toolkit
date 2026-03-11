import numpy as np
import pyk4a
from pyk4a import PyK4APlayback, CalibrationType
import cv2
import os
import pickle
import time
from utils import get_detections, decompose_homography
import scipy
from box_coords import ALL_TAG_COORDS, NUM_TAGS, TAG_SIZE

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

# returns a 6D pose for the box, [rvec (3,), tvec (3,)]
def initialize_box_pose(img, K):
    detections = get_detections(img)
    if detections is None:
        raise Exception("No tags detected in the image, cannot initialize box pose")

    # use the first detection to initialize the box pose
    det = detections[0]
    tag_id = det.tag_id

    T = decompose_homography(det.homography, K)

    T[:3, 3] *= (TAG_SIZE / 2)  # scale translation into meters
    rvec = cv2.Rodrigues(T[:3, :3])[0].flatten()
    tvec = T[:3, 3].flatten()
    # return box_pose

    rvec, _ = cv2.Rodrigues(T[:3, :3])
    tvec = T[:3, 3]
    tag_3d_corners = ALL_TAG_COORDS[tag_id][:, :3] # remove homogenous coords
    tag_3d_center = tag_3d_corners.mean(axis=0)

    # now get the box pose relative to the camera, the idea is to use the box relative to the tag center
    # and the tag relative to the camera to get the box relative to the camera







def project_points(box_pose: np.ndarray,
                   cameras: dict,
                   tag_coords_3d: np.ndarray,
                   tags_found: np.ndarray,
                   tag_pixel_coords: np.ndarray):

    print("projecting points with box pose: ", box_pose)
    residuals = []

    rvec = box_pose[:3].reshape(3, 1)
    tvec = box_pose[3:]

    R, _ = cv2.Rodrigues(rvec)

    box_to_world = np.eye(4)
    box_to_world[:3, :3] = R
    box_to_world[:3, 3] = tvec

    for cam_id, cam in cameras.items():
        found_mask = tags_found[cam_id] == 1

        tag_2d_found = tag_pixel_coords[cam_id, found_mask].reshape(-1, 2)
        tag_3d_found = tag_coords_3d[found_mask].reshape(-1, 4)

        extr = cam.extrinsics
        world_to_cam = np.linalg.inv(extr)
        box_to_cam = world_to_cam @ box_to_world

        box_to_cam_cartesian = box_to_cam[:3, :]   # 3x4
        K = cam.intrinsics                         # 3x3

        tag_3d_projected = K @ box_to_cam_cartesian @ tag_3d_found.T  # 3xN

        u = tag_3d_projected[0, :] / tag_3d_projected[2, :]
        v = tag_3d_projected[1, :] / tag_3d_projected[2, :]
        proj_2d = np.stack((u, v), axis=1)  # Nx2

        residuals.append((proj_2d - tag_2d_found).ravel())

    return np.concatenate(residuals)

if __name__ == "__main__":

    calib_dir = "_DATA\\03-03-2026\\03-03-2026\\calib_2026-03-03"
    cameras = initialize_cameras(calib_dir)

    # TODO un hardcode this
    dir_zero = "_DATA\\03-03-2026\\03-03-2026\\k4a_0_000679600712"    
    dir_one = "_DATA\\03-03-2026\\03-03-2026\\k4a_1_000696700112"
    dir_two = "_DATA\\03-03-2026\\03-03-2026\\k4a_2_000706100112"
    dir_three = '_DATA\\03-03-2026\\03-03-2026\\k4a_3_000279301412'

    dirs = [dir_zero, dir_one, dir_two, dir_three]
    tags_found = np.zeros((4, NUM_TAGS))
    tag_pixel_coords = np.zeros((4, NUM_TAGS, 4, 2)) # pixel coordinates of the 4 corners of each tag, in the same order as ALL_TAG_COORDS

    for cam_id in range(len(cameras)):

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
            tags_found[cam_id, tag_id] = 1
            tag_pixel_coords[cam_id, tag_id] = det.corners

    # initial 6D pose for the box, [r, t]
    initial_box_pose = np.random.randn(6)

    result = scipy.optimize.least_squares(project_points, initial_box_pose, args=(cameras, ALL_TAG_COORDS), method='lm')
    rms = np.sqrt(np.mean(result.fun**2))
    print("final box re projection rms: ", rms)
