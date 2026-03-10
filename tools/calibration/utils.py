import numpy as np
import cv2
from pupil_apriltags import Detector

at_detector = Detector(
    families="tag36h11",
    nthreads=1,
    quad_decimate=1.0,
    quad_sigma=0.0,
    refine_edges=1,
    decode_sharpening=0.25,
    debug=0
)

# Returns a list of april tag detections in a given image frame. 
def get_detections(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detections = at_detector.detect(gray)
    if len(detections) == 0:
        return None
    return detections

def decompose_homography(H, K):
    """
    Returns R (3x3) and t (3,) in tag units as a homogeneous transformation matrix.
    Assumes the planar model used to generate H has Z=0 and tag corners (±1, ±1).
    Takes in the homography matrix H (3x3) and the camera intrinsic matrix K (3x3).
    """
    Kinv = np.linalg.inv(K)
    A = Kinv @ H  # 3x3

    a1 = A[:, 0]
    a2 = A[:, 1]
    a3 = A[:, 2]

    # common scale (use average of norms; robust under noise)
    lam1 = 1.0 / np.linalg.norm(a1)
    lam2 = 1.0 / np.linalg.norm(a2)
    lam = 0.5 * (lam1 + lam2)

    r1 = lam * a1
    r2 = lam * a2
    r3 = np.cross(r1, r2)

    R = np.column_stack((r1, r2, r3))

    # Orthonormalize via SVD and enforce det=+1
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        R[:, 2] *= -1

    t = lam * a3  # translation in "tag units"

    homogenous = np.eye(4)
    homogenous[:3,:3] = R
    homogenous[:3, 3] = t

    return homogenous.astype(np.float64)

# Visualization Utils

def draw_tag_corners(image, corners_2d, color, label=None, radius=6, thickness=2):
    """
    Draws the 4 corners of a tag on the image and connects them with lines.

    Args:
        corners_2d: (4, 2) pixel coordinates
        color: BGR color tuple
        label: optional string label drawn near corner 0
    """
    corners_int = corners_2d.astype(np.int32)

    for i, (x, y) in enumerate(corners_int):
        cv2.circle(image, (x, y), radius, color, -1)
        cv2.putText(image, str(i), (x + 6, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

    cv2.polylines(image, [corners_int.reshape(-1, 1, 2)], isClosed=True, color=color, thickness=thickness)

    if label is not None:
        lx, ly = corners_int[0]
        cv2.putText(image, label, (lx - 10, ly - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def draw_reprojection_errors(image, detected_2d, projected_2d):
    """
    Draws lines between detected and projected corners.

    Args:
        detected_2d:  (N, 2) detected pixel coordinates
        projected_2d: (N, 2) projected pixel coordinates
    """
    for (dx, dy), (px, py) in zip(detected_2d.astype(np.int32), projected_2d.astype(np.int32)):
        cv2.line(image, (dx, dy), (px, py), (0, 165, 255), 1, cv2.LINE_AA)


def overlay_tags(image, detected_corners, projected_corners, cam_id=None):
    """
    Overlays detected (green) and projected (red) tag corners on an image,
    with orange reprojection error lines between them.

    Args:
        image:              (H, W, 3) BGR image — will be copied, not modified in place
        detected_corners:   dict of {tag_id: (4, 2)} detected pixel corners
        projected_corners:  dict of {tag_id: (4, 2)} projected pixel corners
        cam_id:             optional camera index shown in the overlay text

    Returns:
        vis:      annotated BGR image
        avg_err:  mean reprojection error in pixels over all matched tags (or nan)
    """
    vis = image.copy()

    total_err = 0.0
    n_matched = 0

    all_tag_ids = set(detected_corners) | set(projected_corners)

    for tag_id in sorted(all_tag_ids):
        if tag_id in projected_corners:
            draw_tag_corners(vis, projected_corners[tag_id],
                             color=(0, 0, 255), label=f"P{tag_id}")

        if tag_id in detected_corners:
            draw_tag_corners(vis, detected_corners[tag_id],
                             color=(0, 255, 0), label=f"D{tag_id}")

        if tag_id in detected_corners and tag_id in projected_corners:
            draw_reprojection_errors(vis, detected_corners[tag_id], projected_corners[tag_id])
            err = np.linalg.norm(detected_corners[tag_id] - projected_corners[tag_id], axis=1).mean()
            total_err += err
            n_matched += 1

    avg_err = total_err / n_matched if n_matched > 0 else float('nan')
    n_det  = len(detected_corners)
    n_proj = len(projected_corners)

    prefix = f"Cam {cam_id} | " if cam_id is not None else ""
    cv2.putText(vis, f"{prefix}Detected: {n_det}  Projected: {n_proj}  Avg err: {avg_err:.1f}px",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)

    legend_y = vis.shape[0] - 20
    cv2.putText(vis, "GREEN=detected  RED=projected  ORANGE=error",
                (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

    return vis, avg_err


def tile_images(images, max_width=1800, max_height=1000):
    """
    Tiles a list of images into a 2-column grid and scales to fit the screen.

    Args:
        images: list of (H, W, 3) BGR images — assumed all the same size
    Returns:
        combined BGR image
    """
    n = len(images)
    cols = min(n, 2)
    rows = (n + 1) // 2

    # Pad to even count if needed
    while len(images) < rows * cols:
        images.append(np.zeros_like(images[0]))

    grid_rows = [np.hstack(images[i * cols:(i + 1) * cols]) for i in range(rows)]
    combined  = np.vstack(grid_rows)

    scale = min(1.0, max_width / combined.shape[1], max_height / combined.shape[0])
    if scale < 1.0:
        combined = cv2.resize(combined, None, fx=scale, fy=scale)

    return combined