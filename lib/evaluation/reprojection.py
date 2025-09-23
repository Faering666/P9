import numpy as np
import cv2

def compute_reproj_error(pts3D, kp, matches, K, R, t, mask_pose):
    """
    Compute mean reprojection error between 3D points and their 2D matches.
    Returns None if no points available.
    """
    pts2D = np.float32([kp[m.trainIdx].pt for m in matches])[mask_pose == 1]

    if len(pts3D) == 0 or len(pts2D) == 0:
        return None

    proj = K @ np.hstack((R, t))
    pts3D_h = np.hstack([pts3D, np.ones((pts3D.shape[0], 1))]).T
    proj_pts = proj @ pts3D_h
    proj_pts /= proj_pts[2]

    proj_pts = proj_pts[:2].T
    n = min(len(pts2D), len(proj_pts))
    error = np.linalg.norm(pts2D[:n] - proj_pts[:n], axis=1)
    return np.mean(error)
