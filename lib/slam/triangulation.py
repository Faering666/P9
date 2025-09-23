import cv2
import numpy as np

def triangulate_points(kp1, kp2, matches, K, R, t, mask_pose):
    """
    Triangulate 3D points from matched keypoints.
    """
    # Another just trust me bro
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

    pts1 = pts1[mask_pose == 1]
    pts2 = pts2[mask_pose == 1]

    if len(pts1) < 2:
        return np.empty((0, 3))

    # Projection matrices (float32)
    # FYI @ is matrix multiplication if you were wondering :)
    proj1 = (K @ np.hstack((np.eye(3), np.zeros((3, 1))))).astype(np.float32)
    proj2 = (K @ np.hstack((R, t))).astype(np.float32)

    # OpenCV expects (2, N) float32 arrays
    pts1 = pts1.T.astype(np.float32)
    pts2 = pts2.T.astype(np.float32)

    pts4D = cv2.triangulatePoints(proj1, proj2, pts1, pts2)
    pts3D = (pts4D[:3] / pts4D[3]).T

    # Keep only valid points
    mask_valid = np.isfinite(pts3D).all(axis=1) & (pts3D[:, 2] > 0)
    return pts3D[mask_valid]
