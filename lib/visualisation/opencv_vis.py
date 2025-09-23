import cv2
import numpy as np

def draw_keypoints(frame, keypoints):
    """
    Draw keypoints on the frame.
    """
    return cv2.drawKeypoints(frame, keypoints, None, color=(0, 255, 0))

def draw_matches(frame1, kp1, frame2, kp2, matches, max_matches=50):
    """
    Draw matches between two frames.
    """
    return cv2.drawMatches(
        frame1, kp1, frame2, kp2,
        matches[:max_matches], None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

def draw_reprojected_points(frame, K, R, t, pts3D):
    """
    Project 3D points back into the image and draw them.
    """
    if len(pts3D) == 0:
        return frame

    proj = K @ np.hstack((R, t))
    pts3D_h = np.hstack([pts3D, np.ones((pts3D.shape[0], 1))]).T
    pts2D = proj @ pts3D_h
    pts2D /= pts2D[2]

    img = frame.copy()
    pts2D = pts2D[:2].T.astype(int)  # shape (N,2)

    for (x, y) in pts2D:
        if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
            cv2.circle(img, (x, y), 2, (0, 0, 255), -1)
    return img
