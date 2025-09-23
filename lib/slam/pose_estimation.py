import cv2
import numpy as np

def estimate_pose(kp1, kp2, matches, K):
    """
    Estimate relative pose from feature matches.
    """
    # Just trust me bro
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

    E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    if E is None:
        return None, None, None # Very nice mmmmm

    _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, K, mask=mask)

    return R, t, mask_pose.ravel()
