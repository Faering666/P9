import cv2

def extract_features(frame, detector=None):
    """
    Extract ORB keypoints and descriptors from a frame.
    """
    if detector is None:
        detector = cv2.ORB_create(3000)
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = detector.detectAndCompute(gray, None)
    return keypoints, descriptors

def match_features(desc1, desc2, matcher=None, ratio_test=0.75):
    """
    Match descriptors using BFMatcher and apply Lowe's ratio test.
    """
    if matcher is None:
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    matches = matcher.knnMatch(desc1, desc2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < ratio_test * n.distance:
            good_matches.append(m)
    return good_matches