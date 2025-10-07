import cv2
from lib.utils.visualisation import show_matches


class ImageComparer:
    """
    Compares frames against a reference image using ORB feature matching.
    """

    def __init__(self, match_threshold=60):
        """
        Args:
            match_threshold (int): Distance threshold for good matches.
        """
        self.orb = cv2.ORB_create()
        self.match_threshold = match_threshold
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def compare(self, img1, img2, visualize=False):
        """
        Compare two images using ORB feature matching.

        Args:
            img1 (ndarray): Current frame
            img2 (ndarray): Reference image
            visualize (bool): If True, show matches in a debug window

        Returns:
            float: similarity score between 0 and 1
        """
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        kp1, des1 = self.orb.detectAndCompute(gray1, None)
        kp2, des2 = self.orb.detectAndCompute(gray2, None)

        if des1 is None or des2 is None:
            return 0.0

        matches = self.matcher.match(des1, des2)
        if not matches:
            return 0.0

        matches = sorted(matches, key=lambda x: x.distance)
        good_matches = [m for m in matches if m.distance < self.match_threshold]
        similarity = len(good_matches) / len(matches)

        if visualize:
            show_matches(img1, img2, kp1, kp2, matches, similarity)

        return similarity
