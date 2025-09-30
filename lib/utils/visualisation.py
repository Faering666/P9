import cv2

def show_matches(img1, img2, kp1, kp2, matches, similarity, width=1280):
    """
    Draws matches between two images and displays them with a similarity score.

    Args:
        img1, img2 (ndarray): Input images
        kp1, kp2: Keypoints for images
        matches (list): Feature matches
        similarity (float): Similarity score
        width (int): Width for resizing display window
    """
    match_img = cv2.drawMatches(img1, kp1, img2, kp2, matches[:20], None, flags=2)
    cv2.putText(match_img, f"Similarity: {similarity:.2f}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 255, 0), 2)

    # Resize for display
    h, w = match_img.shape[:2]
    scale = width / w
    resized = cv2.resize(match_img, (width, int(h * scale)))

    cv2.imshow("Matches", resized)
    cv2.waitKey(1)
