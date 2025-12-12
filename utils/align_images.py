import cv2
import numpy as np

def read_and_preprocess(path, grayscale=True):
    if grayscale:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
    return img

def align_image(ref_img, img_to_align, use_affine=False):
    # Always use grayscale for feature detection
    ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY) if len(ref_img.shape) == 3 else ref_img
    align_gray = img_to_align if len(img_to_align.shape) == 2 else cv2.cvtColor(img_to_align, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(5000)
    kp1, des1 = orb.detectAndCompute(ref_gray, None)
    kp2, des2 = orb.detectAndCompute(align_gray, None)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    if len(matches) < 4:
        raise Exception("Not enough matches found for alignment.")
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
    if use_affine:
        M, _ = cv2.estimateAffinePartial2D(pts2, pts1, method=cv2.RANSAC)
        aligned = cv2.warpAffine(img_to_align, M, (ref_img.shape[1], ref_img.shape[0]))
    else:
        H, _ = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)
        aligned = cv2.warpPerspective(img_to_align, H, (ref_img.shape[1], ref_img.shape[0]))
    return aligned

def crop_to_valid_overlap(images, extra_margin=0.05):
    # Find intersection of non-black (non-zero) regions in all images
    masks = []
    for img in images:
        if len(img.shape) == 3:
            mask = (cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) > 0).astype(np.uint8)
        else:
            mask = (img > 0).astype(np.uint8)
        masks.append(mask)
    overlap = np.logical_and.reduce(masks).astype(np.uint8)
    # Erode to avoid edge black bars
    kernel = np.ones((15, 15), np.uint8)
    overlap = cv2.erode(overlap, kernel, iterations=1)
    if not np.any(overlap):
        raise Exception("No overlapping area found.")
    x, y, w, h = cv2.boundingRect(overlap)
    # Apply extra margin (zoom in)
    margin_x = int(w * extra_margin)
    margin_y = int(h * extra_margin)
    x += margin_x
    y += margin_y
    w -= 2 * margin_x
    h -= 2 * margin_y
    return [img[y:y+h, x:x+w] for img in images]

paths = {
    'rgb': r'data\sri_lanka\DJI_20230814123320_0001_D.JPG',
    'green': r'data\sri_lanka\DJI_20230814123320_0001_MS_G.TIF',
    'red': r'data\sri_lanka\DJI_20230814123320_0001_MS_R.TIF',
    'red_edge': r'data\sri_lanka\DJI_20230814123320_0001_MS_RE.TIF',
    'nir': r'data\sri_lanka\DJI_20230814123320_0001_MS_NIR.TIF'
}

# Read RGB in color, others in grayscale
ref_img = read_and_preprocess(paths['green'], grayscale=True)
images = []
# Align RGB to reference
rgb_img = read_and_preprocess(paths['rgb'], grayscale=False)
aligned_rgb = align_image(ref_img, rgb_img, use_affine=True)
images.append(aligned_rgb)
# Align all MS bands (including green itself, which will be identical to ref_img)
for band in ['green', 'red', 'red_edge', 'nir']:
    img = read_and_preprocess(paths[band])
    aligned = align_image(ref_img, img, use_affine=True)
    images.append(aligned)

cropped_images = crop_to_valid_overlap(images)

# Save results
for i, band in enumerate(['rgb', 'green', 'red', 'red_edge', 'nir']):
    cv2.imwrite(f'aligned_cropped_{band}.png', cropped_images[i])
