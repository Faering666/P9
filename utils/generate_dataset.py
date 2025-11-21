import os
import cv2
import numpy as np

    
def patch_image(image_path, patch_size=208):
    # Load image
    rgb_path = image_path
    print(rgb_path)

    # Create 4 patches of patch_size x patch_size
    rgb = cv2.imread(rgb_path, cv2.IMREAD_UNCHANGED)
    h, w, _ = (416, 416, 3)
    patches = []
    for i in range(2):
        for j in range(2):
            y0 = i * (h // 2)
            x0 = j * (w // 2)
            patch = rgb[y0:y0 + patch_size, x0:x0 + patch_size]
            patches.append(patch)

    # Save to disk
    for i, patch in enumerate(patches):
        patch_path = f"{rgb_path[:-4]}_{i}.jpg"
        cv2.imwrite(patch_path, patch)

root_dir = "data/"
for image_name in os.listdir(root_dir):
    if image_name.endswith('.jpg'):
        image_path = os.path.join(root_dir, image_name)
        patch_image(image_path)

