import argparse
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt



import os
import sys

def find_image_files(rgb_path):
    base = rgb_path[:-4]
    suffix = rgb_path[-4:] # TODO: Make dynamic
    g_path = base + "_g" + suffix
    r_path = base + "_r" + suffix
    re_path = base + "_re" + suffix
    nir_path = base + "_nir" + suffix

    return {
        "rgb": rgb_path,
        "g": g_path,
        "r": r_path,
        "re": re_path,
        "nir": nir_path
    }

def load_image_as_array(path: str, resize: bool = False, resize_size: tuple = (512, 480)):
    img = Image.open(path)
    img = np.array(img)
    if resize:
        img = cv2.resize(img, resize_size, interpolation=cv2.INTER_CUBIC)
    return img.astype(np.float32)

def save_array_as_image(array, path):
    img = Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))
    img.save(path)

def save_array_as_pink_colormap(array, path):
    # Ensure input is 2D
    if array.ndim != 2:
        raise ValueError(f"Input array must be 2D (grayscale), got shape: {array.shape}")
    # Normalize array to 0-1
    arr_norm = (array - np.min(array)) / (np.max(array) - np.min(array) + 1e-8)
    # Use matplotlib's 'pink' colormap
    cmap = plt.get_cmap('pink')
    arr_colored = cmap(arr_norm)
    # Remove alpha channel if present
    if arr_colored.shape[-1] == 4:
        arr_colored = arr_colored[:, :, :3]
    # Convert to 8-bit RGB
    arr_rgb = (arr_colored * 255).astype(np.uint8)
    img = Image.fromarray(arr_rgb)
    img.save(path)


def ndvi_error_map(pred, gt):
    pred_nir = load_image_as_array(pred_files["nir"])
    pred_r = load_image_as_array(pred_files["r"])
    gt_nir = load_image_as_array(gt_files["nir"])
    gt_r = load_image_as_array(gt_files["r"])
    
    pred = (pred_nir - pred_r) / (pred_nir + pred_r + 1e-6)
    gt = (gt_nir - gt_r) / (gt_nir + gt_r + 1e-6)            
    return gt

def ndre_error_map(pred, gt):
    pred_nir = load_image_as_array(pred_files["nir"])
    pred_re = load_image_as_array(pred_files["re"])
    gt_nir = load_image_as_array(gt_files["nir"])
    gt_re = load_image_as_array(gt_files["re"])
    
    pred = (pred_nir - pred_re) / (pred_nir + pred_re + 1e-6)
    gt = (gt_nir - gt_re) / (gt_nir + gt_re + 1e-6)            
    return gt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate error maps for given prediction and ground truth.")

    # These are the path to the predicted and ground truth of the rgb images.
    # Expects _re, _g, _r, _nir suffix
    parser.add_argument("--pred_path", default="error_maps/vietnam/pred.jpg")
    parser.add_argument("--gt_path", default="error_maps/vietnam/gt.TIF")
    args = parser.parse_args()

    pred_files = find_image_files(args.pred_path)
    gt_files = find_image_files(args.gt_path)

    ndvi_error = ndvi_error_map(pred_files, gt_files)
    # If output is 3D, convert to 2D by averaging across channels
    if ndvi_error.ndim == 3:
        ndvi_error_2d = np.mean(ndvi_error, axis=-1)
    else:
        ndvi_error_2d = ndvi_error
    save_array_as_pink_colormap(ndvi_error_2d, "ndvi_error_map_pink.png")

    ndre_error = ndre_error_map(pred_files, gt_files)
    if ndre_error.ndim == 3:
        ndre_error_2d = np.mean(ndre_error, axis=-1)
    else:
        ndre_error_2d = ndre_error
    save_array_as_pink_colormap(ndre_error_2d, "ndre_error_map_pink.png")

    # # SAM error map (using R, G, RE, NIR bands)
    # sam_error = sam_error_map(pred_files, gt_files)
    # # Save with a perceptually uniform colormap (e.g., 'viridis')
    # arr_norm = (sam_error - np.min(sam_error)) / (np.max(sam_error) - np.min(sam_error) + 1e-8)
    # cmap = plt.get_cmap('viridis')
    # arr_colored = cmap(arr_norm)
    # arr_rgb = (arr_colored[:, :, :3] * 255).astype(np.uint8)
    # img = Image.fromarray(arr_rgb)
    # img.save("sam_error_map_viridis.png")
