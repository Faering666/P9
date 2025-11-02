import os
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np

class DataCarrier(Dataset):
    def __init__(self, root_dir, size=256):
        self.root_dir = root_dir
        self.size = size
        # find all bases that have a _D.JPG and all required MS band files
        all_files = set(os.listdir(root_dir))
        candidate_rgb = sorted([f for f in all_files if f.endswith("_D.JPG")])
        band_suffixes = ["_MS_G.TIF", "_MS_R.TIF", "_MS_RE.TIF", "_MS_NIR.TIF"]
        bases = []
        for rgb_name in candidate_rgb:
            base = rgb_name.replace("_D.JPG", "")
            expected = {base + s for s in band_suffixes}
            if expected.issubset(all_files):
                bases.append(base)
            else:
                # optional: warn about missing files
                missing = expected - all_files
                if missing:
                    print(f"Warning: skipping {base} because missing files: {sorted(missing)}")
        if not bases:
            raise RuntimeError(f"No complete samples found in '{root_dir}'. Check filenames and extensions.")
        # store base names (not full paths)
        self.bases = sorted(bases)


    def __len__(self):
        return len(self.bases)


    def __getitem__(self, idx):
        base = self.bases[idx]
        rgb_path = os.path.join(self.root_dir, base + "_D.JPG")

        # Load multi-spectral bands
        band_files = ["_MS_G.TIF", "_MS_R.TIF", "_MS_RE.TIF", "_MS_NIR.TIF"]
        bands = []
        for bf in band_files:
            path = os.path.join(self.root_dir, base + bf)
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise FileNotFoundError(f"Cannot read file: {path} (file exists but could not be read)")
            bands.append(img)
 
        # Resize to self.size
        rgb = cv2.imread(rgb_path)
        if rgb is None:
            raise FileNotFoundError(f"Cannot read RGB file: {rgb_path}")
        # convert BGR -> RGB and normalize depending on dtype
        rgb = rgb[:, :, ::-1]
        if rgb.dtype == np.uint16:
            rgb = rgb.astype(np.float32) / 65535.0
        else:
            rgb = rgb.astype(np.float32) / 255.0
        rgb = cv2.resize(rgb, (self.size, self.size))
 
        # Target band order: R, G, RE, NIR (user requirement)
        # bands list is [G, R, RE, NIR] so reorder accordingly
        bR = bands[1]
        bG = bands[0]
        bRE = bands[2]
        bNIR = bands[3]
        # normalize each band by dtype
        def norm_band(img):
            if img.dtype == np.uint16:
                return (img.astype(np.float32) / 65535.0)
            else:
                return (img.astype(np.float32) / 255.0)
 
        target = np.stack([norm_band(bR), norm_band(bG), norm_band(bRE), norm_band(bNIR)], axis=-1)
        target = cv2.resize(target, (self.size, self.size))
 
        rgb = torch.from_numpy(rgb).permute(2, 0, 1).float()
        target = torch.from_numpy(target).permute(2, 0, 1).float()
 
        return rgb, target