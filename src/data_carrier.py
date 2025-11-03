import os
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np

class DataCarrier(Dataset):
    """
    Dataset for paired RGB and multi-spectral (MS) images.
    Expects each sample to have:
        - RGB image ending with '_D.JPG'
        - MS bands: '_MS_R.TIF', '_MS_G.TIF', '_MS_RE.TIF', '_MS_NIR.TIF'
    Returns:
        dict with keys 'rgb' and 'ms', each as a torch.FloatTensor [C, H, W]
    """
    BAND_ORDER = ["_MS_R.TIF", "_MS_G.TIF", "_MS_RE.TIF", "_MS_NIR.TIF"]

    def __init__(self, root_dir, size=256):
        self.root_dir = root_dir
        self.size = size

        all_files = set(os.listdir(root_dir))
        candidate_rgb = sorted([f for f in all_files if f.endswith("_D.JPG")])

        self.bases = []
        for rgb_name in candidate_rgb:
            base = rgb_name.replace("_D.JPG", "")
            expected_files = {base + suffix for suffix in self.BAND_ORDER}
            if expected_files.issubset(all_files):
                self.bases.append(base)
            else:
                missing = expected_files - all_files
                print(f"Warning: skipping {base}, missing files: {sorted(missing)}")

        if not self.bases:
            raise RuntimeError(f"No complete samples found in '{root_dir}'.")

    def __len__(self):
        return len(self.bases)

    @staticmethod
    def _load_and_normalize(path):
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Cannot read file: {path}")
        
        # Normalize before casting to float32
        if img.dtype == np.uint16:
            img = img.astype(np.float32) / 65535.0
        else:
            img = img.astype(np.float32) / 255.0
        return img

    def __getitem__(self, idx):
        base = self.bases[idx]

        # Load RGB
        rgb_path = os.path.join(self.root_dir, base + "_D.JPG")
        rgb = self._load_and_normalize(rgb_path)
        rgb = cv2.resize(rgb, (self.size, self.size))[:, :, ::-1].copy()

        # Load MS bands in desired order
        bands = []
        for suffix in self.BAND_ORDER:
            path = os.path.join(self.root_dir, base + suffix)
            band = self._load_and_normalize(path)
            bands.append(band)

        target = np.stack(bands, axis=-1)
        target = cv2.resize(target, (self.size, self.size)).copy()

        rgb_tensor = torch.from_numpy(rgb).permute(2, 0, 1).float()
        target_tensor = torch.from_numpy(target).permute(2, 0, 1).float()

        return {"rgb": rgb_tensor, "ms": target_tensor}
