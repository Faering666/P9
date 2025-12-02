import os

import torch
from torch.utils.data import Dataset
import cv2
import numpy as np

class DataCarrier(Dataset):
    """
    Dataset for paired RGB and multi-spectral (MS) images.

    Returns:
        tuple: (rgb, ms) where each is a torch.FloatTensor [C, H, W]
    """
    BAND_ORDER = ["G", "R", "RE", "NIR"]

    def __init__(self, root_dir):
        self.root_dir = root_dir

        all_files = set(os.listdir(root_dir))
        rgb_paths = sorted([f for f in all_files if any(f.endswith(f"_{x}.JPG") for x in range(71))])

        self.bases = rgb_paths
        self.size = 256

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

        # Load rgb
        rgb_path = os.path.join(self.root_dir, base)
        rgb = self._load_and_normalize(rgb_path)
        rgb = cv2.resize(rgb, (self.size, self.size))[:, :, ::-1].copy()
        rgb = rgb[:,:,::-1].copy() # bgr -> rgb

        # Load ms bands in correct order (G, R, RE, NIR)
        bands = []

        for suffix in self.BAND_ORDER:
            path = os.path.join(self.root_dir, base.replace("_D", f"_MS_{suffix}").replace(".JPG", ".TIF"))
            band = self._load_and_normalize(path)
            bands.append(band)
        target = np.stack(bands, axis=-1)
        target = cv2.resize(target, (self.size, self.size)).copy()

        # Ensure target has 3 dimensions [H, W, C]
        if target.ndim == 2:
            target = target[:, :, np.newaxis]

        # Convert to torch tensors and rearrange to [C, H, W]
        rgb = torch.from_numpy(rgb).permute(2, 0, 1).float()
        target = torch.from_numpy(target).permute(2, 0, 1).float()
        return rgb, target

if __name__ == "__main__":
    print("Testing DataCarrier...")
    dataset = DataCarrier(root_dir="data/")
    print(len(dataset))
    rgb, ms = dataset[0]
    print("rgb patch shape:", rgb.shape)
    print("ms patch shape:", ms.shape)
