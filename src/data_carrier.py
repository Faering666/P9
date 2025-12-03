import os
from types import resolve_bases

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
        rgb = rgb[:,:,::-1].copy() # bgr -> rgb

        # Load ms bands in correct order (G, R, RE, NIR)
        bands = []

        for suffix in self.BAND_ORDER:
            path = os.path.join(self.root_dir, base.replace("_D", f"_MS_{suffix}").replace(".JPG", ".TIF"))
            band = self._load_and_normalize(path)
            bands.append(band)
        target = np.stack(bands, axis=-1)

        print(target.shape, rgb.shape)
        # Convert to torch tensors and rearrange to [C, H, W]
        rgb = torch.from_numpy(rgb).float()
        target = torch.from_numpy(target).float()

        return {"rgb": rgb, "ms": target}

if __name__ == "__main__":
    print("Testing DataCarrier...")
    dataset = DataCarrier(root_dir="../data/Multispectral Images on Paddy- Sri Lanka")
    print(len(dataset))
    rgb, ms = dataset[0]
    print("rgb patch shape:", rgb.shape)
    print("ms patch shape:", ms.shape)
