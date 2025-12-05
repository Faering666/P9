import os
import torch
import cv2
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from typing import Callable


def sri_lanka_data(root_dir: Path, full: bool = False) -> list[str]:
    if not full:
        rgb_paths = sorted([f for f in root_dir.rglob("*") if f.is_file() and f.suffix.lower() in [f"_{x}.jpg"]] for x in range(71))
        return rgb_paths
    else:
        rgb_paths = sorted([f for f in root_dir.rglob("*_D.JPG") if f.is_file()])
        return rgb_paths

class DataCarrier(Dataset):
    """
    Dataset for paired RGB and multi-spectral (MS) images.

    Returns:
        tuple: (rgb, ms) where each is a torch.FloatTensor [C, H, W]
    """
    BAND_ORDER = ["G", "R", "RE", "NIR"]

    def __init__(self, root_dir: str,
                 load_data: Callable[[Path, bool], list[str]],
                 full: bool = False):
        self.root_dir = Path(root_dir)
        self.full = full
        self.bases = load_data(self.root_dir, self.full)

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
        rgb = self._load_and_normalize(base)
        rgb = rgb[:,:,::-1].copy() # bgr -> rgb

        # Load ms bands in correct order (G, R, RE, NIR)
        bands = []

        for suffix in self.BAND_ORDER:
            path = os.path.join(self.root_dir, base.replace("_D", f"_MS_{suffix}").replace(".JPG", ".TIF"))
            band = self._load_and_normalize(path)
            # Take first channel if image is 3-channel (grayscale stored as RGB)
            if band.ndim == 3:
                band = band[:, :, 0]
            bands.append(band)
        target = np.stack(bands, axis=-1)

        # Convert to torch tensors and rearrange to [C, H, W]
        rgb = torch.from_numpy(rgb).permute(2, 0, 1).float()
        target = torch.from_numpy(target).permute(2, 0, 1).float()

        return {"rgb": rgb, "ms": target}

if __name__ == "__main__":
    print("Testing DataCarrier...")
    dataset = DataCarrier(root_dir="data/MS_Sri_Lanka", load_data=sri_lanka_data, full=False)
    print(dataset.__len__())
    rgb, ms = dataset[0]
    print("rgb patch shape:", rgb.shape)
    print("ms patch shape:", ms.shape)
