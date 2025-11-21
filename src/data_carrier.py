import os
import random
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np

class DataCarrier(Dataset):
    """
    Dataset for paired RGB and multi-spectral (MS) images.

    Returns:
        dict with keys 'rgb' and 'ms', each as a torch.FloatTensor [C, H, W]
    """
    BAND_ORDER = ["Red_Channel_", "Green_Channel_", "Red_Edge_Channel_", "Near_Infrared_Channel_"]

    def __init__(self, root_dir, patch_size=128, img_size=416):
        self.root_dir = root_dir
        self.patch_size = patch_size
        self.img_size = img_size

        all_files = set(os.listdir(root_dir))
        candidate_rgb = sorted([f for f in all_files if f.startswith("Image")])

        self.bases = []
        for rgb_name in candidate_rgb:
            # base = rgb_name.replace(".jpg", "")
            base = rgb_name
            expected_files = {prefix + base for prefix in self.BAND_ORDER}
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

        # Load rgb
        rgb_path = os.path.join(self.root_dir, base)
        rgb = self._load_and_normalize(rgb_path)
        rgb = rgb[:,:,::-1].copy() # bgr -> rgb

        # Load ms bands in correct order (G, R, RE, NIR)
        bands = []
        for prefix in self.BAND_ORDER:
            path = os.path.join(self.root_dir, prefix + base)
            band = self._load_and_normalize(path)
            bands.append(band)
        target = np.stack(bands, axis=-1)

        # TODO: Don't hardcode resize to 416 here.
        rgb = cv2.resize(rgb, (self.img_size, self.img_size))
        target = cv2.resize(target, (self.img_size, self.img_size))

        # Random patch sampling
        max_offset = self.img_size - self.patch_size
        x = random.randint(0, max_offset)
        y = random.randint(0, max_offset)
        rgb_patch = rgb[y:y+self.patch_size, x:x+self.patch_size, :]
        ms_patch = target[y:y+self.patch_size, x:x+self.patch_size, :]

        rgb_tensor = torch.from_numpy(rgb_patch).permute(2, 0, 1).float()
        ms_tensor = torch.from_numpy(ms_patch).permute(2, 0, 1).float()

        return {"rgb": rgb_tensor, "ms": ms_tensor}

if __name__ == "__main__":
    print("Testing DataCarrier...")
    dataset = DataCarrier(root_dir="data/", patch_size=128, img_size=416)
    print(len(dataset))
    sample = dataset[0]
    print("rgb patch shape:", sample["rgb"].shape)
    print("ms patch shape:", sample["ms"].shape)
