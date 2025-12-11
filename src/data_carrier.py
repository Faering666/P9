import os
import torch
import cv2
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from typing import Callable


def load_sri_lanka_full(root_dir: Path) -> list[Path]:
    rgb_paths = sorted([f for f in root_dir.rglob("*_D.JPG") if f.is_file()])
    return rgb_paths

def load_sri_lanka_patch(root_dir: Path) -> list[Path]:
    rgb_paths = sorted([f for f in root_dir.rglob("*") if f.is_file() and any(f.name.lower().endswith(f"_{x}.jpg") for x in range(71))])
    return rgb_paths

def load_east_kaz (root_dir: Path) -> list[Path]:
    rgb_paths = sorted([f for f in root_dir.rglob("*.JPG") if f.is_file()])
    return rgb_paths

def load_east_kaz_patch (root_dir: Path) -> list[Path]:
    rgb_paths = sorted([f for f in root_dir.rglob("_*0_*.jpg")])
    return rgb_paths

def load_weedy_rice (root_dir: Path) -> list[Path]:
    rgb_paths = sorted([f for f in root_dir.rglob("*.JPG") if f.is_file()])
    return rgb_paths

def load_single_picture (root_dir: Path) -> list[Path]: 
    rgb_paths = [root_dir]
    return rgb_paths


class DataCarrier(Dataset):
    """
    Dataset for paired RGB and multi-spectral (MS) images.

    Returns:
        tuple: (rgb, ms) where each is a torch.FloatTensor [C, H, W]
    """
    BAND_ORDER = ["G", "R", "RE", "NIR"]

    def __init__(self, root_dir: str, load_data: Callable[[Path], list[Path]], data_type="Sri-Lanka"):
        self.root_dir = Path(root_dir)
        self.bases = load_data(self.root_dir)
        self.full = True
        match load_data.__name__:
            case "load_sri_lanka_full":
                self.data_type = "Sri-Lanka"
                self.full = True
            case "load_sri_lanka_patch":
                self.data_type = "Sri-Lanka"
                self.full = False
            case "load_east_kaz":
                self.data_type = "Kazahkstan"
                self.full = True
            case "load_east_kaz_patch":
                self.data_type = "Kazahkstan"
                self.full = False
            case "load_weedy_rice":
                self.data_type = "Weedy-Rice"
                self.full = True
            case "load_weedy_rice_patch":
                breakpoint() #Not yet implemented
                self.data_type = "Weedy-Rice"
                self.full = False
            case "load_single_picture":
                self.data_type = data_type
                self.root_dir = self.root_dir.parent
            
            

    def __len__(self):
        return len(self.bases)

    @staticmethod
    def _load_and_normalize(path):
        path = str(path)
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
        base = str(self.bases[idx])

        # Load rgb
        rgb = self._load_and_normalize(base)
        rgb = rgb[:,:,::-1].copy() # bgr -> rgb

        # Load ms bands in correct order (G, R, RE, NIR)
        bands = []
        match self.data_type:
            case "Sri_Lanka":
                for suffix in self.BAND_ORDER:
                    path = os.path.join(base.replace("_D", f"_MS_{suffix}").replace(".JPG", ".TIF"))
                    band = self._load_and_normalize(path)
                    # Take first channel if image is 3-channel (grayscale stored as RGB)
                    if band.ndim == 3:
                        band = band[:, :, 0]
                    bands.append(band)
                target = np.stack(bands, axis=-1)

            case "Kazahkstan":
                if self.full:
                    for x in range(2,6):
                        path = base.replace("0.JPG", f"{x}.TIF")
                        band = self._load_and_normalize(path)
                        bands.append(band)
                    target = np.stack(bands, axis=-1)
                else:
                    for x in range(2,6):
                        path = base.replace("0_", f"{x}_").replace(".JPG", ".TIF")
                        band = self._load_and_normalize(path)
                        bands.append(band)
                    target = np.stack(bands, axis=-1)
            case "Weedy-Rice":
                breakpoint() #Not implemented




        # Convert to torch tensors and rearrange to [C, H, W]
        rgb = torch.from_numpy(rgb).permute(2, 0, 1).float()
        target = torch.from_numpy(target).permute(2, 0, 1).float()

        return {"rgb": rgb, "ms": target}

if __name__ == "__main__":
    print("Testing DataCarrier...")
    dataset = DataCarrier(root_dir="data/kz", load_data=load_east_kaz, data_type="Kazahkstan")
    print(dataset.__len__())
    sample = dataset[0]
    print("rgb patch shape:", sample["rgb"].shape)
    print("ms patch shape:", sample["ms"].shape)
