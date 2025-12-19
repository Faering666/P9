import re
from typing import Callable, Any
from pathlib import Path
from PIL import Image
import h5py
import numpy as np

"""
A loader should return a dictionary of dictionaries with layout:
{
    "id": {
        "cube": np.ndarray | torch.Tensor,
        "path": str | Path | None,
        "source": str
    }
}
"""

def get_loader(data_type: str, pred_path: str, truth_path: str):
    match data_type:
        case "Sri-Lanka":
            return make_sri_lanka_npy_loader(pred_path), make_sri_lanka_loader(truth_path)
        case "Kazakhstan":
            return make_kazakhstan_npy_loader(pred_path), make_kazakhstan_loader(truth_path)
        case "Weedy-Rice":
            return make_weedy_rice_npy_tif_loader(pred_path), make_weedy_rice_tif_loader(truth_path)
        case _: # default sri-lanka
            return make_sri_lanka_npy_loader(pred_path), make_sri_lanka_loader(truth_path)

def make_npy_loader(root_dir: str) -> Callable[[], dict[str, dict[str, str]]]:
    """
    Create a zero-arg loader that recursively scans `root_dir` for .npy files,
    loads them with np.load, casts to float32, and returns a dict:

        {
            "<stem>": {
                "cube": np.ndarray (float32),
                "path": "/abs/path/to/file.npy",
            },
            ...
        }

    where <stem> is the filename without extension.
    """
    
    root = Path(root_dir)
    
    def loader() -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}

        for npy_path in root.rglob("*.npy"):
            arr = np.load(npy_path).astype(np.float32)
            sid = npy_path.stem
            
            if sid in out:
                raise ValueError(
                    f"Duplicate id '{sid}' from files:\n"
                    f"  {out[sid]['path']}\n"
                    f"  {npy_path}"
                )

            out[sid] = {
                "cube": arr,
                "path": str(npy_path.resolve())
            }
            
        return out
    
    return loader

def make_mat_loader(root_dir: str, cube_key: str = "cube") -> Callable[[], dict[str, dict[str, str]]]:
    """
    Create a zero-arg loader that recursively scans `root_dir` for .mat files,
    reads them via h5py, converts the selected dataset to np.float32, and
    returns a dict:

        {
            "<stem>": {
                "cube": np.ndarray (float32),
                "path": "/abs/path/to/file.mat",
            },
            ...
        }

    If `dataset_key` is provided, that key is used in each .mat file.
    If `dataset_key` is None, the first top-level dataset that is not
    obviously metadata is used.
    """
    
    root = Path(root_dir)
    
    def loader() -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}

        for mat_path in root.rglob("*.mat"):
            with h5py.File(mat_path, "r") as f:
                dset = f[cube_key]
                arr = np.array(dset, dtype=np.float32)
            
            sid = mat_path.stem
            
            if sid in out:
                raise ValueError(
                    f"Duplicate id '{sid}' from files:\n"
                    f"  {out[sid]['path']}\n"
                    f"  {mat_path}"
                )

            out[sid] = {
                "cube": arr,
                "path": str(mat_path.resolve()),
            }
        
        return out
    
    return loader

def _load_tif_as_gray(path: Path) -> np.ndarray:
    """Load a TIF file as a float32 2D array (H, W)."""
    img = Image.open(path)
    arr = np.array(img, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[..., 0]

    return arr

def make_weedy_rice_tif_loader(root_dir: str) -> Callable[[], dict[str, dict[str, str]]]:
    """
    Create a zero-arg loader for the WeedyRice dataset.

    Expects TIF filenames like:
        "<some description>.<id>.TIF"
    where <id> has the form:
        "<number with leading zeros>m_<spectrum>"

    Examples:
        "whatever.004m_R.TIF"
        "foo.bar.999m_NIR.TIF"

    All files sharing the same numeric ID are grouped into a 4-band cube
    with channel order: ["G", "R", "RE", "NIR"].

    Returns a dict:
        {
            "<true_id>": {
                "cube": np.ndarray(float32, shape=(4, H, W)),
                "path": "/abs/path/to/one-band-file",
                "paths": { "G": "...", "R": "...", "RE": "...", "NIR": "..." },
            },
            ...
        }

    where <true_id> is the numeric ID as a string without leading zeros
    (e.g. "4" for "004m_R").
    """
    
    root = Path(root_dir)
    band_order = ["G", "R", "RE", "NIR"]

    id_pattern = re.compile(r"^(\d+)m_([A-Za-z0-9]+)$")
    
    def loader() -> dict[str, dict[str, str]]:
        grouped: dict[str, dict[str, Path]] = {}

        for tif_path in root.rglob("*"):
            if not tif_path.is_file():
                continue
            if tif_path.suffix != ".TIF":
                continue
            
            stem = tif_path.stem
            
            parts = stem.split(".")
            id_part = parts[-1]

            m = id_pattern.match(id_part)
            if m is None:
                # Not matching the format
                continue
            
            num_str, spectrum = m.groups()
            spectrum = spectrum.upper()
            
            if spectrum not in band_order:
                # Unknown spectrum
                continue
            
            grouped.setdefault(num_str, {})[spectrum] = tif_path
                
        out: dict[str, dict[str, Any]] = {}

        for true_id, spec_map in grouped.items():
            if not all(b in spec_map for b in band_order):
                # Only keep samples that have all required spectra
                continue
            
            layers: list[np.ndarray] = []
            for band in band_order:
                arr = _load_tif_as_gray(spec_map[band])
                layers.append(arr)

            cube = np.stack(layers, axis=0).astype(np.float32)

            # Representative path
            repr_path = spec_map[band_order[0]]

            out[true_id] = {
                "cube": cube,
                "path": str(repr_path.resolve()),
                "paths": {band: str(spec_map[band].resolve()) for band in band_order},
            }
                
        return out

    return loader

def make_weedy_rice_npy_tif_loader(root_dir: str) -> Callable[[], dict[str, dict[str, str]]]:
    pass

def make_sri_lanka_loader(root_dir: str) -> Callable[[], dict[str, dict[str, str]]]:
    root = Path(root_dir)
    band_order = ["G", "R", "RE", "NIR"]

    id_pattern = re.compile(r"^(.*)_MS_([A-Za-z0-9]+)$")
    
    def loader() -> dict[str, dict[str, str]]:
        grouped: dict[str, dict[str, Path]] = {}

        for tif_path in root.rglob("*"):
            if not tif_path.is_file():
                continue
            if tif_path.suffix != ".TIF":
                continue
            
            stem = tif_path.stem
            
            m = id_pattern.match(stem)
            if m is None:
                # Not matching the format
                continue
            
            num_str, spectrum = m.groups()
            spectrum = spectrum.upper()
            
            if spectrum not in band_order:
                # Unknown spectrum
                continue
            
            grouped.setdefault(num_str, {})[spectrum] = tif_path
                
        out: dict[str, dict[str, Any]] = {}

        for true_id, spec_map in grouped.items():
            if not all(b in spec_map for b in band_order):
                # Only keep samples that have all required spectra
                continue
            
            layers: list[np.ndarray] = []
            for band in band_order:
                arr = _load_tif_as_gray(spec_map[band])
                layers.append(arr)

            cube = np.stack(layers, axis=0).astype(np.float32)

            # Representative path
            repr_path = spec_map[band_order[0]]

            out[true_id] = {
                "cube": cube,
                "path": str(repr_path.resolve()),
                "paths": {band: str(spec_map[band].resolve()) for band in band_order},
            }
                
        return out

    return loader

def make_sri_lanka_npy_loader(root_dir: str) -> Callable[[], dict[str, dict[str, str]]]:
    root = Path(root_dir)
    
    def loader() -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}

        for npy_path in root.rglob("*.npy"):
            arr = np.load(npy_path).astype(np.float32)
            sid = npy_path.stem[:-2]
            
            if sid in out:
                raise ValueError(
                    f"Duplicate id '{sid}' from files:\n"
                    f"  {out[sid]['path']}\n"
                    f"  {npy_path}"
                )

            out[sid] = {
                "cube": arr,
                "path": str(npy_path.resolve())
            }
            
        return out
    
    return loader

def make_kazakhstan_loader(root_dir: str) -> Callable[[], dict[str, dict[str, str]]]:
    raise NotImplementedError("Kazakhstan loader not implemented")

def make_kazakhstan_npy_loader(root_dir: str) -> Callable[[], dict[str, dict[str, str]]]:
    raise NotImplementedError("Kazakhstan npy loader not implemented")