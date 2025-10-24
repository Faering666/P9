import os
import time
import math
import glob
import uuid
import warnings
import numpy as np
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
from PIL import Image
from scipy.io import savemat
from architecture import model_generator


def _suppress_warnings():
    def _noop(*args, **kwargs): pass
    warnings.warn = _noop

class ModelRunner:
    # ---------- Construction ----------
    def __init__(
        self,
        method: str = "mst_plus_plus",
        model_path: str = "./model_zoo/mst_plus_plus.pth",
        save_dir: str = "./results/",
        device: str | None = None,
        pad_multiple: int = 16,
        # saving toggles
        save_mat: bool = False,
        save_npy: bool = False,
        save_gray_grid: bool = False,
        save_color_grid: bool = False,
        save_gray_band: bool = False,
        save_color_band: bool = False,
        # visualization settings
        stretch_low: float = 1.0,
        stretch_high: float = 99.0,
        wavelengths: list[int] | None = None,  # if None, auto (400-700nm)
        verbose: bool = False,
        suppress_lib_warnings: bool = True,
        prefix_mode: str = "guid",
        guid_length: int = 8,
        batch_id_mode: str = "guid",
        batch_guid_length: int = 8,
    ):
        if suppress_lib_warnings:
            _suppress_warnings()

        self.method = method
        self.model_path = model_path
        self.save_dir = save_dir
        self._model_dir = os.path.join(self.save_dir, self.method)
        self.pad_multiple = pad_multiple

        self.save_mat = save_mat
        self.save_npy = save_npy
        self.save_gray_grid = save_gray_grid
        self.save_color_grid = save_color_grid
        self.save_gray_band = save_gray_band
        self.save_color_band = save_color_band

        self.stretch_low = stretch_low
        self.stretch_high = stretch_high
        self._wavelengths = wavelengths  # defer validation until we know C
        self.verbose = verbose
        
        self.prefix_mode = prefix_mode
        self.guid_length = max(1, int(guid_length))
        self.batch_id_mode = batch_id_mode
        self.batch_guid_length = max(1, int(batch_guid_length))

        cudnn.benchmark = True
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self._log(f"PyTorch device: {self.device}")
        if self.device.type == "cuda":
            self._log(f"CUDA available, using: {torch.cuda.get_device_name(torch.cuda.current_device())}", True)

        # build/load model
        self._log(f"Building model '{self.method}' and loading weights: {self.model_path}", True)
        self.model = model_generator(self.method, self.model_path).to(self.device).eval()
        self._log(" :: Model ready (eval mode).", True)

        # ensure base save dir
        self._ensure_dir(self.save_dir)
        self._ensure_dir(self._model_dir)
        self._log(f"## Results will be written to: {self._model_dir}")

    # ---------- Public API ----------
    def infer_one(
        self,
        image: str | np.ndarray | torch.Tensor | None = None,
        *,
        use_random_input: bool = False,
        random_size: tuple[int, int] = (512, 512),
        base_name: str | None = None,
        return_arrays: bool = False,
        prefix: str | None = None,
        batch_id: str | None = None,
        batch_label: str | None = None,
    ) -> dict:
        t_all = time.time()
        
        
        batch_id = batch_id or self._make_batch_id()
        batch_dir = self._make_batch_dir(batch_id, batch_label)
        self._ensure_dir(batch_dir)

        # ---- Prepare input tensor (1x3xHxW) ----
        if use_random_input:
            H, W = random_size
            self._log(f"Generating random input: 1x3x{H}x{W}")
            x = torch.rand(1, 3, H, W, dtype=torch.float32)
            base = base_name or "random"
        else:
            x, base = self._to_tensor(image, base_name)

        x = x.to(self.device)
        self._log(f"Input on device {self.device}: {tuple(x.shape)}")

        # ---- Pad ----
        x_pad, pads = self._pad_to_multiple(x, multiple=self.pad_multiple, mode="reflect")

        # ---- Forward ----
        self._log("Forward pass...")
        torch.cuda.synchronize() if self.device.type == "cuda" else None
        t0 = time.time()
        with torch.no_grad():
            y_pad = self.model(x_pad)  # 1 x C x H x W
        torch.cuda.synchronize() if self.device.type == "cuda" else None
        t1 = time.time()
        self._log(f" :: Done in {(t1 - t0):.2f} s. Output(padded)={tuple(y_pad.shape)}")

        # ---- Unpad & clamp ----
        self._log("Unpadding and clamping to [0,1]")
        y = self._unpad(y_pad, pads)
        y = torch.clamp(y, 0.0, 1.0)
        self._log(" :: Done")

        # ---- To numpy: CHW -> HWC ----
        cube_chw = y.squeeze(0).detach().cpu().numpy()     # (C,H,W)
        C, H, W = cube_chw.shape
        cube_hwc = np.transpose(cube_chw, (1, 2, 0))       # (H,W,C)
        self._log(f"Cube (H,W,C)={cube_hwc.shape}; dtype={cube_hwc.dtype}")
        if C != 31:
            self._log("Note: band count != 31 (ok if your checkpoint differs).")

        # ---- Wavelengths ----
        wavelengths = self._resolve_wavelengths(C)
        self._log(f"Wavelengths: {wavelengths[0]}-{wavelengths[-1]} nm (count={len(wavelengths)})")

        # ---- Id ----
        run_prefix = prefix if (prefix is not None) else self._make_prefix()
        base_tag = f"{run_prefix}_{base}" if run_prefix else base
        image_dir = os.path.join(batch_dir, base_tag)
        self._ensure_dir(image_dir)
        self._log(f"Batch dir: {batch_dir}")
        self._log(f"Image subdir: {image_dir}")
        

        # ---- Save numerics ----
        paths = {}
        if self.save_mat:
            p = os.path.join(image_dir, f"{base_tag}_mstpp.mat")
            self._log(f"Saving MAT to: {p}")
            savemat(p, {"cube": cube_hwc.astype(np.float32)})
            self._log(f" :: MAT saved")
            paths["mat"] = p
        if self.save_npy:
            p = os.path.join(image_dir, f"{base_tag}_mstpp.npy")
            self._log(f"Saving NPY to: {p}")
            np.save(p, cube_hwc.astype(np.float32))
            self._log(f" :: NPY saved")
            paths["npy"] = p

        # ---- Visualizations ----
        if self.save_gray_grid:
            p = os.path.join(image_dir, f"{base_tag}_bands_gray_grid.png")
            self._save_grayscale_grid(cube_hwc, p, "MST++ Output (All Bands)")
            paths["gray_grid"] = p

        if self.save_color_grid:
            p = os.path.join(image_dir, f"{base_tag}_bands_color_grid.png")
            self._save_color_band_grid(cube_hwc, wavelengths, p, "MST++ Output (Colorized Bands)")
            paths["color_grid"] = p

        if self.save_gray_band:
            gray_dir = os.path.join(image_dir, f"{base_tag}_bands_gray")
            self._save_grayscale_bands(cube_hwc, gray_dir)
            paths["gray_dir"] = gray_dir

        if self.save_color_band:
            color_dir = os.path.join(image_dir, f"{base_tag}_bands_color")
            self._save_colorized_bands(cube_hwc, wavelengths, color_dir)
            paths["color_dir"] = color_dir

        stats = {
            "bands": int(C),
            "min": float(cube_hwc.min()),
            "max": float(cube_hwc.max()),
            "mean": float(cube_hwc.mean()),
            "forward_ms": (t1 - t0) * 1000.0,
            "total_s": time.time() - t_all,
        }

        result = {
            "id": run_prefix,
            "base": base,
            "base_tag": base_tag,
            "image_dir": image_dir,
            "paths": paths,
            "stats": stats
        }
        if return_arrays:
            result["cube_hwc"] = cube_hwc
            result["cube_chw"] = cube_chw
            result["wavelengths"] = wavelengths
        return result

    def infer_many(
        self,
        images: list[str] | str,
        *,
        glob_recursive: bool = False,
        return_arrays: bool = False,
        batch_id: str | None = None,
        batch_label: str | None = None,
        limit: int | None = None,
    ) -> list[dict]:
        # normalize list of paths
        if isinstance(images, str):
            paths = sorted(glob.glob(images, recursive=glob_recursive))
        else:
            paths = list(images)

        if limit is not None:
            paths = paths[:limit]

        batch_id = batch_id or self._make_batch_id()
        batch_dir = self._make_batch_dir(batch_id, batch_label)
        self._ensure_dir(batch_dir)
        self._log(f"Batch dir: {batch_dir}   (images={len(paths)})")

        outs = []
        for p in paths:
            try:
                self._log(f"Evaluating '{p}'", True)
                res = self.infer_one(
                                     p,
                                     batch_id=batch_id,
                                     batch_label=batch_label,
                                     return_arrays=return_arrays)
                outs.append(res)
                self._log(" :: Done", True)
            except Exception as e:
                self._log(f"[WARN] Skipping '{p}' due to error: {e}")
        return outs

    # ---------- Helpers (I/O, math, viz) ----------
    def _to_tensor(self, image, base_name=None) -> tuple[torch.Tensor, str]:
        """
        Accepts path, ndarray, or tensor. Returns (1x3xHxW tensor, base_name).
        """
        if isinstance(image, str):
            self._log(f"Loading image: {image}")
            img = Image.open(image).convert("RGB")
            arr = np.asarray(img)
            base = base_name or os.path.splitext(os.path.basename(image))[0]
        elif isinstance(image, np.ndarray):
            self._log("Loading np array")
            arr = image
            if arr.ndim == 3 and arr.shape[2] == 3:
                pass
            else:
                raise ValueError("ndarray must be HxWx3")
            base = base_name or "array"

        elif torch.is_tensor(image):
            self._log("Loading tensor")
            t = image
            if t.ndim == 4 and t.shape[0] == 1 and t.shape[1] == 3:
                scale = 255.0 if t.max().item() > 1.0 else 1.0
                return t.float().clamp(0, 255) / scale, (base_name or "tensor")
                
            if t.ndim == 3 and t.shape[0] == 3:
                scale = 255.0 if t.max().item() > 1.0 else 1.0
                return t.unsqueeze(0).float().clamp(0, 255) / scale, (base_name or "tensor")
                
            if t.ndim == 3 and t.shape[2] == 3:
                arr = t.cpu().numpy()
                base = base_name or "tensor"
            else:
                raise ValueError("tensor must be 3xHxW, 1x3xHxW, or HxWx3")
        else:
            raise ValueError("image must be a path, ndarray, or torch.Tensor")

        arr = arr.astype(np.float32)
        if arr.max() > 1.001:  # assume 0..255
            arr = arr / 255.0
        # to NCHW
        nchw = torch.from_numpy(np.transpose(arr, (2, 0, 1))).unsqueeze(0)  # 1 3 H W
        self._log(f" :: Loaded. Tensor shape={tuple(nchw.shape)}")
        return nchw, base

    def _ensure_dir(self, d: str):
        os.makedirs(d, exist_ok=True)

    def _pad_to_multiple(self, tensor, multiple=16, mode="reflect"):
        _, _, H, W = tensor.shape
        pad_h = (multiple - H % multiple) % multiple
        pad_w = (multiple - W % multiple) % multiple
        
        self._log(f"Padding check (mult={multiple}): pad_h={pad_h}, pad_w={pad_w}")
        if pad_h == 0 and pad_w == 0:
            self._log(" :: No padding needed.")
            return tensor, (0, 0, 0, 0)
        padded = F.pad(tensor, (0, pad_w, 0, pad_h), mode=mode)  # (l,r,t,b)
        self._log(f" :: Padded -> {tuple(padded.shape)}")
        return padded, (0, pad_w, 0, pad_h)

    def _unpad(self, tensor, pads):
        l, r, t, b = pads
        _, _, H, W = tensor.shape
        self._log(f" :: Unpad (l={l}, r={r}, t={t}, b={b}) from {H}x{W}")
        return tensor[:, :, t: H - b if b else H, l: W - r if r else W]

    def _contrast_stretch(self, x):
        lo = np.percentile(x, self.stretch_low)
        hi = np.percentile(x, self.stretch_high)
        if hi <= lo:
            return np.clip(x, 0.0, 1.0)
        y = (x - lo) / (hi - lo)
        return np.clip(y, 0.0, 1.0)

    def _save_grayscale_grid(self, cube_hwc, out_path, title):
        self._log(f"Saving grayscale grid: {out_path}")
        H, W, C = cube_hwc.shape
        cols = math.ceil(math.sqrt(C))
        rows = math.ceil(C / cols)

        fig, axs = plt.subplots(rows, cols, figsize=(cols * 3.0, rows * 3.0))
        if rows == 1 and cols == 1:
            axs = np.array([[axs]])
        elif rows == 1 or cols == 1:
            axs = np.atleast_2d(axs)

        band = 0
        for r in range(rows):
            for c in range(cols):
                ax = axs[r, c]
                ax.axis("off")
                if band < C:
                    img = self._contrast_stretch(cube_hwc[..., band])
                    ax.imshow(img, cmap="gray", vmin=0.0, vmax=1.0)
                    ax.set_title(f"Band {band:02d}", fontsize=10)
                    band += 1
                else:
                    ax.imshow(np.zeros((H, W)), cmap="gray", vmin=0.0, vmax=1.0)
        fig.suptitle(title, fontsize=12)
        fig.tight_layout(rect=(0, 0.02, 1, 0.98))
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        
        self._log(" :: Saving complete")

    def _save_color_band_grid(self, cube_hwc, wavelengths, out_path, title):
        self._log(f"Saving colorized grid: {out_path}")
        H, W, C = cube_hwc.shape
        assert len(wavelengths) == C
        cols = math.ceil(math.sqrt(C))
        rows = math.ceil(C / cols)

        colors = np.stack([self._wavelength_to_srgb(w) for w in wavelengths], axis=0)  # Cx3

        fig, axs = plt.subplots(rows, cols, figsize=(cols * 3.0, rows * 3.0))
        if rows == 1 and cols == 1:
            axs = np.array([[axs]])
        elif rows == 1 or cols == 1:
            axs = np.atleast_2d(axs)

        band = 0
        for r in range(rows):
            for c in range(cols):
                ax = axs[r, c]
                ax.axis("off")
                if band < C:
                    band_img = self._contrast_stretch(cube_hwc[..., band])
                    rgb = band_img[..., None] * colors[band][None, None, :]
                    ax.imshow(np.clip(rgb, 0.0, 1.0))
                    ax.set_title(f"Band {band:02d} ({wavelengths[band]} nm)", fontsize=10)
                    band += 1
                else:
                    ax.imshow(np.zeros((H, W, 3)), vmin=0.0, vmax=1.0)
        fig.suptitle(title, fontsize=12)
        fig.tight_layout(rect=(0, 0.02, 1, 0.98))
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        
        self._log(" :: Saving complete")

    def _save_grayscale_bands(self, cube_hwc, out_dir):
        self._ensure_dir(out_dir)
        H, W, C = cube_hwc.shape
        self._log(f"Saving grayscale bands -> {out_dir}")
        for k in range(C):
            band = self._contrast_stretch(cube_hwc[..., k])
            band8 = (band * 255.0 + 0.5).astype(np.uint8)
            Image.fromarray(band8).save(os.path.join(out_dir, f"band_{k:02d}.png"))

        self._log(" :: Saving complete")

    def _save_colorized_bands(self, cube_hwc, wavelengths, out_dir):
        self._ensure_dir(out_dir)
        H, W, C = cube_hwc.shape
        assert len(wavelengths) == C
        self._log(f"Saving colorized bands -> {out_dir}")
        colors = np.stack([self._wavelength_to_srgb(w) for w in wavelengths], axis=0)  # Cx3
        for k in range(C):
            band = self._contrast_stretch(cube_hwc[..., k])
            rgb = band[..., None] * colors[k][None, None, :]
            rgb8 = (np.clip(rgb, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
            Image.fromarray(rgb8).save(os.path.join(out_dir, f"band_{k:02d}.png"))

        self._log(" :: Saving complete")

    # ---------- Wavelength utils ----------
    def _resolve_wavelengths(self, C: int) -> list[int]:
        if self._wavelengths is not None:
            if len(self._wavelengths) != C:
                raise ValueError(f"Provided wavelengths length ({len(self._wavelengths)}) != band count ({C})")
            return list(self._wavelengths)
        # default: 31 bands from 400-700 nm (10 nm step); else linear
        if C == 31:
            return [400 + 10 * i for i in range(31)]
        return np.linspace(400, 700, C).astype(int).tolist()

    def _wavelength_to_srgb(self, nm: float) -> np.ndarray:
        w = float(np.clip(nm, 380.0, 780.0))
        if w < 380 or w > 780:
            r = g = b = 0.0
        elif w < 440:
            r = -(w - 440.0) / (440.0 - 380.0); g = 0.0; b = 1.0
        elif w < 490:
            r = 0.0; g = (w - 440.0) / (490.0 - 440.0); b = 1.0
        elif w < 510:
            r = 0.0; g = 1.0; b = -(w - 510.0) / (510.0 - 490.0)
        elif w < 580:
            r = (w - 510.0) / (580.0 - 510.0); g = 1.0; b = 0.0
        elif w < 645:
            r = 1.0; g = -(w - 645.0) / (645.0 - 580.0); b = 0.0
        else:
            r = 1.0; g = 0.0; b = 0.0

        if w < 420:
            factor = 0.3 + 0.7 * (w - 380.0) / (420.0 - 380.0)
        elif w > 700:
            factor = 0.3 + 0.7 * (780.0 - w) / (780.0 - 700.0)
        else:
            factor = 1.0

        gamma = 0.8
        def adj(c): 
            c = max(c * factor, 0.0)
            return c ** gamma
        return np.array([adj(r), adj(g), adj(b)], dtype=np.float32)

    # ---------- Misc ----------
    def _make_prefix(self) -> str:
        if self.prefix_mode == "guid":
            return uuid.uuid4().hex[:self.guid_length]
        if self.prefix_mode == "timestamp":
            return time.strftime("%y%m%d-%H%M%S")
        return ""

    def _make_batch_id(self) -> str:
        if self.batch_id_mode == "guid":
            return uuid.uuid4().hex[:self.batch_guid_length]

        return time.strftime("%y%m%d-%H%M%S")

    def _make_batch_dir(self, batch_id: str, batch_label: str | None) -> str:
        name = f"batch_{batch_id}"
        if batch_label:
            name = f"{name}_{batch_label}"

        return os.path.join(self._model_dir, name)

    def _log(self, msg: str, force: bool = False):
        if self.verbose or force:
            print(msg, flush=True)
