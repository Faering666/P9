# SPDX-License-Identifier: MIT
# Copyright (c) 2025 <Hugin J. Zachariasen, Magnus H. Jensen, Martin C. B. Nielsen, Tobias S. Madsen>.

import torch
from torchmetrics.functional.regression import mean_squared_error
from torchmetrics.functional.image import (
    peak_signal_noise_ratio,
    structural_similarity_index_measure,
    spectral_angle_mapper,
)


class MetricCalculator:
    """
    Compute metrics for a *single* prediction-ground-truth pair.

    Metrics:
      - MRAE, RMSE              on the full cube (pred vs gt)
      - PSNR, SSIM, SAM         on the full cube (pred vs gt)

      - NDVI_PRED, NDVI_GT      mean NDVI per image (no cross-use)
      - NDRE_PRED, NDRE_GT      mean NDRE per image (no cross-use)

    Expected input shapes per call (pred, gt):
      - (C, H, W)    (e.g. [bands, H, W])
      - (H, W, C)
      - (B, C, H, W) (if you pass a batch; metrics are averaged over all)
    """

    def __init__(
        self,
        data_range: float = 1.0,
        nir_index: int = 0,
        red_index: int = 1,
        rededge_index: int = 2,
        eps: float = 0,
        device: str | None = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.data_range = data_range
        self.nir_index = nir_index
        self.red_index = red_index
        self.rededge_index = rededge_index
        self.eps = eps

    # ---------- public API ---------- #
    @torch.no_grad()
    def compute(self, pred, gt) -> dict[str, float]:
        """
        Compute all metrics for a single (pred, gt) pair.

        Returns a dict of Python floats.
        """
        pred_t, gt_t = self._prepare_tensors(pred, gt)   # -> (B, C, H, W)

        # --- cube-level error metrics ---
        mrae_val = self._mrae(pred_t, gt_t)

        mse_val = mean_squared_error(pred_t, gt_t)
        rmse_val = torch.sqrt(mse_val)

        psnr_val = peak_signal_noise_ratio(
            pred_t, gt_t, data_range=self.data_range
        )
        ssim_val = structural_similarity_index_measure(
            pred_t, gt_t, data_range=self.data_range
        )
        sam_val = spectral_angle_mapper(pred_t, gt_t)

        # --- NDVI / NDRE per-image scores (no cross-use) ---
        ndvi_pred_map, ndvi_gt_map = self._compute_ndvi(pred_t, gt_t)
        ndre_pred_map, ndre_gt_map = self._compute_ndre(pred_t, gt_t)

        # mean over batch + spatial dimensions
        ndvi_pred_mean = ndvi_pred_map.mean()
        ndvi_gt_mean   = ndvi_gt_map.mean()

        ndre_pred_mean = ndre_pred_map.mean()
        ndre_gt_mean   = ndre_gt_map.mean()

        # Convert all to plain floats on CPU
        return {
            "MRAE": float(mrae_val.cpu()),
            "MSE" : float(torch.pow(rmse_val, 2).cpu()),
            "RMSE": float(rmse_val.cpu()),
            "PSNR": float(psnr_val.cpu()),
            "SSIM": float(ssim_val.cpu()),
            "SAM": float(sam_val.cpu()),
            "NDVI_PRED": float(ndvi_pred_mean.cpu()),
            "NDVI_GT": float(ndvi_gt_mean.cpu()),
            "NDRE_PRED": float(ndre_pred_mean.cpu()),
            "NDRE_GT": float(ndre_gt_mean.cpu()),
        }

    # ---------- internal helpers ---------- #
    def _prepare_tensors(self, pred, gt):
        # Accept numpy or torch
        if not isinstance(pred, torch.Tensor):
            pred = torch.as_tensor(pred)
        if not isinstance(gt, torch.Tensor):
            gt = torch.as_tensor(gt)

        pred = pred.to(self.device, dtype=torch.float32)
        gt = gt.to(self.device, dtype=torch.float32)

        # Handle (H, W, C) -> (C, H, W)
        if pred.ndim == 3 and pred.shape[-1] == gt.shape[-1] and pred.shape[-1] <= 16:
            pred = pred.permute(2, 0, 1)
            gt = gt.permute(2, 0, 1)

        # If 3D, treat as single sample: (C, H, W) -> (1, C, H, W)
        if pred.ndim == 3:
            pred = pred.unsqueeze(0)
            gt = gt.unsqueeze(0)
        elif pred.ndim != 4:
            raise ValueError(
                f"Expected pred/gt to be 3D or 4D (C,H,W) or (B,C,H,W), "
                f"got {pred.shape=}, {gt.shape=}"
            )

        if pred.shape != gt.shape:
            raise ValueError(f"Shape mismatch: pred {pred.shape}, gt {gt.shape}")

        # Validate band indices
        b, c, h, w = pred.shape
        for idx, name in [
            (self.nir_index, "nir_index"),
            (self.red_index, "red_index"),
            (self.rededge_index, "rededge_index"),
        ]:
            if not (0 <= idx < c):
                raise ValueError(
                    f"{name}={idx} is out of range for cube with {c} channels"
                )

        return pred, gt

    def _mrae(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """Mean Relative Absolute Error = mean(|pred - gt| / (|gt| + eps))."""
        pred = pred.float()
        gt = gt.float()
        rel_err = (pred - gt).abs() / (gt.abs() + self.eps)
        return rel_err.mean()

    def _compute_ndvi(self, pred: torch.Tensor, gt: torch.Tensor):
        """
        pred, gt: (B, C, H, W)
        Returns:
          ndvi_pred, ndvi_gt: (B, H, W) each
        """
        nir_p = pred[:, self.nir_index, :, :]
        red_p = pred[:, self.red_index, :, :]
        nir_g = gt[:, self.nir_index, :, :]
        red_g = gt[:, self.red_index, :, :]

        ndvi_pred = (nir_p - red_p) / (nir_p + red_p + self.eps)
        ndvi_gt   = (nir_g - red_g) / (nir_g + red_g + self.eps)

        return ndvi_pred, ndvi_gt

    def _compute_ndre(self, pred: torch.Tensor, gt: torch.Tensor):
        """
        pred, gt: (B, C, H, W)
        Returns:
          ndre_pred, ndre_gt: (B, H, W) each
        """
        nir_p = pred[:, self.nir_index, :, :]
        re_p  = pred[:, self.rededge_index, :, :]
        nir_g = gt[:, self.nir_index, :, :]
        re_g  = gt[:, self.rededge_index, :, :]

        ndre_pred = (nir_p - re_p) / (nir_p + re_p + self.eps)
        ndre_gt   = (nir_g - re_g) / (nir_g + re_g + self.eps)

        return ndre_pred, ndre_gt
