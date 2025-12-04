from warnings import catch_warnings
import numpy as np
import os
import json
import glob
import h5py  # for reading .mat files saved like in your dataset script


class MetricCalculator:
    """
    NumPy implementations of the metrics from utils.py, plus MSE.

    Supports both:
    - Single image:  (H, W, C) or (C, H, W)
    - Batch images:  (N, C, H, W) or (N, H, W, C)

    By default, PSNR is computed with data_range=255 to match the original
    Loss_PSNR implementation in utils.py.
    """

    def __init__(
        self,
        data_range=255.0,
        eps=1e-8,
        ndvi_red_idx: int = 25, # 650
        ndvi_nir_idx: int = 30, # 700
        channel_axis: int = 0,
        ):
        """
        data_range: used for PSNR (e.g. 255.0 or 1.0)
        eps:        small constant to avoid division by zero
        ndvi_red_idx / ndvi_nir_idx:
            indices of the RED and NIR bands along `channel_axis`.
            If either is None, NDVI metric will be disabled.
        channel_axis:
            which axis is the spectral/channel dimension:
            - 0 for (C,H,W)
            - 1 for (N,C,H,W)
            - 2 or -1 if you use (H,W,C) / (N,H,W,C), etc.
        """
        self.data_range = float(data_range)
        self.eps = float(eps)
        self.ndvi_red_idx: int = ndvi_red_idx
        self.ndvi_nir_idx: int = ndvi_nir_idx
        self.channel_axis = channel_axis

    # ---------- internal helpers ----------

    @staticmethod
    def _to_numpy(x):
        """Convert input to float32 NumPy array."""
        return np.asarray(x, dtype=np.float32)

    @staticmethod
    def _ensure_same_shape(pred, target):
        if pred.shape != target.shape:
            raise ValueError(
                f"Shape mismatch: pred {pred.shape} vs target {target.shape}"
            )

    @staticmethod
    def _add_batch_dim_if_needed(arr):
        """
        If arr has no batch dimension (e.g. HWC or CHW),
        treat it as a batch of size 1.
        """
        if arr.ndim == 3:
            return arr[None, ...]  # (1, C, H, W) or (1, H, W, C)
        elif arr.ndim < 3:
            raise ValueError(
                f"Expected at least 3D array (C,H,W or H,W,C), got shape {arr.shape}"
            )
        return arr

    # ---------- metrics ----------

    def mse(self, pred, target):
        """
        Mean Squared Error over all pixels & channels.
        """
        pred = self._to_numpy(pred)
        target = self._to_numpy(target)
        self._ensure_same_shape(pred, target)

        diff = pred - target
        return float(np.mean(diff ** 2))

    def rmse(self, pred, target):
        """
        Root Mean Squared Error (sqrt of MSE).
        Mirrors Loss_RMSE in utils.py (but using NumPy).
        """
        return float(np.sqrt(self.mse(pred, target)))

    def mrae(self, pred, target):
        """
        Mean Relative Absolute Error:
            mean( |pred - target| / target )

        We add eps in the denominator to avoid division by zero,
        which is slightly more numerically stable than the original.
        """
        pred = self._to_numpy(pred)
        target = self._to_numpy(target)
        self._ensure_same_shape(pred, target)

        denom = np.abs(target) + self.eps
        rel_err = np.abs(pred - target) / denom
        return float(np.mean(rel_err))

    def psnr(self, pred, target):
        """
        Peak Signal-to-Noise Ratio.

        Matches the original Loss_PSNR semantics:
        - clamp inputs to [0, 1]
        - multiply by data_range (default 255)
        - compute MSE over all pixels per image
        - take PSNR = mean over batch
        """
        pred = self._to_numpy(pred)
        target = self._to_numpy(target)
        self._ensure_same_shape(pred, target)

        # Clamp to [0, 1] like in Loss_PSNR
        pred = np.clip(pred, 0.0, 1.0)
        target = np.clip(target, 0.0, 1.0)

        # Apply data_range
        pred = pred * self.data_range
        target = target * self.data_range

        # Ensure batch dimension
        pred = self._add_batch_dim_if_needed(pred)
        target = self._add_batch_dim_if_needed(target)

        # Flatten per sample: (N, -1)
        N = pred.shape[0]
        pred_flat = pred.reshape(N, -1)
        target_flat = target.reshape(N, -1)

        mse_per_sample = np.mean((pred_flat - target_flat) ** 2, axis=1)

        # Avoid division by zero
        mse_per_sample = np.maximum(mse_per_sample, self.eps)

        psnr_per_sample = 10.0 * np.log10((self.data_range ** 2) / mse_per_sample)
        return float(np.mean(psnr_per_sample))
    
    # -------------- NDVI --------------
    def _ndvi_map(self, cube: np.ndarray) -> np.ndarray:
        """
        Compute NDVI map from a hyperspectral cube.

        cube: shape (..., C, H, W) or (..., H, W, C) depending on channel_axis.
              We support 3D or 4D (batch) arrays.
        Returns:
            NDVI map with shape matching cube except channel_axis removed,
            e.g. (H,W) or (N,H,W).
        """
        if self.ndvi_red_idx is None or self.ndvi_nir_idx is None:
            raise RuntimeError(
                "NDVI indices not set; please pass ndvi_red_idx and "
                "ndvi_nir_idx to MetricCalculator.__init__"
            )

        cube = self._to_numpy(cube)
        if cube.ndim not in (3, 4):
            raise ValueError(f"NDVI expects 3D or 4D cube, got {cube.ndim}D")

        red = np.take(cube, self.ndvi_red_idx, axis=self.channel_axis)
        nir = np.take(cube, self.ndvi_nir_idx, axis=self.channel_axis)

        # Both red and nir now have shape (..., H, W)
        ndvi = (nir - red) / (nir + red + self.eps)
        return ndvi

    def ndvi_rmse(self, pred, target):
        """
        RMSE between predicted and ground-truth NDVI maps.
        Returns a single scalar.

        This is what will be reported as 'NDVI_RMSE' in compute_all().
        """
        pred = self._to_numpy(pred)
        target = self._to_numpy(target)
        self._ensure_same_shape(pred, target)

        ndvi_pred = self._ndvi_map(pred)
        ndvi_true = self._ndvi_map(target)

        diff = ndvi_pred - ndvi_true
        return float(np.sqrt(np.mean(diff ** 2)))
        
    def _ndvi_mean(self, cube: np.ndarray) -> float:
        """
        Mean NDVI over all pixels (and batch elements if present).
        This is the scalar 'NDVI score'
        """
        ndvi = self._ndvi_map(cube)
        return float(np.mean(ndvi))

    

    def compute_all(self, pred, target):
        """
        Convenience method to get all metrics at once.
        Returns a dict of floats.
        """
        return {
            "MRAE": self.mrae(pred, target),
            "RMSE": self.rmse(pred, target),
            "MSE": self.mse(pred, target),
            "PSNR": self.psnr(pred, target),
            "NDVI_PRED": self._ndvi_mean(pred),
            "NDVI_GT": self._ndvi_mean(target),
            "NDVI_RMSE": self.ndvi_rmse(pred, target)
        }

class DirectoryMetricEvaluator:
    def __init__(
        self,
        result_path: str,
        correct_path: str,
        metric_calculator: MetricCalculator | None = None,
        gt_key: str = "cube",
        pred_ext: str = "npy",
        gt_ext: str = "mat",
    ):
        self.result_path = result_path
        self.correct_path = correct_path
        self.gt_key = gt_key
        self.pred_ext = pred_ext
        self.gt_ext = gt_ext
        self.metrics = metric_calculator or MetricCalculator()

    # ---- your helper, minimally adapted into the class ----
    def _scan_dir_for_extension(self, dir_path: str, file_extension: str) -> list[dict[str, str]]:
        """
        Recursively find all files with given extension under dir_path.
        Returns list of dicts:
            {"name": <basename without ext>, "<ext>": <abs path>}
        """
        results: list[dict[str, str]] = []

        pattern = os.path.join(dir_path, "**", f"*.{file_extension}")
        for path in glob.glob(pattern, recursive=True):
            if os.path.isdir(path):
                continue

            base = os.path.basename(path)
            name = os.path.splitext(base)[0]

            results.append({
                "name": name,
                file_extension: os.path.abspath(path),
            })

        return results
    
    def _match_pairs(
        self,
        npy_list: list[dict[str, str]],
        mat_list: list[dict[str, str]],
    ) -> tuple[list[dict[str, str]], list[str], list[str]]:
        """
        Join the npy_list and mat_list on "name".

        Returns:
            pairs:    [{"name": ..., "npy": ..., "mat": ...}, ...]  (only common names)
            only_npy: [name, ...]  names that had .npy but no .mat
            only_mat: [name, ...]  names that had .mat but no .npy
        """
        npy_by_name = {item["name"]: item["npy"] for item in npy_list}
        mat_by_name = {item["name"]: item["mat"] for item in mat_list}

        npy_names = set(npy_by_name.keys())
        mat_names = set(mat_by_name.keys())

        common_names = npy_names & mat_names
        only_npy = sorted(npy_names - mat_names)
        only_mat = sorted(mat_names - npy_names)

        pairs: list[dict[str, str]] = []
        for name in sorted(common_names):
            pairs.append({
                "name": name,
                "npy": npy_by_name[name],
                "mat": mat_by_name[name],
            })

        return pairs, only_npy, only_mat

    # ---- I/O ----

    def _load_prediction(self, path: str) -> np.ndarray:
        """
        Load predicted hyperspectral image from .npy.
        Adjust this if you saved .mat instead.
        """
        arr = np.load(path)
        return arr.astype(np.float32)

    def _load_gt(self, path: str) -> np.ndarray:
        """
        Load ground truth from .mat with key self.gt_key.
        Mirrors the way you read 'cube' in the hsi_dataset loader.
        """
        with h5py.File(path, "r") as f:
            gt = np.array(f[self.gt_key], dtype=np.float32)

        # If your GT is stored as (C, W, H) like some HSI datasets,
        # you might need this transpose:
        if gt.ndim == 3 and gt.shape[1] != gt.shape[2]:
            # Heuristic: make it (C, H, W)
            gt = np.transpose(gt, (0, 2, 1))

        return gt
            

    # ---- main API ----

    def evaluate(self, results_file: str | None = None, verbose: bool = True):
        """
        Walk both directories, match files by basename, and compute mean metrics.

        Returns:
            dict: {
                "MRAE": float,
                "RMSE": float,
                "MSE": float,
                "PSNR": float
                "NDVI_PRED": float
                "NDVI_GT": float
                "NDVI_RMSE": float
            }
        """
        result_files = self._scan_dir_for_extension(self.result_path, self.pred_ext)
        correct_files = self._scan_dir_for_extension(self.correct_path, self.gt_ext)

        pairs, only_npy, only_mat = self._match_pairs(result_files, correct_files)

        if verbose:
            if only_npy:
                print(f"[INFO] Files with .{self.pred_ext} but no .{self.gt_ext}:")
                for name in only_npy:
                    print("  ", name)
            if only_mat:
                print(f"[INFO] Files with .{self.gt_ext} but no .{self.pred_ext}:")
                for name in only_mat:
                    print("  ", name)

        if not pairs:
            raise RuntimeError(
                "No common basenames found between prediction and GT directories."
            )

        per_file_scores: list[dict] = []

        for (i, pair) in enumerate(pairs):
            name = pair["name"]
            pred_path = pair["npy"]
            gt_path = pair["mat"]

            if verbose:
                print(f"Processing {i+1} / {len(result_files)}")

            try:
                pred = self._load_prediction(pred_path)
            except:
                print(f":: Skipping prediction: {pred_path}")
                continue

            try:
                gt = self._load_gt(gt_path)
            except:
                print(f":: Skipping ground truth: {gt_path}")
                continue

            # If needed, crop/align shapes here.
            if pred.shape != gt.shape:
                # Example: if one is (H, W, C) and the other (C, H, W)
                if pred.ndim == 3 and gt.ndim == 3 and pred.shape[::-1] == gt.shape:
                    gt = np.transpose(gt, (2, 1, 0))
                else:
                    raise ValueError(f"Shape mismatch for {name}: pred {pred.shape}, gt {gt.shape}")

            metrics_dict = self.metrics.compute_all(pred, gt)

            per_file_scores.append({
                "name": name,
                **{k: float(v) for k, v in metrics_dict.items()},
            })
            
        agg = {}
        metric_names = [k for k in per_file_scores[0].keys() if k != "name"]

        for metric_name in metric_names:
            agg[metric_name] = float(
                np.mean([r[metric_name] for r in per_file_scores])
            )

        if results_file is not None:
            with open(results_file, "w", encoding="utf-8") as f:
                json.dump(per_file_scores, f, indent=2)
            print(f"\nScores written to: {results_file}")

        return agg

    def evaluate_from_file(self, results_file: str) -> dict:
        """
        Load existing per-file scores (written by `evaluate`) and compute/print
        aggregate metrics, without reading any .npy/.mat.

        Args:
            results_file: path to JSON file with per-file scores.
            verbose:      if True, print aggregate metrics.
        """
        with open(results_file, "r", encoding="utf-8") as f:
            per_file_scores = json.load(f)

        if not per_file_scores:
            raise RuntimeError(f"No per-file scores found in '{results_file}'.")

        agg = {}
        metric_names = [k for k in per_file_scores[0].keys() if k != "name"]

        for metric_name in metric_names:
            agg[metric_name] = float(
                np.mean([r[metric_name] for r in per_file_scores])
            )

        return agg

        

if __name__ == "__main__":
    result_dir = "./class_exp/mst_plus_plus/mst_all"
    gt_dir = "C:/Users/tobia/Downloads/ARAD_1K_Mirror/*_spectral/"

    evaluator = DirectoryMetricEvaluator(
        result_path=result_dir,
        correct_path=gt_dir,
        metric_calculator=MetricCalculator(data_range=255.0),
        gt_key="cube",       # change if your .mat files use a different variable name
        pred_ext="npy",
        gt_ext="mat",
    )

    results_file = "./new-results.json"
    scores = evaluator.evaluate(results_file)
    print("MRAE:", scores["MRAE"])
    print("RMSE:", scores["RMSE"])
    print("MSE: ", scores["MSE"])
    print("PSNR:", scores["PSNR"])
    print("NDVI_PRED:", scores["NDVI_PRED"])
    print("NDVI_GT:", scores["NDVI_GT"])
    print("NDVI_RMSE:", scores["NDVI_RMSE"])
    
    # print("=========================")
    # scores = evaluator.evaluate_from_file(results_file)
    # print("MRAE:", scores["MRAE"])
    # print("RMSE:", scores["RMSE"])
    # print("MSE: ", scores["MSE"])
    # print("PSNR:", scores["PSNR"])
    # print("NDVI_PRED:", scores["NDVI_PRED"])
    # print("NDVI_GT:", scores["NDVI_GT"])
    # print("NDVI_RMSE:", scores["NDVI_RMSE"])