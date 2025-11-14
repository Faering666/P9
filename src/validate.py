import glob
import os
import time
import numpy as np
import h5py
import csv
import pandas as pd


class Validate:
    """
    End-to-end evaluator for hyperspectral cube predictions (.npy)
    against ground truth cubes stored in MATLAB .mat files.

    Workflow:
      1. evaluator = CubeEvaluator(npy_root, mat_root, csv_path)
      2. evaluator.run_evaluation()       # compute metrics + write CSV
      3. evaluator.summarize_results()    # print global summary from CSV
    """

    def __init__(
        self, 
        npy_root: str,
        mat_root: str,
        csv_path: str = "comparison_results.csv",
        verbose: bool = False,
    ):
        self.npy_root = npy_root
        self.mat_root = mat_root
        self.csv_path = csv_path
        self.verbose = verbose
        

    # ---------- Public API ----------

    def run_evaluation(self) -> None:
        """
        - Scan both roots for .npy and .mat files
        - Pair them by common name
        - Compute metrics for each pair
        - Save full results CSV to self.csv_path
        - Print timing + progress + missing matches
        """
        npy_files = self._scan_dir_for_extension(self.npy_root, "npy")
        mat_files = self._scan_dir_for_extension(self.mat_root, "mat")

        if len(npy_files) == 0:
            print("[WARNING] :: no numpy files found. Have you used the correct path?")
            return

        pairs, only_npy, only_mat = self._match_pairs(npy_files, mat_files)

        if only_npy:
            self._log("\n[Info] Found .npy with NO matching .mat:")
            for name in only_npy:
                self._log(f" :: {name}")

        if only_mat:
            self._log("\n[Info] Found .mat with NO matching .npy:")
            for name in only_mat:
                self._log(f" :: {name}")

        all_results: list[dict[str, float]] = []

        num_pairs = len(pairs)
        t0 = time.time()

        for i, item in enumerate(pairs):
            name = item["name"]
            npy_path = item["npy"]
            mat_path = item["mat"]

            self._log(f"\n=== Processing [{i+1} / {num_pairs}] :: {name} ===")
            self._log(f"NPY: {npy_path}")
            self._log(f"MAT: {mat_path}")

            # safety check files exist
            if not os.path.exists(npy_path):
                self._log(f"WARNING: NPY not found, skipping {name}")
                continue
            if not os.path.exists(mat_path):
                self._log(f"WARNING: MAT not found, skipping {name}")
                continue

            # load data
            pred = self._load_npy(npy_path)        # (H, W, C)
            gt   = self._load_mat_cube(mat_path)   # (H, W, C)

            # sanity check ranges
            self._log(f"  pred  min/max: {pred.min()} {pred.max()}")
            self._log(f"  gt    min/max: {gt.min()} {gt.max()}")

            # compute metrics
            metrics = self._compute_pair_metrics(pred, gt)

            self._log(f"  mean_abs_err: {metrics['mean_abs_err']}")
            self._log(f"  max_abs_err : {metrics['max_abs_err']}")
            self._log(f"  psnr_1      : {metrics['psnr_1']}")
            self._log(f"  psnr_255    : {metrics['psnr_255']}")
            self._log(f"  rmse        : {metrics['rmse']}")

            row = {
                "name": name,
                "mean_abs_err": metrics["mean_abs_err"],
                "max_abs_err": metrics["max_abs_err"],
                "mse": metrics["mse"],
                "psnr_1": metrics["psnr_1"],
                "psnr_255": metrics["psnr_255"],
                "rmse": metrics["rmse"],
                **{f"band_{i:02d}_mae": metrics["per_band_mae"][i] for i in range(len(metrics["per_band_mae"]))},
                **{f"band_{i:02d}_rmse": metrics["per_band_rmse"][i] for i in range(len(metrics["per_band_rmse"]))}
            }

            all_results.append(row)

        t1 = time.time()

        # write CSV
        if all_results:
            # Collect header keys across all rows so DictWriter is stable
            fieldnames = sorted({k for row in all_results for k in row.keys()})

            with open(self.csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in all_results:
                    writer.writerow(row)

            self._log(f"\nWrote metrics for {len(all_results)} items to {self.csv_path}")
        else:
            self._log("\nNo results to write (no successful pairs).")

        self._log(f"Done in {(t1 - t0):.2f} seconds")

    def summarize_results(self) -> None:
        """
        Read the CSV at self.csv_path and print:
        - overall MAE, MSE, RMSE, etc (averaged over samples)
        - per-band average MAE / RMSE
        """
        df = pd.read_csv(self.csv_path)

        # Global metrics averaged across samples
        overall_mean_abs_err = df["mean_abs_err"].mean()
        overall_max_abs_err  = df["max_abs_err"].mean()
        overall_mse          = df["mse"].mean()
        overall_psnr_1       = df["psnr_1"].mean()
        overall_psnr_255     = df["psnr_255"].mean()
        overall_rmse         = df["rmse"].mean()

        self._log("=== OVERALL (averaged across all samples) ===")
        self._log(f"Mean Abs Error (MAE): {overall_mean_abs_err}")
        self._log(f"Max Abs Error:        {overall_max_abs_err}")
        self._log(f"MSE:                  {overall_mse}")
        self._log(f"PSNR_1:               {overall_psnr_1}")
        self._log(f"PSNR_255:             {overall_psnr_255}")
        self._log(f"RMSE:                 {overall_rmse}")

        # Per-band metrics
        band_mae_cols  = [c for c in df.columns if c.startswith("band_") and c.endswith("_mae")]
        band_rmse_cols = [c for c in df.columns if c.startswith("band_") and c.endswith("_rmse")]

        def band_index(colname: str) -> int:
            # "band_07_mae" -> 7
            return int(colname.split("_")[1])

        band_mae_cols  = sorted(band_mae_cols,  key=band_index)
        band_rmse_cols = sorted(band_rmse_cols, key=band_index)

        per_band_mae_mean  = df[band_mae_cols].mean(axis=0)
        per_band_rmse_mean = df[band_rmse_cols].mean(axis=0)

        self._log("\n=== PER-BAND AVERAGE METRICS (across all samples) ===")
        for mae_col, rmse_col in zip(band_mae_cols, band_rmse_cols):
            band_id = band_index(mae_col)
            mae_val = per_band_mae_mean[mae_col]
            rmse_val = per_band_rmse_mean[rmse_col]
            self._log(f"Band {band_id:02d}:  avg MAE={mae_val:.6f},  avg RMSE={rmse_val:.6f}")

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

    def _load_npy(self, npy_path: str) -> np.ndarray:
        """
        Load model prediction cube from .npy
        Expected shape: (H, W, C), dtype float32
        """
        arr = np.load(npy_path)
        return arr

    def _load_mat_cube(self, mat_path: str) -> np.ndarray:
        """
        Load ground truth cube from .mat
        Assumes dataset is under key "cube" with shape (C, H, W)
        Returns aligned float32 array of shape (H, W, C).
        """
        with h5py.File(mat_path, "r") as f:
            cube = f["cube"][:]  # (C, H, W), float64
        cube_hwk = np.transpose(cube, (1, 2, 0))  # -> (H, W, C)
        cube_hwk = cube_hwk.astype(np.float32)
        return cube_hwk

    def _compute_pair_metrics(self, pred: np.ndarray, gt: np.ndarray):
        """
        Compute error statistics between pred and gt.

        pred, gt: (H, W, C), float32
        Returns dict with:
          - mean_abs_err
          - max_abs_err
          - mse
          - rmse
          - per_band_mae  (list[float] length C)
          - per_band_rmse (list[float] length C)
        """
        if pred.shape != gt.shape:
            raise ValueError(f"Shape mismatch: pred {pred.shape} vs gt {gt.shape}")

        diff = pred - gt
        abs_diff = np.abs(diff)

        mean_abs_err = float(np.mean(abs_diff))
        max_abs_err  = float(np.max(abs_diff))
        mse          = float(np.mean(diff ** 2))
        rmse         = float(np.sqrt(mse))
        if mse < 1.0e-10:
            psnr_1   = 100
        else:
            psnr_1   = float(10.0 * np.log10((1 ** 2) / mse))

        if mse < 1.0e-10:
            psnr_255 = 100
        else:
            psnr_255 = float(10.0 * np.log10((255 ** 2) / mse))
        

        # Per-band stats
        per_band_mae  = np.mean(abs_diff, axis=(0, 1))            # (C,)
        per_band_rmse = np.sqrt(np.mean(diff**2, axis=(0, 1)))    # (C,)

        return {
            "mean_abs_err": mean_abs_err,
            "max_abs_err": max_abs_err,
            "mse": mse,
            "rmse": rmse,
            "psnr_1": psnr_1,
            "psnr_255": psnr_255,
            "per_band_mae":  [float(x) for x in per_band_mae],
            "per_band_rmse": [float(x) for x in per_band_rmse],
        }
        
    def _log(self, msg: str, force: bool = False):
        if self.verbose or force:
            print(msg, flush=True)
        
if __name__ == "__main__":
    evaluator = Validate(
        npy_root="./class_exp/mst_plus_plus/batch_76b3373e/",
        mat_root="C:/Users/tobia/Downloads/hyper-skin-data/Hyper-Skin(RGB, VIS)",
        csv_path="mst_pp_results.csv",
        verbose=True
    )

    # evaluator.run_evaluation()
    evaluator.summarize_results()