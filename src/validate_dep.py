# SPDX-License-Identifier: MIT
# Copyright (c) 2025 <Hugin J. Zachariasen, Magnus H. Jensen, Martin C. B. Nielsen, Tobias S. Madsen>.

import numpy as np
import os
import json
import glob
import h5py

from metric_calculator import MetricCalculator  # for reading .mat files saved like in your dataset script

class Evaluator:
    def __init__(
        self,
        result_path: str,
        correct_path: str,
        metric_calculator: MetricCalculator,
        gt_key: str = "cube",
        pred_ext: str = "npy",
        gt_ext: str = "mat",
    ):
        self.result_path = result_path
        self.correct_path = correct_path
        self.gt_key = gt_key
        self.pred_ext = pred_ext
        self.gt_ext = gt_ext
        self.metrics = metric_calculator

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

            metrics_dict = self.metrics.compute(pred, gt)

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

    evaluator = Evaluator(
        result_path=result_dir,
        correct_path=gt_dir,
        gt_key="cube",       # change if your .mat files use a different variable name
        pred_ext="npy",
        gt_ext="mat",
    )

    results_file = "./new-results.json"
    scores = evaluator.evaluate(results_file)
    for name, value in scores.items():
        print(f"{name}: {value:.6f}")
    
    # print("=========================")
    # scores = evaluator.evaluate_from_file(results_file)
    # print("MRAE:", scores["MRAE"])
    # print("RMSE:", scores["RMSE"])
    # print("MSE: ", scores["MSE"])
    # print("PSNR:", scores["PSNR"])
    # print("NDVI_PRED:", scores["NDVI_PRED"])
    # print("NDVI_GT:", scores["NDVI_GT"])
    # print("NDVI_RMSE:", scores["NDVI_RMSE"])