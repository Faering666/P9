import glob
import os
import time
import numpy as np
import h5py
import csv
import pandas as pd

NPY_PATH = "./class_exp/mst_plus_plus/batch_de7894aa/"
MAT_PATH = "C:/Users/tobia/Downloads/hyper-skin-data/Hyper-Skin(RGB, VIS)"
CSV_PATH = "comparison_results.csv"

def _get_files(dir_path: str, file_extension: str) -> list[dict[str, str]]:
    results: list[dict[str, str]] = []

    pattern = os.path.join(dir_path, "**", f"*.{file_extension}")
    for npy_path in glob.glob(pattern, recursive=True):
        if os.path.isdir(npy_path):
            continue
        
        base = os.path.basename(npy_path)
        name = os.path.splitext(base)[0]
        
        results.append({
            "name": name,
            file_extension: os.path.abspath(npy_path),
        })
            
    return results

def _join_npy_mat(npy_list: list[dict[str, str]], mat_list: list[dict[str, str]]) -> list[dict[str, str]]:
    # Index both lists by name for fast lookup
    npy_by_name = {item["name"]: item["npy"] for item in npy_list}
    mat_by_name = {item["name"]: item["mat"] for item in mat_list}

    # Names present in each side
    npy_names = set(npy_by_name.keys())
    mat_names = set(mat_by_name.keys())

    # Intersection = we can build pairs for these
    common_names = npy_names & mat_names

    # Only-on-one-side = we'll report these
    only_npy = npy_names - mat_names
    only_mat = mat_names - npy_names

    # Build the merged pairs list
    pairs = []
    for name in sorted(common_names):
        pairs.append({
            "name": name,
            "npy": npy_by_name[name],
            "mat": mat_by_name[name],
        })

    if len(only_npy) > 0:
        print(only_npy)
    
    if len(only_mat) > 0:
        print(only_mat)

    return pairs

def _load_npy(npy_path: str) -> np.ndarray:
    arr = np.load(npy_path)  # expect (H, W, 31), float32
    return arr

def _load_mat_cube(mat_path: str) -> np.ndarray:
    with h5py.File(mat_path, "r") as f:
        # assume variable is always called "cube"
        cube = f["cube"][:]  # shape (31, H, W), float64
    # reorder axes to (H, W, 31)
    cube_hwk = np.transpose(cube, (1, 2, 0))
    # match dtype to float32
    cube_hwk = cube_hwk.astype(np.float32)
    return cube_hwk

def _compute_metrics(model_pred: np.ndarray, expected: np.ndarray):
    """
    model_pred: (H, W, C) float32
    expected  : (H, W, C) float32
    returns a dict of summary metrics
    """
    if model_pred.shape != expected.shape:
        raise ValueError(f"Shape mismatch: pred {model_pred.shape} vs exp {expected.shape}")

    diff = model_pred - expected
    abs_diff = np.abs(diff)

    mean_abs_err = float(np.mean(abs_diff))
    max_abs_err  = float(np.max(abs_diff))

    mse  = float(np.mean(diff ** 2))
    rmse = float(np.sqrt(mse))

    # Per-band metrics (length C)
    per_band_mae = np.mean(abs_diff, axis=(0, 1))             # (C,)
    per_band_rmse = np.sqrt(np.mean(diff**2, axis=(0, 1)))    # (C,)

    # Cast those to regular Python floats for CSV friendliness
    per_band_mae_list = [float(x) for x in per_band_mae]
    per_band_rmse_list = [float(x) for x in per_band_rmse]

    return {
        "mean_abs_err": mean_abs_err,
        "max_abs_err": max_abs_err,
        "mse": mse,
        "rmse": rmse,
        "per_band_mae": per_band_mae_list,
        "per_band_rmse": per_band_rmse_list,
    }


def validate_results(npy_path: str, mat_path: str, csv_path: str):
    npy_files = _get_files(npy_path, "npy")
    mat_files = _get_files(mat_path, "mat")

    pairs = _join_npy_mat(npy_files, mat_files)
    
    all_results = []

    num_pairs = len(pairs)
    t0 = time.time()
    for i, item in enumerate(pairs):
        name = item["name"]
        npy_path = item["npy"]
        mat_path = item["mat"]
        
        print(f"\n=== Processing [{i+1} / {num_pairs}] :: {name} ===")
        print(f"NPY: {npy_path}")
        print(f"MAT: {mat_path}")
        
        # safety check files exist
        if not os.path.exists(npy_path):
            print(f"WARNING: NPY not found, skipping {name}")
            continue
        if not os.path.exists(mat_path):
            print(f"WARNING: MAT not found, skipping {name}")
            continue
        
        # load
        pred = _load_npy(npy_path)
        gt   = _load_mat_cube(mat_path)

        # (Optional) sanity check ranges
        print("  pred  min/max:", pred.min(), pred.max())
        print("  gt    min/max:", gt.min(), gt.max())

        # compute metrics
        metrics = _compute_metrics(pred, gt)

        print("  mean_abs_err:", metrics["mean_abs_err"])
        print("  max_abs_err :", metrics["max_abs_err"])
        print("  rmse        :", metrics["rmse"])

        # store summary row for CSV / table
        all_results.append({
            "name": name,
            "mean_abs_err": metrics["mean_abs_err"],
            "max_abs_err": metrics["max_abs_err"],
            "mse": metrics["mse"],
            "rmse": metrics["rmse"],
            # per-band gets stored separately below
            **{f"band_{i:02d}_mae": metrics["per_band_mae"][i] for i in range(len(metrics["per_band_mae"]))},
            **{f"band_{i:02d}_rmse": metrics["per_band_rmse"][i] for i in range(len(metrics["per_band_rmse"]))}
        })
    
    t1 = time.time()
    
    if all_results:
        # Grab all unique keys for header
        fieldnames = sorted({k for row in all_results for k in row.keys()})

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in all_results:
                writer.writerow(row)

        print(f"\nWrote metrics for {len(all_results)} items to {csv_path}")
    else:
        print("\nNo results to write (no successful pairs).")

    print(f"Done in {(t1 - t0):.2f}")

def global_data():
    df = pd.read_csv(CSV_PATH)

    overall_mean_abs_err = df["mean_abs_err"].mean()
    overall_max_abs_err  = df["max_abs_err"].mean()
    overall_mse          = df["mse"].mean()
    overall_rmse         = df["rmse"].mean()

    print("=== OVERALL (averaged across all samples) ===")
    print(f"Mean Abs Error (MAE): {overall_mean_abs_err}")
    print(f"Max Abs Error:        {overall_max_abs_err}")
    print(f"MSE:                  {overall_mse}")
    print(f"RMSE:                 {overall_rmse}")

    band_mae_cols  = [c for c in df.columns if c.endswith("_mae") and c.startswith("band_")]
    band_rmse_cols = [c for c in df.columns if c.endswith("_rmse") and c.startswith("band_")]

    def band_index(colname):
        # colname like "band_07_mae" -> 7
        return int(colname.split("_")[1])

    band_mae_cols  = sorted(band_mae_cols,  key=band_index)
    band_rmse_cols = sorted(band_rmse_cols, key=band_index)

    per_band_mae_mean  = df[band_mae_cols].mean(axis=0)
    per_band_rmse_mean = df[band_rmse_cols].mean(axis=0)

    print("\n=== PER-BAND AVERAGE METRICS (across all samples) ===")
    for mae_col, rmse_col in zip(band_mae_cols, band_rmse_cols):
        band_id = band_index(mae_col)
        mae_val = per_band_mae_mean[mae_col]
        rmse_val = per_band_rmse_mean[rmse_col]
        print(f"Band {band_id:02d}:  avg MAE={mae_val:.6f},  avg RMSE={rmse_val:.6f}")
