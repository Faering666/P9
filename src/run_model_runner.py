from mstpp import load_modified_mst_pp
from model_runner import ModelRunner
import argparse


parser = argparse.ArgumentParser(description="Model runner.")
parser.add_argument("--model_path", type=str, help="Path to the model to run.", required=True)
parser.add_argument("--input_path", type=str, help="Path to the directory with images (takes glob) or a file (if '--single_inference' flag is used)", required=True)
parser.add_argument("--out_path", type=str, help="Path to the output dir", required=True)
parser.add_argument("--batch_label", type=str, help="Batch label of run - '{batch-label}_{batch_id}", default=None)
parser.add_argument("--batch_id", type=str, help="Batch id of run - '{batch-label}_{batch_id}", default=None)
parser.add_argument("--single_inference", type=bool, help="Run inference of a single image", action=argparse.BooleanOptionalAction)
parser.add_argument("--guid_len", type=str, help="Path to the output dir", default=4)
parser.add_argument("--verbose", type=bool, help="Print more information to the terminal", action=argparse.BooleanOptionalAction)

# Which types to save
parser.add_argument("--npy", type=bool, help="Save the result as numpy file", action=argparse.BooleanOptionalAction)
parser.add_argument("--mat", type=bool, help="Save the result as matlab file", action=argparse.BooleanOptionalAction)
parser.add_argument("--gray_grid", type=bool, help="Save the result as a grid of gray images", action=argparse.BooleanOptionalAction)
parser.add_argument("--color_grid", type=bool, help="Save the result as a grid of color images", action=argparse.BooleanOptionalAction)

args = parser.parse_args()

mst_runner = ModelRunner(
    method="mst_plus_plus",
    model_path=args.model_path,
    load_model=load_modified_mst_pp,
    save_dir=args.out_path,
    save_mat=args.mat,
    save_npy=args.npy,
    save_gray_grid=args.gray_grid,
    save_color_grid=args.color_grid,
    verbose=args.verbose,
    guid_length=args.guid_len,
    
)

if args.single_inference:
    mst_runner.infer_one(args.input_path, batch_label=args.batch_label, batch_id=args.batch_id, return_arrays=False)
else:
    mst_runner.infer_many(args.input_path, batch_label=args.batch_label, batch_id=args.batch_id, glob_recursive=True, no_prefix=True)