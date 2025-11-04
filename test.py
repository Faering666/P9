import pickle

import numpy as np
import torch
from lib.model_runner import ModelRunner


# print("############## MST ##############")
# mst_runner = ModelRunner(
#     method="mst_plus_plus",
#     model_path="./lib/model_runner/model_zoo/mst_plus_plus.pth",
#     save_dir="./class_exp/",
#     # save_mat=True,
#     # save_npy=True,
#     # save_gray_grid=True,
#     save_color_grid=True,
#     # save_each_band=True,
#     verbose=True,
#     guid_length=4,
# )
# single image
# out = mst_runner.infer_one("./lib/model_runner/imgs/picture.jpg", batch_id="mst", return_arrays=False)
# print(out["stats"])

# outs = runner.infer_many(["./lib/model_runner/imgs/picture.jpg", "./imgs/picture2.jpg"])

# multiple images (via glob)
# outs = runner.infer_many("./imgs/**/*.jpg", glob_recursive=True, limit=50)



obj = torch.load(
    "./lib/model_runner/model_zoo/model_S.pkl",
    map_location="cuda",
    weights_only=False
)

print("############## SSR ##############")
ssr_runner = ModelRunner(
    method="ssr_s",
    model_path="./lib/model_runner/model_zoo/model_S.pkl",
    save_dir="./class_exp/",
    # save_mat=True,
    # save_npy=True,
    # save_gray_grid=True,
    save_color_grid=True,
    # save_each_band=True,
    verbose=True,
    guid_length=4,
)
out = ssr_runner.infer_one("./lib/model_runner/imgs/picture.jpg", batch_id="ssr", return_arrays=False)

