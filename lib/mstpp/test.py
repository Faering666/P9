from mstpp_runner import MSTPPRunner

runner = MSTPPRunner(
    method="mst_plus_plus",
    model_path="./model_zoo/mst_plus_plus.pth",
    save_dir="./class_exp/",
    # save_mat=True,
    # save_npy=True,
    # save_gray_grid=True,
    save_color_grid=True,
    save_each_band=True,
    verbose=False,
    guid_length=4,
)

# single image
# out = runner.infer_one("./imgs/picture.jpg", return_arrays=False)
# print(out["stats"])

# multiple images (via list)
# outs = runner.infer_many(["./imgs/picture.jpg", "./imgs/picture2.jpg"], save_subdir="batch1")

# multiple images (via glob)
outs = runner.infer_many("./imgs/**/*.jpg", save_subdir="batch2", glob_recursive=True, limit=50)
