from lib.mstpp.model_runner import ModelRunner

runner = ModelRunner(
    method="mst_plus_plus",
    model_path="./model_zoo/mst_plus_plus.pth",
    save_dir="./class_exp/",
    # save_mat=True,
    # save_npy=True,
    # save_gray_grid=True,
    save_color_grid=True,
    # save_each_band=True,
    verbose=True,
    guid_length=4,
)

# single image
out = runner.infer_one("./imgs/picture.jpg", return_arrays=False)
# print(out["stats"])

# outs = runner.infer_many(["./imgs/picture.jpg", "./imgs/picture2.jpg"])

# multiple images (via glob)
# outs = runner.infer_many("./imgs/**/*.jpg", glob_recursive=True, limit=50)
