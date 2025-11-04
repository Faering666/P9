from model_runner.model_runner import ModelRunner


mst_runner = ModelRunner(
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

ssr_runner = ModelRunner(
    method="ssr_s",
    model_path="./model_zoo/model_S.pkl",
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
print("############## MST ##############")
out = mst_runner.infer_one("./imgs/picture.jpg", batch_id="mst", return_arrays=False)
print("############## SSR ##############")
out = ssr_runner.infer_one("./imgs/picture.jpg", batch_id="ssr", return_arrays=False)
# print(out["stats"])

# outs = runner.infer_many(["./imgs/picture.jpg", "./imgs/picture2.jpg"])

# multiple images (via glob)
# outs = runner.infer_many("./imgs/**/*.jpg", glob_recursive=True, limit=50)
