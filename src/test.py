from baseline_models import forward_ssr, load_mst_pp, load_ssr
from model_runner import ModelRunner

mst_runner = ModelRunner(
    method="mst_plus_plus",
    model_path="./baseline_models/mst_plus_plus.pth",
    load_model=load_mst_pp,
    save_dir="./class_exp/",
    # save_mat=True,
    # save_npy=True,
    save_gray_grid=True,
    # save_color_grid=True,
    # save_each_band=True,
    verbose=True,
    guid_length=4,
)

print("############## MST ##############")
out = mst_runner.infer_one("./imgs/picture.jpg", batch_id="mst", return_arrays=False)

ssr_runner = ModelRunner(
    method="ssr_s",
    model_path="./baseline_models/model_S.pkl",
    load_model=load_ssr,
    forward_model=forward_ssr,
    save_dir="./class_exp/",
    # save_mat=True,
    # save_npy=True,
    save_gray_grid=True,
    # save_color_grid=True,
    # save_each_band=True,
    verbose=True,
    guid_length=4,
)


print("############## SSR-S ##############")
out = ssr_runner.infer_one("./imgs/picture.jpg", prefix="third", batch_id="ssr_s", return_arrays=False)