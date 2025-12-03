from baseline_models import load_mst_pp
from model_runner import ModelRunner

print("############## MST ##############")
mst_runner = ModelRunner(
    method="mst_plus_plus",
    model_path="./baseline_models/mst_plus_plus.pth",
    load_model=load_mst_pp,
    save_dir="./class_exp/",
    save_mat=True,
    save_npy=True,
    # save_gray_grid=True,
    # save_color_grid=True,
    # save_each_band=True,
    # verbose=True,
    guid_length=4,
)

# out = mst_runner.infer_one("./imgs/picture.jpg", batch_id="mst", return_arrays=False)
# mst_runner.infer_many("C:/Users/tobia/Downloads/hyper-skin-data/Hyper-Skin(RGB, VIS)/**/*.jpg", glob_recursive=True, no_prefix=True)
# mst_runner.infer_many("C:/Users/tobia/Downloads/ARAD_1K_Mirror/Valid_RGB/**/*.jpg", batch_label="mst", batch_id="valid", glob_recursive=True, no_prefix=True)
mst_runner.infer_many("C:/Users/tobia/Downloads/ARAD_1K_Mirror/Train_RGB/**/*.jpg", batch_label="mst", batch_id="train", glob_recursive=True, no_prefix=True)