import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from mstpp.model import MST_Plus_Plus
from data_carrier import DataCarrier
import os
import argparse
from pathlib import Path

device = "cuda" if torch.cuda.is_available() else "cpu"

class Opt:
    def __init__(self):
        self.stage = 3
        self.bands = 4
        self.size = 256

def run(root_dir, data_type, save_dir, single, single_picture, amount, modelpath, full_picture):
    opt = Opt()
    model = MST_Plus_Plus(in_channels=3, out_channels=4, n_feat=4, stage=3).to(device)
    ouput_dir= Path(save_dir)
    ouput_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(modelpath, map_location=device)
    state_dict = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt

    model_sd = model.state_dict()
    filtered = {}
    for k, v in state_dict.items():
        key = k
        if key.startswith("module."):
            key = key[len("module."):]
            
        if key in model_sd and v.size() == model_sd[key].size():
            filtered[key] = v

    missing = set(model_sd.keys()) - set(filtered.keys())
    extra = set(state_dict.keys()) - set(filtered.keys())

    print(f"Loading checkpoint: kept {len(filtered)} params, missing {len(missing)} params, skipped {len(extra)} params")

    model.load_state_dict(filtered, strict=False)
    model.eval()

    dataset = DataCarrier(root_dir=root_dir, single=single, single_picture=single_picture, full_or_patch=full_picture) #Expect data carrier to handle single or multiple pictures

    index = 0

    if single:
        limit = 1
    elif amount == "Full":
        limit = None
    else:
        limit = int(amount)

    for sample in dataset[0:limit]:
        rgb = sample["rgb"] 
        target = sample["ms"]
    
        rgb_vis = rgb.unsqueeze(0).permute(0, 2, 3, 1).squeeze(0).cpu().numpy()
        rgb = rgb.unsqueeze(0).to(device)             

        with torch.no_grad():
            output = model(rgb)
            if isinstance(output, list): 
                output = output[-1]
            pred = output.squeeze(0).cpu().numpy()

        pred = np.clip(pred, 0, 1)
        target = target.numpy()

        fig, axes = plt.subplots(2, 5, figsize=(14, 5))
        axes[0, 0].imshow(rgb_vis)
        axes[0, 0].set_title("RGB Input")
        axes[0, 0].axis("off")

        for i in range(4):
            axes[0, i+1].imshow(target[i], cmap='gray')
            axes[0, i+1].set_title(f"GT Band {i+1}")
            axes[0, i+1].axis("off")

        for i in range(4):
            axes[1, i].imshow(pred[i], cmap='gray')
            axes[1, i].set_title(f"Pred Band {i+1}")
            axes[1, i].axis("off")

        axes[1, 4].axis("off")
        plt.tight_layout()
        file_name = "validation_result"+index+".png"
        plt.savefig("validation_result.png", dpi=150, bbox_inches="tight")
        plt.savefig(ouput_dir / file_name, dpi=150, bbox_inces="tight")
        plt.close()
        print(f"Saved visualization to {file_name}")
        index += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Creates patches from spectral bands.")
    parser.add_argument("--data_path", help="Path to directory with data, default=data/", default="data/")
    parser.add_argument("--single", type=bool, help="One or many pictures, default=many", default=False)
    parser.add_argument("--jpg", help="path to single picture, only applies if --single=True", default=None)
    parser.add_argument("--full_picture", type=bool, help="Use full pictures or patches, default=False/Patches", default=False)
    parser.add_argument("--amount", help="Amount of pictures the eval should run through, only applies if single=False, default=Full/entire dataset", default="Full")
    parser.add_argument("--save_path", help="Name of save directory", default="default")
    parser.add_argument("--data_type", help="Which dataset Sri-Lanka or Kazakhstan, default=Sri-Lanka", default="Sri-Lanka")
    parser.add_argument("--model", help="Which model to use, and path to the model from project dir, default=model_final.pkl", default="model_final.pkl")
    args = parser.parse_args()
    root_dir = args.data_path # Root directory of data (data/)
    data_type = args.data_type #Dataset type (Sri-Lanka or Kazakhstan)
    save_dir = args.save_path #Save path for results (also saves latest result in validation_result.png in main folder)
    single = args.single # One or many pictures
    single_picture = args.jpg #Only one picture
    amount = args.amount #If not single, gives the amount of pictures to process
    modelpath = args.model #MST++ model to evaluate
    full_picture = args.full_picture #Patches or full picture
    run(root_dir, data_type, save_dir, single, single_picture, amount, modelpath, full_picture)
    
