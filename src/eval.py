import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
# from ssr.rgb_transfer import SSRNetRGBTransfer 
from mstpp.model import MST_Plus_Plus
from data_carrier import DataCarrier
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

class Opt:
    def __init__(self):
        self.stage = 3
        self.bands = 4
        self.size = 256

opt = Opt()
# model = SSRNetRGBTransfer(opt, device=device).to(device)
model = MST_Plus_Plus(in_channels=3, out_channels=4, n_feat=4, stage=3).to(device)

ckpt = torch.load("model_final.pkl", map_location=device)
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

def create_dummy_mask(batch_size, bands, H, W, device):
    Phi = torch.ones(batch_size, bands, H, W, device=device) 
    PhiPhiT = torch.ones(batch_size, 1, H, W, device=device)
    return (Phi, PhiPhiT)

dataset = DataCarrier(root_dir="data/Potato/train/img/", size=opt.size)

sample = dataset[67]
rgb = sample["rgb"] 
target = sample["ms"]

rgb_vis = rgb.unsqueeze(0).permute(0, 2, 3, 1).squeeze(0).cpu().numpy()
rgb = rgb.unsqueeze(0).to(device)             
dummy_mask = create_dummy_mask(rgb.size(0), opt.bands, rgb.size(2), rgb.size(3), device)

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
out_path = "validation_result.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved visualization to {out_path}")
