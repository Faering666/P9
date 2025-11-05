from types import SimpleNamespace
from typing import Any
import torch
from torch import nn

from .mstpp import MST_Plus_Plus as MST_PP
from .ssr import Net as SSR
from pathlib import Path

def load_mst_pp(model_path: str, device: torch.device) -> nn.Module:
    model = MST_PP().cuda()
    checkpoint = torch.load(model_path)
    model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()},
                              strict=True)
    model.eval()
    return model

def load_ssr(model_path: str, device: torch.device) -> nn.Module:
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    model_size = Path(model_path).stem[-1].lower()
    if model_size == 's':
         stage = 3
    elif model_size == 'm':
        stage = 6
    elif model_size == 'l':
        stage = 9
    else:
        raise ValueError("The modelname's tail must be 's', 'm' or 'l' to specify which model size to use.")

    opt = SimpleNamespace(stage=stage, bands=28, size=256)
    model = SSR(opt).cuda()
    model.load_state_dict(checkpoint['model'])
        
    model.eval()
    return model

def forward_ssr(model: nn.Module, rgb: torch.Tensor) -> list[Any]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if not isinstance(model, SSR):
        raise TypeError(f"{type(model).__name__} doesn't implement forward_with_f0()")

    nC = getattr(model, "nC", None)
    if nC is None:
        raise AttributeError("Model must define `nC` (number of bands/channels).")

    rgb_adapter = nn.Sequential(
        nn.Conv2d(3, nC, kernel_size=1, stride=1, padding=0, bias=True),
        nn.ReLU(inplace=True)
    ).to(device)
    
    rgb = rgb.to(device)
    f0 = rgb_adapter(rgb)
    assert f0.size(1) == nC, f"Adapter out channels {f0.size(1)} != model.nC {nC}"
    
    dummy_mask = _create_dummy_mask(
        batch_size=rgb.size(0),
        bands=nC,
        H=rgb.size(2),
        W=rgb.size(3),
        device=device
    )
    
    out = model.forward_with_f0(f0, input_mask=dummy_mask)
    print("out size: ", len(out))
        
    return out[2]

def _create_dummy_mask(batch_size, bands, H, W, device):
    Phi = torch.ones(batch_size, bands, H, W, device=device) 
    PhiPhiT = torch.ones(batch_size, 1, H, W, device=device)
    return (Phi, PhiPhiT)
