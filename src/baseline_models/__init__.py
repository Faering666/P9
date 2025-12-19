import torch
from torch import nn

from .mstpp import MST_Plus_Plus as MST_PP

def load_stock_mst_pp(model_path: str, device: torch.device) -> nn.Module:
    model = MST_PP().cuda()
    checkpoint = torch.load(model_path)
    model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()},
                              strict=True)
    model.eval()
    return model