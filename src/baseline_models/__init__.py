# SPDX-License-Identifier: MIT
# Adapted from MST-plus-plus (caiyuanhao1998/MST-plus-plus), licensed under the MIT License.
# Copyright (c) <Yuanhao Cai>.
# Modifications Copyright (c) 2025 <Hugin J. Zachariasen, Magnus H. Jensen, Martin C. B. Nielsen, Tobias S. Madsen>.

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