import torch
import torch.nn as nn
import torch.nn.functional as F

from ssr.model import Net as SSRNet

class SSRNetRGBTransfer(nn.Module):
    """
    Wrapper that adapts RGB input to the original SSR Net.
    Create an rgb_adapter that maps 3-channels RGB -> nC channels (f0),
    then pass f0 into the original pipeline similarly to how temp_g/f0 was used.
    This approach preserves the internal architecture of SSRNet.
    """
    def __init__(self, opt, load_pretrained_checkpoint=None, device='cpu'):
        super().__init__()
        self.opt = opt
        self.device = device
        self.ssr = SSRNet(opt)

        # RGB to nC adapter (3 -> 4 channels)
        self.rgb_adapter = nn.Sequential(
            nn.Conv2d(3, opt.bands, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(inplace=True)
        )

        if load_pretrained_checkpoint is not None:
            self._load_pretrained(load_pretrained_checkpoint)

    def _load_pretrained(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        # pretrained_dict = checkpoint.get("model_state_dict", checkpoint)
        if 'model' in checkpoint:
            pretrained_dict = checkpoint['model']
        elif 'model_state_dict' in checkpoint:
            pretrained_dict = checkpoint['model_state_dict']
        else:
            pretrained_dict = checkpoint

        ssr_state = self.ssr.state_dict()
        filtered = {}
        skipped = []

        for k, v in pretrained_dict.items():
            key = k
            if key.startswith("module."):
                key = key[len("module."):]
            
            if key in ssr_state and ssr_state[key].shape == v.shape:
                filtered[key] = v
            else:
                skipped.append(key)
        
        # Update and load
        ssr_state.update(filtered)
        self.ssr.load_state_dict(ssr_state)

        print(f"[Pretrained loading] Loaded {len(filtered)} params, skipped {len(skipped)} params (incompatible shapes).")
        if skipped:
            print("Skipped keys:", skipped[:10], "..." if len(skipped) > 10 else "")

    def forward(self, rgb, input_mask=None):
        f0 = self.rgb_adapter(rgb) # -> (B, 4, H, W)
        return self.ssr.forward_with_f0(f0, input_mask=input_mask)