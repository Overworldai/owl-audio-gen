import torch
from torch import nn

class AudioRFTCore(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def forward(self, x, ts):
        return x # TODO

class AudioRFT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.core = AudioRFTCore(cfg)

    def forward(self, x):
        with torch.no_grad():
            ts = torch.randn(len(x), device = x.device, dtype = x.dtype).sigmoid() # logit normal
            ts_exp = ts[:,None,None]

            z = torch.randn_like(x)
            noisy = x * (1 - ts_exp) + ts_exp * z

        pred = self.core(noisy, ts)
        loss = F.mse_loss(pred, z)
        return loss