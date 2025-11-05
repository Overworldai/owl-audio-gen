from .schedulers import get_sd3_euler
import torch
from tqdm import tqdm

@torch.no_grad()
def flow_sample(model, dummy, z, steps, decoder=None, scaling_factor = 1.0, progress_bar = True):
    x = torch.randn_like(dummy)
    ts = torch.ones(len(z), device = z.device, dtype = z.dtype)

    if steps > 1:
        dt = get_sd3_euler(steps).to(z.device)
        for i in tqdm(range(steps), disable = not progress_bar):
            x = x - dt[i] * model(x, z, ts)
            ts = ts - dt[i]
    else:
        x = x - model(x, z, ts)

    if decoder is not None:
        x = x.bfloat16() * scaling_factor
        x = decoder(x)
    
    return x