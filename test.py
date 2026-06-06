import time
import torch
from taehv.taehv import TAEHV

ckpt = '/workspace/owl-audio-gen/taehv/taehv1_5.pth'
taehv_cpu = TAEHV(ckpt).cpu().bfloat16().eval()
taehv_gpu = TAEHV(ckpt).to('cuda').bfloat16().eval()

dummy = torch.randn(1, 300, 3, 256, 256).bfloat16()

with torch.no_grad():
    # # CPU
    # t0 = time.time()
    # for _ in range(5):
    #     taehv_cpu.encode_video(dummy)
    # print(f"CPU: {(time.time()-t0)/5:.2f}s per sample")

    # GPU
    dummy_gpu = dummy.cuda()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(5):
        out = taehv_gpu.encode_video(dummy_gpu)
        print(out.shape)
    torch.cuda.synchronize()
    print(f"GPU: {(time.time()-t0)/5:.2f}s per sample")