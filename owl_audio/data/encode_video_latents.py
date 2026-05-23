import os, glob, argparse
from pathlib import Path
import numpy as np

from tqdm import tqdm
import threading
import queue
import torch.multiprocessing as mp
import torch
import av

from taehv.taehv import TAEHV
from owl_audio.configs import Config


def _find_mp4s(source):
    s = os.path.expanduser(str(source))
    p = Path(s)
    registry = p / "valid_mp4s.txt"
    if p.is_dir() and registry.exists():
        print(f"Using registry {registry}")
        paths = [line.strip() for line in registry.read_text().splitlines() if line.strip()]
    elif p.is_dir():
        paths = glob.glob(str(p / "**/*.mp4"), recursive=True)
    else:
        paths = glob.glob(s, recursive=True)
    filtered = [x for x in paths if x.lower().endswith(".mp4") and '_' not in Path(x).stem]
    return [Path(x) for x in sorted(set(filtered))]

def _iter_video_chunks(mp4_path, decode_chunk_size, resize, target_fps=24):
    """Only decodes chunks belonging to this worker."""
    container = av.open(str(mp4_path))

    video_stream = container.streams.video[0]
    video_stream.thread_type = "AUTO"
    video_stream.thread_count = os.cpu_count() // torch.cuda.device_count()
    
    frames = []

    orig_fps = float(video_stream.average_rate)
    if target_fps is None:
        target_fps = orig_fps
        
    frame_interval = 1.0 / target_fps
    next_pts_time = 0.0
    
    try:
        for frame in container.decode(video_stream):
            # frame timestamp in seconds
            t = float(frame.pts * video_stream.time_base)
            # skip frames until next target time
            if t + 1e-6 < next_pts_time:
                continue
            next_pts_time += frame_interval 

            # Convert to numpy array
            if resize is not None:
                frame = frame.reformat(width=resize[0], height=resize[1])
            frame_array = frame.to_ndarray(format="rgb24")
            frames.append(frame_array)
            
            if len(frames) >= decode_chunk_size:
                yield np.stack(frames, axis=0)
                frames = []
        if frames:
            yield np.stack(frames, axis=0)
    finally:
        container.close()

@torch.no_grad()
def encode_video_chunks(
     video_path, 
     vae_encode_fn, 
     chunk_size, 
     decode_chunk_size,
     device, 
     resize, 
     target_fps,
     batch_size=2,
     prefetch=16
    ):
        """Load a video in chunks and encode each chunk with a VAE.
        Yields per-chunk latents on CPU to avoid accumulating full video in memory.
        """
        assert decode_chunk_size % chunk_size == 0, \
            f"decode_chunk_size={decode_chunk_size} must be a multiple of chunk_size={chunk_size}"

        # prefetch chunks in the background
        q = queue.Queue(maxsize=prefetch)

        def _producer():
            for chunk in _iter_video_chunks(video_path, decode_chunk_size, resize, target_fps):
                q.put(chunk)
            q.put(None)

        def _process_buffer(buffer):
            batch = torch.stack(buffer, dim=0)  # (B, T, C, H, W)
            encoded = vae_encode_fn(batch)
            latent = encoded.latent_dist.mode() if hasattr(encoded, "latent_dist") else encoded
            result = latent.cpu()
            del encoded, latent, batch
            return result
        
        def _pad_video(video):
            T = video.shape[0]
            if T < chunk_size:
                pad = video[-1].expand(chunk_size-T, -1, -1, -1)
                video = torch.cat([video, pad], dim=0)
            return video 

        def _flush_buffer():
            nonlocal buffer
            latents = _process_buffer(buffer)
            for i in range(latents.shape[0]):
                yield latents[i]
            buffer.clear()
        
        t = threading.Thread(target=_producer, daemon=True)
        t.start()

        buffer = []
        while True:
            chunk = q.get()
            if chunk is None:
                break
            
            video = torch.from_numpy(chunk).pin_memory().to(device, dtype=torch.bfloat16, non_blocking=True)
            video = video.permute(0, 3, 1, 2).contiguous() / 255.0

            T = video.shape[0]
            num_chunks = T // chunk_size
            for i in range(num_chunks):
                buffer.append(video[i*chunk_size:(i+1)*chunk_size])
                if len(buffer) >= batch_size:
                    yield from _flush_buffer()

            # end of video chunk can be shorter
            remainder = T % chunk_size
            if remainder > 0:
                buffer.append(_pad_video(video[num_chunks*chunk_size:]))

        # flush remaining 
        if buffer:
            yield from _flush_buffer()

def _find_missing_latents(source, encoded, vae_name):
    source = Path(source)
    encoded = Path(encoded)
    
    mp4s = _find_mp4s(source)
    missing = []
    for mp4_path in mp4s:
        folder = mp4_path.parent.name
        video_name = mp4_path.stem
        latent_dir = encoded / folder / vae_name / f"{video_name}_latent"
        if not latent_dir.exists() or not any(latent_dir.glob("*.pt")):
            missing.append((mp4_path, latent_dir))  
    return missing   

def encode_missing_videos(
    source, 
    encoded, 
    vae_encode_fn,
    resize,
    vae_name="taehv1_5", 
    chunk_size=240,
    decode_chunk_size=240,
    target_fps=30,
    device="cuda", 
    rank=0,
    world_size=1,
    dtype="int8"
):
    missing = _find_missing_latents(source, encoded, vae_name)
    missing = missing[rank::world_size] # assign each worker its own slice
    
    if len(missing) == 0:
        if rank == 0:
            print("All videos already encoded")
        return
            
    print(f"[GPU {rank}] Encoding {len(missing)} missing videos")

    for mp4_path, latent_path in missing:
        try:
            chunk_dir = latent_path.with_suffix("")
            chunk_dir.mkdir(parents=True, exist_ok=True)

            with tqdm(
                desc=f"[GPU {rank}] {mp4_path.name}", 
                unit="chunks", 
                position=rank,
                leave=True
            ) as pbar:
                 for chunk_idx, latent in enumerate(
                      encode_video_chunks(
                           mp4_path,
                           vae_encode_fn=vae_encode_fn,
                           chunk_size=chunk_size,
                           decode_chunk_size=decode_chunk_size,
                           device=device,
                           resize=resize,
                           target_fps=target_fps,
                      )
                 ):
                    # save latent chunks directly
                    chunk_path = chunk_dir / f"{chunk_idx:06d}.pt"
                    if pbar is not None:
                        pbar.update(1)
                    
                    lat_q, scale = convert_latent_dtype(latent, dtype)
                    payload = {'lat_q': lat_q, 'scale': scale, 'dtype': dtype} if scale else lat_q
                    torch.save(payload, chunk_path)
        except Exception as e:
            print(f"[GPU {rank}] Encoding failed for {mp4_path.name}: {e}", flush=True)

            if chunk_dir.exists():
                for p in chunk_dir.glob("*.pt"):
                    p.unlink()
            continue

        print(f"[GPU {rank}] Done: {mp4_path.name}", flush=True)
        torch.cuda.empty_cache()

def _worker(rank, world_size, cfg, dtype="int8"):
    # load vae encode fn
    device = f"cuda:{rank}"
    _taehv = TAEHV(cfg.train.video_vae_ckpt).to(device).bfloat16().eval()
    video_encode_fn = lambda x: _taehv.encode_video(x, show_progress_bar=False)
     
    encode_missing_videos(
        source=cfg.train.data_kwargs.source,
        encoded=cfg.train.data_kwargs.encoded,
        vae_encode_fn=video_encode_fn,
        vae_name="taehv1_5",
        resize=cfg.train.video_size,
        chunk_size=cfg.train.data_kwargs.video_window_frames,
        decode_chunk_size=cfg.train.data_kwargs.video_window_frames*2,
        target_fps=cfg.train.video_fps,
        device=device,
        rank=rank,
        world_size=world_size,
        dtype=dtype
    )

def convert_latent_dtype(latent, save_dtype="bf16"):
    _DTYPE_MAP = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }

    if save_dtype in _DTYPE_MAP:
        return latent.to(_DTYPE_MAP[save_dtype]), None

    if save_dtype == "int8":
        latent = latent.float()
        scale = torch.amax(latent.abs()).clamp_(min=1e-8)
        latent_q = (latent * (127.0 / scale)).round_().clamp_(-127, 127).to(torch.int8)
        return latent_q, scale.cpu()

    raise ValueError(f"Unknown save_dtype={save_dtype}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--config_path", type=str, help="Path to config YAML file")
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--dtype', type=str, default="int8")
    args = parser.parse_args()
    cfg = Config.from_yaml(args.config_path)

    # mutli-gpu processing
    dtype = args.dtype
    world_size = torch.cuda.device_count()
    if world_size > 1:
        mp.spawn(_worker, args=(world_size, cfg, dtype), nprocs=world_size)
    else:
        _worker(rank=0, world_size=1, cfg=cfg, dtype=dtype)