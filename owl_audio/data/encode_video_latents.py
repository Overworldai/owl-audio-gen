import glob
import os
from pathlib import Path
import av
import numpy as np
import torch
from tqdm import tqdm
import sys


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

def _iter_video_chunks(mp4_path, chunk_size, resize, target_fps=24):
        container = av.open(str(mp4_path))
        video_stream = container.streams.video[0]
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
                
                if len(frames) >= chunk_size:
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
     device, 
     dtype, 
     resize, 
     target_fps,
     pbar=None
    ):
        """Load a video in chunks and encode each chunk with a VAE.
        Yields per-chunk latents on CPU to avoid accumulating full video in memory.
        """
        for chunk in _iter_video_chunks(video_path, chunk_size, resize, target_fps):
            video = torch.from_numpy(chunk).pin_memory().to(device, dtype=dtype, non_blocking=True)
            video = video.permute(0, 3, 1, 2).contiguous() / 255.0
            video = video.unsqueeze(0)
            encoded = vae_encode_fn(video)
            if hasattr(encoded, "latent_dist"):
                latent = encoded.latent_dist.mode() if hasattr(encoded.latent_dist, "mode") else encoded.latent_dist.sample()
            else:
                latent = encoded

            if pbar is not None:
                pbar.update(1)
            yield latent.detach().cpu()

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
          target_fps=30,
          device="cuda", 
          dtype=torch.bfloat16,
):
    missing = _find_missing_latents(source, encoded, vae_name)
    
    if len(missing) == 0:
         print("All videos already encoded")
         return
    print(f"Encoding {len(missing)} missing latents")

    for mp4_path, latent_path in missing:
        try:
            chunk_dir = latent_path.with_suffix("")
            chunk_dir.mkdir(parents=True, exist_ok=True)

            with tqdm(desc=mp4_path.name, unit="chunk") as pbar:
                 for chunk_idx, latent in enumerate(
                      encode_video_chunks(
                           mp4_path,
                           vae_encode_fn=vae_encode_fn,
                           chunk_size=chunk_size,
                           device=device,
                           dtype=dtype,
                           resize=resize,
                           target_fps=target_fps,
                           pbar=pbar
                      )
                 ):
                      # save latent chunks directly
                      chunk_path = chunk_dir / f"{chunk_idx:06d}.pt"
                      torch.save(latent.to(torch.float16), chunk_path)
        except Exception as e:
            print(f"Encoding failed for {mp4_path.name}: {e}")

            if chunk_dir.exists():
                for p in chunk_dir.glob("*.pt"):
                    p.unlink()
            continue

        print("Video encoding complete...")


if __name__ == '__main__':
    video_vae_ckpt = "/workspace/owl-audio-gen/taehv/taehv1_5.pth"
    
    # load vae encode fn
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.append(str(repo_root))
    from taehv import TAEHV
    _taehv = TAEHV(video_vae_ckpt).cuda().bfloat16().eval()
    video_encode_fn = lambda x: _taehv.encode_video(x)
     
    target_fps = 30
    window_length = 10.0
    desired_chunk_size = window_length * target_fps # chunk <> window

    encode_missing_videos(
        source="/workspace/dataset/source/",
        encoded="/workspace/dataset/encoded/",
        vae_encode_fn=video_encode_fn,
        vae_name="taehv1_5",
        resize=(640, 360), # set to 360p for now
        chunk_size=desired_chunk_size,
        target_fps=target_fps
    )