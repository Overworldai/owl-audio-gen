"""Direct-from-mp4 (video + audio) dataloader — no pre-encoded latents.

For the owlctrl-style dataset: a split root holds hundreds of thousands of
sample dirs, each with one video.mp4 (720p / 60fps / ~10min, audio embedded):

    {source}/{split}/<sample_id>/video.mp4

Each dataset item is a fixed-length window decoded straight from an mp4:

    audio : [C, T_audio] bf16            raw waveform @ sample_rate
    video : [T, 3, H, W] uint8           raw RGB frames in [0, 255] @ video_fps

Both windows cover the SAME time span [t_start, t_start + window_length), so the
trainer can encode each with its own VAE on the fly. Video is returned as uint8
(4x smaller over the worker pipe); the trainer casts to bf16 / [0,1] on the GPU.

Speed / scale design:
  * The split is enumerated ONCE with os.scandir (a single syscall stream, NOT
    os.listdir / glob / os.walk, which crash or hang over a 700k-entry root in a
    multi-process loader). Only sample-IDs are cached as an .npz under cache_dir.
    Rank 0 builds it; other ranks wait on a barrier and read it. No per-file
    probing, no separate offline script.
  * Clip length is assumed nominal (`clip_seconds`); the global timeline is
    virtualized into a CONSTANT number of fixed windows per video, so the global
    index maps to (video, window) by integer division — no per-window arrays.
  * Each window is decoded with container.seek(backward=True) to the nearest
    keyframe before the window, then frames are subsampled to `video_fps` (a true
    raw-frame downsample BEFORE encoding — never a latent frameskip) and resized
    inside the decoder. Cost is independent of where the window sits in the file.
  * Decode is single-threaded per stream; parallelism comes from DataLoader
    workers. Corrupt/short clips retry a DIFFERENT deterministic window rather
    than dropping, so DDP ranks keep matching batch sizes.
"""

import os, hashlib
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
import av

VIDEO_FILENAME = "video.mp4"


# --------------------------------------------------------------------------- #
# Index: scandir sample-IDs once (single process), cache, reuse everywhere
# --------------------------------------------------------------------------- #
def _scan_sample_ids(split_dir):
    """One os.scandir pass over the split root → sorted sample-dir names.

    Uses entry.is_dir(follow_symlinks=False), which reads the dirent d_type
    cached by scandir (no extra stat per entry on Linux). Never os.listdir.
    """
    ids = []
    with os.scandir(split_dir) as it:
        for e in it:
            if e.name.startswith("."):
                continue
            if e.is_dir(follow_symlinks=False):
                ids.append(e.name)
    return sorted(ids)


def _index_path(split_dir, cache_dir):
    key = f"{os.path.abspath(split_dir)}|video_audio_v2"
    h = hashlib.sha1(key.encode()).hexdigest()[:12]
    return Path(cache_dir) / f"video_audio_ids_{h}.npz"


def _build_index(split_dir, cache_dir):
    """Cache the sample-ID list. Rank 0 scandirs + writes; others barrier + read."""
    cache_dir = Path(cache_dir)
    idx_path = _index_path(split_dir, cache_dir)
    rank = dist.get_rank() if dist.is_initialized() else 0

    if not idx_path.exists() and rank == 0:
        cache_dir.mkdir(parents=True, exist_ok=True)
        ids = _scan_sample_ids(split_dir)
        if not ids:
            raise RuntimeError(f"No sample dirs under {split_dir}")
        tmp = idx_path.with_name(f"{idx_path.name}.{os.getpid()}.tmp")
        # Pass an open file handle, NOT a path: np.savez appends ".npz" to a path
        # that doesn't end in .npz, which breaks the os.replace below.
        with open(tmp, "wb") as f:
            np.savez(f, ids=np.asarray(ids))
        os.replace(tmp, idx_path)

    if dist.is_initialized():
        dist.barrier()

    with np.load(idx_path, allow_pickle=False) as z:
        return z["ids"]


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class VideoAudioWindows(Dataset):
    """Window-sampled (audio, raw-video-frames) dataset decoded directly from mp4."""

    _MAX_DECODE_RETRIES = 8     # random alternate windows on failure
    _FALLBACK_VIDEOS = 32       # then scan window-0 of successive videos

    def __init__(self, split_dir, *, window_length=10.0, sample_rate=48000,
                 video_fps=30, video_size=(640, 320), clip_seconds=600.0,
                 cache_dir=None, split="train"):
        super().__init__()
        self.split_dir = str(split_dir)
        self.window_length = float(window_length)
        self.sample_rate = int(sample_rate)
        self.window_samples = int(self.window_length * self.sample_rate)
        self.video_fps = float(video_fps)
        self.window_frames = int(round(self.window_length * self.video_fps))
        # video_size is (width, height) — same convention as encode_video_latents
        self.target_w = int(video_size[0])
        self.target_h = int(video_size[1])

        cache_dir = cache_dir or os.path.join(os.getcwd(), ".cache", "video_audio")
        self.ids = _build_index(self.split_dir, cache_dir)

        # Constant number of non-overlapping windows per (nominal-length) clip,
        # so global_idx -> (video, window) is pure integer arithmetic. Short
        # clips simply fail a tail window and fall back to the retry path.
        n_clip_frames = int(float(clip_seconds) * self.video_fps)
        self.windows_per_video = max(1, n_clip_frames // self.window_frames)

        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"[video_audio_loader({split})] {len(self.ids)} videos x "
                  f"{self.windows_per_video} windows = {len(self)} items "
                  f"({self.window_frames} frames @ {self.video_fps}fps, "
                  f"{self.target_w}x{self.target_h}); root={self.split_dir}")

    def __len__(self):
        return int(len(self.ids) * self.windows_per_video)

    def _video_path(self, vid):
        return os.path.join(self.split_dir, str(self.ids[vid]), VIDEO_FILENAME)

    def _load_window(self, vid, t_start):
        path = self._video_path(vid)
        video = self._decode_video(path, t_start)   # [T, 3, H, W] uint8
        audio = self._decode_audio(path, t_start)   # [C, T_audio] float32
        return torch.from_numpy(audio).bfloat16(), torch.from_numpy(video)

    def _zeros_item(self):
        # Silent / black fallback with correct shapes so batches & DDP stay aligned.
        audio = torch.zeros(2, self.window_samples, dtype=torch.bfloat16)
        video = torch.zeros(self.window_frames, 3, self.target_h, self.target_w, dtype=torch.uint8)
        return audio, video

    def __getitem__(self, idx):
        # Bad/short/corrupt clips are common at 700k scale. __getitem__ must NEVER
        # raise — a raised exception kills the worker and desyncs that DDP rank
        # while others keep going. So we fall back, never crash.
        n = len(self)
        W = self.windows_per_video
        nv = len(self.ids)
        last_err = None

        # 1. requested window, then a few random alternates across the dataset
        for attempt in range(self._MAX_DECODE_RETRIES):
            j = int(idx) if attempt == 0 else int(
                np.random.RandomState((int(idx) * 2654435761 + attempt) & 0xFFFFFFFF).randint(n)
            )
            vid, win = divmod(j, W)
            t_start = float(win * self.window_frames) / self.video_fps
            try:
                return self._load_window(vid, t_start)
            except Exception as e:
                last_err = e

        # 2. guaranteed-ish fallback: window 0 (first window_length s) of successive
        #    videos — the start of a clip is essentially always decodable, so this
        #    almost always returns REAL data rather than zeros.
        base_vid = (int(idx) // W) % nv
        for k in range(self._FALLBACK_VIDEOS):
            try:
                return self._load_window((base_vid + k) % nv, 0.0)
            except Exception as e:
                last_err = e

        # 3. absolute last resort: never crash the rank.
        print(f"[video_audio_loader] WARNING: idx={idx} -> zeros fallback; "
              f"last error: {type(last_err).__name__}: {last_err}", flush=True)
        return self._zeros_item()

    def _decode_video(self, path, t_start):
        n = self.window_frames
        th, tw = self.target_h, self.target_w
        dt = 1.0 / self.video_fps
        out = np.empty((n, th, tw, 3), dtype=np.uint8)

        with av.open(path) as container:
            stream = container.streams.video[0]
            try:
                stream.thread_type = "NONE"
            except Exception:
                stream.thread_count = 1
            tb = stream.time_base

            container.seek(int(max(0.0, t_start) / float(tb)),
                           stream=stream, backward=True, any_frame=False)

            got = 0
            next_t = t_start
            for frame in container.decode(stream):
                if frame.pts is None:
                    continue
                t = float(frame.pts * tb)
                if t + 1e-6 < next_t:
                    continue  # raw-frame downsample to video_fps (pre-encode)
                rgb = frame.reformat(width=tw, height=th, format="rgb24").to_ndarray()
                out[got] = rgb
                got += 1
                next_t += dt
                if got >= n:
                    break

        if got < n:
            raise RuntimeError(f"got {got}/{n} video frames ({path} @ {t_start:.2f}s)")
        # [T, H, W, 3] uint8 -> [T, 3, H, W] uint8
        return np.ascontiguousarray(out.transpose(0, 3, 1, 2))

    def _decode_audio(self, path, t_start):
        resampler = av.AudioResampler(format="fltp", layout="stereo", rate=self.sample_rate)
        chunks = []
        needed = self.window_samples

        with av.open(path) as container:
            a_stream = next(s for s in container.streams if s.type == "audio")
            tb = a_stream.time_base
            container.seek(int(max(0.0, t_start) / float(tb)), stream=a_stream, backward=True)

            started = False
            done = False
            for pkt in container.demux(a_stream):
                for frame in pkt.decode():
                    if not started:
                        if frame.time is not None:
                            frame_end = frame.time + frame.samples / max(1, frame.sample_rate)
                            if frame_end + 1e-3 < t_start:
                                continue
                        started = True
                    for rf in resampler.resample(frame):
                        chunks.append(rf.to_ndarray())
                    if sum(c.shape[1] for c in chunks) >= needed:
                        done = True
                        break
                if done:
                    break
            for rf in resampler.resample(None):
                chunks.append(rf.to_ndarray())

        if not chunks:
            raise RuntimeError(f"no audio decoded ({path} @ {t_start:.2f}s)")

        audio = np.concatenate(chunks, axis=1)
        if audio.shape[1] < needed:
            pad = np.zeros((audio.shape[0], needed - audio.shape[1]), dtype=audio.dtype)
            audio = np.concatenate([audio, pad], axis=1)
        else:
            audio = audio[:, :needed]
        return audio  # [C, T_audio] float32


# --------------------------------------------------------------------------- #
# Sampler + loader
# --------------------------------------------------------------------------- #
class _AutoEpochDistributedSampler(DistributedSampler):
    """DistributedSampler that re-shuffles each time it is iterated, so the
    trainer's epoch loop sees a fresh ordering without calling set_epoch."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._auto_epoch = 0

    def __iter__(self):
        super().set_epoch(self._auto_epoch)
        self._auto_epoch += 1
        return super().__iter__()


def get_loader(batch_size, source, split="train", window_length=10.0,
               sample_rate=48000, video_fps=30, video_size=(640, 320),
               clip_seconds=600.0, num_workers=4, prefetch_factor=4,
               pin_memory=True, cache_dir=None, **_):
    # Explicit train/ and eval/ subdirs under `source`.
    split_dir = os.path.join(os.path.expanduser(str(source)), str(split))

    ds = VideoAudioWindows(
        split_dir,
        window_length=window_length,
        sample_rate=sample_rate,
        video_fps=video_fps,
        video_size=video_size,
        clip_seconds=clip_seconds,
        cache_dir=cache_dir,
        split=split,
    )

    kw = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        drop_last=True,
        multiprocessing_context="spawn" if num_workers > 0 else None,
    )
    if num_workers > 0:
        kw["prefetch_factor"] = prefetch_factor

    world = dist.get_world_size() if dist.is_initialized() else 1
    if world > 1:
        sampler = _AutoEpochDistributedSampler(
            ds, num_replicas=world, rank=dist.get_rank(),
            shuffle=(split == "train"), seed=0, drop_last=True,
        )
        return DataLoader(ds, sampler=sampler, **kw)
    return DataLoader(ds, shuffle=(split == "train"), **kw)


# --------------------------------------------------------------------------- #
# Standalone cache build:  python owl_audio/data/video_audio_loader.py <config>
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import argparse
    from owl_audio.configs import Config

    parser = argparse.ArgumentParser(
        description="Pre-build the video_audio_loader sample-ID cache (single process)."
    )
    parser.add_argument("config_path", help="Path to config YAML")
    args = parser.parse_args()

    dk = Config.from_yaml(args.config_path).train.data_kwargs
    source = os.path.expanduser(str(dk.source))
    cache_dir = dk.get("cache_dir", None) or os.path.join(os.getcwd(), ".cache", "video_audio")

    for split in ("train", "eval"):
        split_dir = os.path.join(source, split)
        print(f"[video_audio_loader] scanning {split_dir} ...", flush=True)
        ids = _build_index(split_dir, cache_dir)
        print(f"[video_audio_loader] {split}: {len(ids)} sample dirs cached -> "
              f"{_index_path(split_dir, cache_dir)}", flush=True)

