"""Direct-from-mp4 (video + audio) loader, manifest-driven.

Adapted from the world-model RGB loader (different_loader.py): same robust
manifest index + games blacklist + global window virtualization, but for the
video->audio task (no controls), and it ALSO decodes the aligned audio window.

Per-sample shared-FS layout (built upstream by scripts/index_mp4s.py):
    {root}/{split}/{id}/video.mp4
    {root}/{split}/{id}/metadata.json               (game_exe -> blacklist)
    {root}/{split}/{id}/object_store_manifest.json  (v2: mp4_uri, fps, frames, h, w)

Index: each rank reads a shard of the manifests (fps/frames/h/w come straight
from JSON — NO per-file probe), filters blacklisted games, all_gathers, and rank
0 writes one cached .npz. Because window counts come from the REAL frame count,
no window ever runs past a clip's end (the old short-clip stall is gone).

Each item is one time window:
    audio : [C, T_audio] bf16            raw waveform @ sample_rate
    video : decode_mode == "yuv" -> [T, H*3//2, W] uint8  packed yuv420p, native
                                                          res (convert+resize on GPU)
            decode_mode == "rgb" -> [T, 3, oh, ow] uint8  RGB, resized in-worker
Both windows cover the same [t_start, t_start+window_length) span. Source frames
are subsampled to `video_fps` IN THE WORKER (a real frame downsample before any
encoding — never a latent frameskip).
"""

import hashlib
import json
import os
import urllib.parse
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
import av


MANIFEST_FILENAME = "object_store_manifest.json"
MANIFEST_VERSION_MIN = 2

# Excluded `game_exe` values are listed in blacklist.txt (one per line, '#'
# comments allowed), kept OUT of the repo. If the file is absent, no exe
# blacklist is enforced.
_DEFAULT_BLACKLIST_PATH = Path(__file__).resolve().parents[2] / "blacklist.txt"


def _load_blacklist(path=None):
    p = Path(path) if path else _DEFAULT_BLACKLIST_PATH
    if not p.exists():
        return frozenset()
    out = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            out.append(line.casefold())
    return frozenset(out)


def _resolve_uri(uri: str) -> str:
    return urllib.parse.urlparse(uri).path if uri.startswith("file://") else uri


@torch.no_grad()
def yuv420p_to_rgb01(yuv_u8, out_hw, chunk=64):
    """Packed yuv420p [B,T,H*3//2,W] uint8 -> RGB [B,T,3,oh,ow] bf16 in [0,1].

    For decode_mode="yuv": the worker ships native packed yuv420p and this does
    the colorspace convert + resize on the GPU. Chunked over frames because a
    full-batch native-res float conversion would OOM. BT.709 limited-range
    approximation (good for taehv; not bit-exact to swscale's rgb24).
    """
    B, T, yh, W = yuv_u8.shape
    H = yh * 2 // 3
    ch = H // 4
    x = yuv_u8.reshape(B * T, yh, W)
    N = x.shape[0]
    oh, ow = int(out_hw[0]), int(out_hw[1])
    out = torch.empty(N, 3, oh, ow, device=x.device, dtype=torch.bfloat16)
    for s in range(0, N, chunk):
        e = min(N, s + chunk)
        c = x[s:e].float()
        Y = c[:, :H, :]
        U = c[:, H:H + ch, :].reshape(e - s, H // 2, W // 2)
        V = c[:, H + ch:H + 2 * ch, :].reshape(e - s, H // 2, W // 2)
        U = U.repeat_interleave(2, 1).repeat_interleave(2, 2)
        V = V.repeat_interleave(2, 1).repeat_interleave(2, 2)
        Yn = (Y - 16.0) * (255.0 / 219.0)
        Un = (U - 128.0) * (255.0 / 224.0)
        Vn = (V - 128.0) * (255.0 / 224.0)
        R = Yn + 1.5748 * Vn
        G = Yn - 0.1873 * Un - 0.4681 * Vn
        Bl = Yn + 1.8556 * Un
        rgb = torch.stack([R, G, Bl], 1).clamp_(0, 255).div_(255.0)
        rgb = F.interpolate(rgb, size=(oh, ow), mode="bilinear", align_corners=False)
        out[s:e] = rgb.bfloat16()
    return out.reshape(B, T, 3, oh, ow)


# --------------------------------------------------------------------------- #
# Manifest index (rank-sharded build, cached .npz). No per-file probing.
# --------------------------------------------------------------------------- #
def _index_path(split_dir, cache_dir, blacklist_sig=""):
    key = f"{os.path.abspath(split_dir)}|video_audio_manifest_v1|bl={blacklist_sig}"
    h = hashlib.sha1(key.encode()).hexdigest()[:12]
    return Path(cache_dir) / f"video_audio_index_{h}.npz"


def _scan_sample_names(split_dir):
    """One os.scandir pass over the split root → sample-dir names. NOT glob /
    listdir (both crawl for minutes over a 700k-entry NFS dir)."""
    names = []
    with os.scandir(split_dir) as it:
        for e in it:
            if e.name.startswith("."):
                continue
            if e.is_dir(follow_symlinks=False):
                names.append(e.name)
    return names


def _read_sample(split_dir, name, blacklist):
    """Read one sample's manifest (+ blacklist check). Returns a metadata tuple
    or None to drop the sample. JSON-only — no mp4 decode."""
    d = os.path.join(split_dir, name)
    try:
        if blacklist:   # only pay the metadata.json read when a blacklist is active
            try:
                with open(os.path.join(d, "metadata.json")) as f:
                    if json.load(f).get("game_exe", "").casefold() in blacklist:
                        return None
            except FileNotFoundError:
                pass
        with open(os.path.join(d, MANIFEST_FILENAME)) as f:
            man = json.load(f)
        if int(man.get("version", 0)) < MANIFEST_VERSION_MIN:
            return None
        return (str(man["mp4_uri"]), int(round(float(man["fps"]))),
                int(man["frames"]), int(man["height"]), int(man["width"]))
    except Exception:
        return None


def _load_or_build_index(split_dir, cache_dir):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    blacklist = _load_blacklist()
    blacklist_sig = hashlib.sha1(",".join(sorted(blacklist)).encode()).hexdigest()[:8]
    cache_path = _index_path(split_dir, cache_dir, blacklist_sig)
    rank = dist.get_rank() if dist.is_initialized() else 0
    world = dist.get_world_size() if dist.is_initialized() else 1

    if dist.is_initialized():
        dist.barrier()
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as z:
            idx = {k: z[k] for k in z.files}
        if dist.is_initialized():
            dist.barrier()
        return idx

    names = _scan_sample_names(split_dir)        # single os.scandir pass
    if not names:
        raise FileNotFoundError(f"no sample dirs under {split_dir}")
    if not dist.is_initialized() or rank == 0:
        print(f"[video_audio_loader] blacklist: {len(blacklist)} entries"
              + ("" if blacklist else " (no blacklist.txt — not enforcing)"), flush=True)

    from concurrent.futures import ThreadPoolExecutor
    from tqdm import tqdm
    my = names[rank::world]
    cols = {"mp4_uri": [], "fps": [], "frames": [], "height": [], "width": []}
    n_bad = 0
    # JSON reads are I/O-bound → thread them so the (single-process) build over
    # hundreds of thousands of samples completes in minutes, not hours.
    with ThreadPoolExecutor(max_workers=64) as ex:
        it = ex.map(lambda nm: _read_sample(split_dir, nm, blacklist), my)
        for res in tqdm(it, total=len(my),
                        desc=f"[video_audio_loader] index/rank{rank}", disable=(rank != 0)):
            if res is None:
                n_bad += 1
                continue
            uri, fps, frames, h, w = res
            cols["mp4_uri"].append(uri)
            cols["fps"].append(fps)
            cols["frames"].append(frames)
            cols["height"].append(h)
            cols["width"].append(w)

    if dist.is_initialized():
        parts = [None] * world
        dist.all_gather_object(parts, (cols, n_bad))
        cols = {k: [] for k in cols}
        n_bad = 0
        for c, b in parts:
            for k, v in c.items():
                cols[k].extend(v)
            n_bad += b

    if not cols["mp4_uri"]:
        raise RuntimeError("no videos remain after blacklist/manifest filtering")
    if (not dist.is_initialized() or rank == 0):
        print(f"[video_audio_loader] indexed {len(cols['mp4_uri'])} videos "
              f"({n_bad} excluded by blacklist/bad manifest)", flush=True)

    idx = {
        "mp4_uri": np.asarray(cols["mp4_uri"]),
        "fps": np.asarray(cols["fps"], np.int64),
        "frames": np.asarray(cols["frames"], np.int64),
        "height": np.asarray(cols["height"], np.int64),
        "width": np.asarray(cols["width"], np.int64),
    }
    if rank == 0:
        tmp = cache_path.with_name(f"{cache_path.name}.{os.getpid()}.tmp")
        with open(tmp, "wb") as f:           # file handle: np.savez won't append .npz
            np.savez(f, **idx)
        os.replace(tmp, cache_path)
    if dist.is_initialized():
        dist.barrier()
    return idx


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class VideoAudioWindows(Dataset):
    """Per-video, time-aligned (audio, raw-video-frames) windows decoded from mp4."""

    _FALLBACK_VIDEOS = 4        # random window-0 alternates after same-video fallback
    _DECODE_TIMEOUT = 15.0      # seconds; ffmpeg I/O interrupt so a corrupt file can't hang

    def __init__(self, split_dir, *, window_length=10.0, sample_rate=48000,
                 video_fps=30, video_size=(640, 320), decode_mode="yuv",
                 cache_dir=None, split="train"):
        super().__init__()
        self.split_dir = str(split_dir)
        self.window_length = float(window_length)
        self.sample_rate = int(sample_rate)
        self.window_samples = int(self.window_length * self.sample_rate)
        self.video_fps = float(video_fps)
        self.window_frames = int(round(self.window_length * self.video_fps))
        self.target_w = int(video_size[0])     # (width, height) convention
        self.target_h = int(video_size[1])
        self.decode_mode = str(decode_mode)
        assert self.decode_mode in ("yuv", "rgb"), self.decode_mode

        cache_dir = cache_dir or os.path.join(os.getcwd(), ".cache", "video_audio")
        idx = _load_or_build_index(self.split_dir, cache_dir)
        self.mp4_uris = idx["mp4_uri"]
        fps = np.maximum(idx["fps"], 1)
        durations = idx["frames"] / fps                       # seconds (REAL per clip)

        # Non-overlapping time windows laid only within each clip's real length,
        # so a window NEVER runs past EOF. global idx -> (video, window) via cumsum.
        nwin = np.floor(durations / self.window_length).astype(np.int64)
        valid = nwin > 0
        self.mp4_uris = self.mp4_uris[valid]
        nwin = nwin[valid]
        if len(nwin) == 0:
            raise RuntimeError(f"no clip long enough for one {self.window_length}s window")
        self.cum = np.cumsum(nwin)
        self.total = int(self.cum[-1])

        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"[video_audio_loader({split})] {len(self.mp4_uris)} videos -> "
                  f"{self.total} windows ({self.window_frames} frames @ {self.video_fps}fps, "
                  f"mode={self.decode_mode}); root={self.split_dir}", flush=True)

    def __len__(self):
        return self.total

    def _idx_to_vid_win(self, j):
        vid = int(np.searchsorted(self.cum, j, side="right"))
        base = 0 if vid == 0 else int(self.cum[vid - 1])
        return vid, j - base

    def _zeros_item(self):
        audio = torch.zeros(2, self.window_samples, dtype=torch.bfloat16)
        if self.decode_mode == "yuv":
            yuv_h = self.target_h + self.target_h // 2   # placeholder shape (rare path)
            video = torch.zeros(self.window_frames, yuv_h, self.target_w, dtype=torch.uint8)
        else:
            video = torch.zeros(self.window_frames, 3, self.target_h, self.target_w, dtype=torch.uint8)
        return audio, video

    def __getitem__(self, idx):
        # Never raise and never stall long: bounded fallback, each decode itself
        # I/O-timeout-bounded. With real manifest frame counts + the blacklist,
        # failures should be rare; this is just defense in depth.
        nv = len(self.mp4_uris)
        req_vid, req_win = self._idx_to_vid_win(int(idx) % self.total)
        attempts = [
            (req_vid, float(req_win) * self.window_length),
            (req_vid, 0.0),
        ]
        for a in range(self._FALLBACK_VIDEOS):
            rv = int(np.random.RandomState((int(idx) * 2654435761 + a) & 0xFFFFFFFF).randint(nv))
            attempts.append((rv, 0.0))

        last_err = None
        for vid, t_start in attempts:
            try:
                path = _resolve_uri(str(self.mp4_uris[vid]))
                video = self._decode_video(path, t_start)
                audio = self._decode_audio(path, t_start)
                return torch.from_numpy(audio).bfloat16(), torch.from_numpy(video)
            except Exception as e:
                last_err = e

        print(f"[video_audio_loader] WARNING: idx={idx} -> zeros fallback; "
              f"last error: {type(last_err).__name__}: {last_err}", flush=True)
        return self._zeros_item()

    def _decode_video(self, path, t_start):
        n = self.window_frames
        dt = 1.0 / self.video_fps
        rgb_mode = self.decode_mode == "rgb"
        out = None

        with av.open(path, timeout=self._DECODE_TIMEOUT) as container:
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
                    continue  # subsample source fps -> video_fps (real frame drop)
                if rgb_mode:
                    arr = frame.reformat(width=self.target_w, height=self.target_h,
                                         format="rgb24").to_ndarray()  # [th, tw, 3]
                    if out is None:
                        out = np.empty((n, self.target_h, self.target_w, 3), np.uint8)
                else:
                    arr = frame.to_ndarray()  # packed yuv420p [H*3//2, W]
                    if out is None:
                        out = np.empty((n,) + arr.shape, np.uint8)
                out[got] = arr
                got += 1
                next_t += dt
                if got >= n:
                    break

        if out is None or got < n:
            raise RuntimeError(f"got {got}/{n} video frames ({path} @ {t_start:.2f}s)")
        if rgb_mode:
            return np.ascontiguousarray(out.transpose(0, 3, 1, 2))  # [T, 3, H, W]
        return out                                                  # [T, H*3//2, W]

    def _decode_audio(self, path, t_start):
        resampler = av.AudioResampler(format="fltp", layout="stereo", rate=self.sample_rate)
        chunks = []
        needed = self.window_samples

        with av.open(path, timeout=self._DECODE_TIMEOUT) as container:
            a_stream = next(s for s in container.streams if s.type == "audio")
            tb = a_stream.time_base
            container.seek(int(max(0.0, t_start) / float(tb)), stream=a_stream, backward=True)

            started = done = False
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
    """Re-shuffles each time it is iterated, so the epoch loop reshuffles without
    the caller having to call set_epoch."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._auto_epoch = 0

    def __iter__(self):
        super().set_epoch(self._auto_epoch)
        self._auto_epoch += 1
        return super().__iter__()


def get_loader(batch_size, source, split="train", window_length=10.0,
               sample_rate=48000, video_fps=30, video_size=(640, 320),
               decode_mode="yuv", num_workers=4, prefetch_factor=4,
               pin_memory=True, cache_dir=None, seed=0, **_):
    split_dir = os.path.join(os.path.expanduser(str(source)), str(split))
    ds = VideoAudioWindows(
        split_dir,
        window_length=window_length, sample_rate=sample_rate,
        video_fps=video_fps, video_size=video_size, decode_mode=decode_mode,
        cache_dir=cache_dir, split=split,
    )
    kw = dict(
        batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=(num_workers > 0), drop_last=True,
        multiprocessing_context="spawn" if num_workers > 0 else None,
    )
    if num_workers > 0:
        kw["prefetch_factor"] = prefetch_factor

    world = dist.get_world_size() if dist.is_initialized() else 1
    if world > 1:
        sampler = _AutoEpochDistributedSampler(
            ds, num_replicas=world, rank=dist.get_rank(),
            shuffle=(split == "train"), seed=int(seed), drop_last=True,
        )
        return DataLoader(ds, sampler=sampler, **kw)
    return DataLoader(ds, shuffle=(split == "train"), **kw)


# --------------------------------------------------------------------------- #
# Standalone index build:  python owl_audio/data/video_audio_loader.py <config>
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import argparse
    from owl_audio.configs import Config

    ap = argparse.ArgumentParser(description="Pre-build the manifest index cache.")
    ap.add_argument("config_path")
    args = ap.parse_args()

    dk = Config.from_yaml(args.config_path).train.data_kwargs
    source = os.path.expanduser(str(dk.source))
    cache_dir = dk.get("cache_dir", None) or os.path.join(os.getcwd(), ".cache", "video_audio")
    for split in ("train", "eval"):
        split_dir = os.path.join(source, split)
        print(f"[video_audio_loader] indexing {split_dir} ...", flush=True)
        idx = _load_or_build_index(split_dir, cache_dir)
        print(f"[video_audio_loader] {split}: {len(idx['mp4_uri'])} videos indexed", flush=True)
