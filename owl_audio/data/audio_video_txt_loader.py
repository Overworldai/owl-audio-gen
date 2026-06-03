import os, glob, random, json
from pathlib import Path
import numpy as np
import hashlib

import torch
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
import av


class RandomAudioVideoTextFromMP4s:
    """
    Continuous iterator yielding (audio, video_latent, caption) triples.
    - audio       : [C, T_audio] float32, from raw mp4
    - video_latent: [T_lat, C, H, W] float32, mmap-loaded from pre-encoded .pt
    - caption     : str, loaded from a JSON captions file (empty string if missing)

    source   : directory of mp4s  ({source}/foo/bar.mp4)
    encoded  : directory of latents ({encoded}/foo/taehv1_5/000000_latent.pt)
    Matching is by immediate parent folder name.

    Temporal alignment:
      t_start / video_duration * n_latent_frames  → latent_start_frame
    so the latent window lines up with the audio window regardless of original fps.
    """

    def __init__(self, source, encoded, seed=None,
                 window_length=10.0, sample_rate=44100,
                 video_window_frames=75, expected_hw=(16, 32),
                 split='train', holdout_ratio=0.1, split_seed=123):
        self.window_length = window_length
        self.sample_rate = sample_rate
        self.window_length_samples = int(window_length * sample_rate)
        self.expected_hw = tuple(expected_hw)
        self.video_window_frames = video_window_frames
        self.encoded = Path(encoded)
        self.split = split
        self.holdout_ratio = holdout_ratio
        self.split_seed = split_seed

        self.pairs = self._find_pairs(source, encoded)
        self.caption_path = Path(self.encoded / 'captions.json')

        # select the pairs based on current 'split'
        if self.holdout_ratio > 0:
            filtered = []
            for mp4_path, chunk_path in self.pairs:
                is_holdout = self._is_holdout_latent(chunk_path)
                if (self.split == 'holdout') == is_holdout:
                    filtered.append((mp4_path, chunk_path))
            self.pairs = filtered

        if not self.pairs:
            raise RuntimeError("No paired (mp4, latent) files found.")
        print(f"[DataLoader({self.split})] Found {len(self.pairs)} paired chunks.")

        # Build a set of valid (stem, win_idx) keys from the already-filtered pairs
        valid_keys = set()
        for mp4_path, chunk_path in self.pairs:
            stem = mp4_path.stem
            win_idx = int(chunk_path.stem)
            valid_keys.add((stem, win_idx))

        self.caption_map = {}
        if self.caption_path is not None:
            raw_map = self._load_captions(self.caption_path)
            self.caption_map = {k: v for k, v in raw_map.items() if k in valid_keys}
            print(f"[DataLoader({self.split})] Loaded {len(self.caption_map)} captions")

        self.rng = random.Random(seed)
        self.meta = {}  # mp4 Path -> duration_s

    @staticmethod
    def _load_captions(captions_path):
        """
        Load captions JSON into a dict keyed by (mp4_stem, win_idx).

        Expected JSON format: a list of objects, each with at minimum:
          "mp4"     : "/path/to/video.mp4"   (stem used as key)
          "win_idx" : 711                    (int)
          "caption" : "..."                  (str)
        """
        with open(captions_path, "r") as f:
            records = json.load(f)
        mapping = {}
        for rec in records:
            stem = Path(rec["mp4"]).stem
            win  = int(rec["win_idx"])
            mapping[(stem, win)] = rec.get("caption", "")
        return mapping

    def _get_caption(self, mp4_path: Path, chunk_path: Path) -> str:
        """
        Look up the caption for this (mp4, chunk) pair.
        chunk_path.stem is the zero-padded chunk index, which equals win_idx.
        Returns an empty string when no caption is found.
        """
        if not self.caption_map:
            return ""
        stem    = mp4_path.stem
        win_idx = int(chunk_path.stem)
        return self.caption_map.get((stem, win_idx), "")

    @staticmethod
    def _find_mp4s(source):
        s = os.path.expanduser(str(source))
        p = Path(s)
        registry = p / "valid_mp4s.txt"
        if p.is_dir() and registry.exists():
            print(f"[DataLoader] Using registry {registry}")
            paths = [line.strip() for line in registry.read_text().splitlines() if line.strip()]
        elif p.is_dir():
            paths = glob.glob(str(p / "**/*.mp4"), recursive=True)
        else:
            paths = glob.glob(s, recursive=True)
        filtered = [x for x in paths if x.lower().endswith(".mp4") and '_' not in Path(x).stem]
        return [Path(x) for x in sorted(set(filtered))]

    @staticmethod
    def _find_pairs(source, encoded):
        encoded = Path(encoded)
        pairs = []
        for mp4 in RandomAudioVideoTextFromMP4s._find_mp4s(source):
            folder = mp4.parent.name
            video_name = mp4.stem
            latent_dir = encoded / folder / "taehv1_5" / f"{video_name}_latent"
            if latent_dir.exists():
                chunk_files = sorted(latent_dir.glob("*.pt"))
                for chunk_path in chunk_files:
                    pairs.append((mp4, chunk_path))
        return pairs
    
    def __iter__(self):
        return self

    def __next__(self):
        while True:
            if not self.pairs:
                raise RuntimeError("All pairs failed to load.")

            idx = self.rng.randrange(len(self.pairs))
            mp4_path, chunk_path = self.pairs[idx]

            try:
                # duration cache
                if mp4_path not in self.meta:
                    with av.open(str(mp4_path)) as c:
                        if not any(s.type == "audio" for s in c.streams):
                            self.pairs.pop(idx)
                            continue
                
                        dur = (c.duration / 1e6) if c.duration is not None else 600.0
                    self.meta[mp4_path] = float(dur)

                dur = self.meta[mp4_path]
                
                # chunk index from filename
                chunk_idx = int(chunk_path.stem)
                # since chunk <> window
                t_start = chunk_idx * self.window_length

                if t_start + self.window_length > dur:
                    self.pairs.pop(idx)
                    continue

                # decode aligned audio
                audio = self._decode_audio(str(mp4_path), t_start) # [C, T_audio]
                
                # load latent chunk
                video = self._load_latent_chunk(chunk_path)

                # look up caption (empty string if not found)
                caption = self._get_caption(mp4_path, chunk_path)

                return audio, video, caption  # ([C, T_audio], [T_lat, C, H, W], str)
            
            except Exception as e:
                self.pairs.pop(idx)
                continue

    def _decode_audio(self, path, t_sec):
        resampler = av.AudioResampler(format="fltp", layout="stereo", rate=self.sample_rate)
        chunks = []
        needed = self.window_length_samples

        with av.open(path) as container:
            a_stream = next(s for s in container.streams if s.type == "audio")
            tb = a_stream.time_base
            container.seek(int(max(0.0, t_sec) / float(tb)), stream=a_stream, backward=True)

            started = False
            done = False
            for pkt in container.demux(a_stream):
                for frame in pkt.decode():
                    if not started:
                        if frame.time is not None:
                            frame_end = frame.time + frame.samples / max(1, frame.sample_rate)
                            if frame_end + 1e-3 < t_sec:
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
            raise RuntimeError("No audio decoded")

        audio = np.concatenate(chunks, axis=1)
        if audio.shape[1] < needed:
            pad = np.zeros((audio.shape[0], needed - audio.shape[1]), dtype=audio.dtype)
            audio = np.concatenate([audio, pad], axis=1)
        else:
            audio = audio[:, :needed]
        return audio  # [C, T_audio] float32

    def _is_holdout_latent(self, chunk_path):
        if self.holdout_ratio <= 0:
            return False
        key = f"{self.split_seed}:{str(chunk_path)}".encode("utf-8")
        h = hashlib.sha1(key).hexdigest()
        v = int(h[:8], 16) / 0xFFFFFFFF
        return v < self.holdout_ratio

    def _load_latent_chunk(self, chunk_path):
        payload = torch.load(str(chunk_path), map_location='cpu', weights_only=True)
        if isinstance(payload, dict):   # int8
            latent = payload['lat_q'].float()
            scale = payload['scale'].float()
            return (latent / 127.0) * scale
        return torch.load(str(chunk_path), map_location='cpu', mmap=True)   # bf16/fp16/fp32


class RandomAudioVideoTextDataset(IterableDataset):
    """Infinite stream of (audio [C, T], video [T_lat, C, H, W], caption str) triples in bfloat16."""
    def __init__(self, source, encoded, seed=0,
                 window_length=10.0, sample_rate=44100,
                 video_window_frames=75, expected_hw=(16, 32),
                 split='train', holdout_ratio=0.1):
        super().__init__()
        self.source = source
        self.encoded = encoded
        self.seed = int(seed)
        self.window_length = window_length
        self.sample_rate = sample_rate
        self.video_window_frames = video_window_frames
        self.expected_hw = expected_hw
        self.split = split
        self.holdout_ratio = holdout_ratio

    def __iter__(self):
        info = get_worker_info()
        wid = info.id if info else 0
        wseed = (torch.initial_seed() + self.seed + wid) % (2**32)
        rng = RandomAudioVideoTextFromMP4s(
            self.source, self.encoded, seed=int(wseed),
            window_length=self.window_length,
            sample_rate=self.sample_rate,
            video_window_frames=self.video_window_frames,
            expected_hw=self.expected_hw,
            split=self.split,
            holdout_ratio=self.holdout_ratio,
            split_seed=self.seed,  # fixed at 123
        )
        for audio, video, caption in rng:
            yield torch.from_numpy(audio).bfloat16(), video.bfloat16(), caption

def get_loader(batch_size, **data_kwargs):
    if "seed" not in data_kwargs:
        data_kwargs["seed"] = 123
    ds = RandomAudioVideoTextDataset(**data_kwargs)
    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
        drop_last=True,
        multiprocessing_context="spawn",
    )


if __name__ == "__main__":
    import time

    loader = get_loader(
        2,
        source="/path/to/mp4s",
        encoded="/path/to/encoded_latents",
        captions="/path/to/captions.json",  # optional; omit or set None to disable
        window_length=10.0,
        sample_rate=44100,
        video_window_frames=75,
    )

    loader_iter = iter(loader)
    total_time = 0.0
    for i in range(10):
        t0 = time.time()
        audio, video, captions = next(loader_iter)
        t1 = time.time()
        elapsed = t1 - t0
        total_time += elapsed
        if i == 0:
            print(f"Audio shape  : {tuple(audio.shape)}, dtype={audio.dtype}")
            print(f"Video shape  : {tuple(video.shape)}, dtype={video.dtype}")
            print(f"Caption[0]   : {captions[0][:80]}...")
        print(f"Batch {i+1:2d}: {elapsed:.3f}s")

    print(f"\nAverage over 10 batches: {total_time / 10:.3f}s")