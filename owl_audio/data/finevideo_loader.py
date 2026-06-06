import random, json
from pathlib import Path
import hashlib
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
from torchaudio.transforms import Resample
import av

class FineVideoLoader:
    """Loads videos from FineVideo (to be used with IterableDataset)"""
    def __init__(self, source, seed=None,
                 window_length=10.0, sample_rate=44100,
                 video_window_frames=75, expected_hw=(16, 32),
                 video_size=(640, 320), video_fps=30.0, 
                 split='train', eval_ratio=0.1, split_seed=123):
        super().__init__()
        self.source = source
        self.seed = seed
        self.window_length = window_length
        self.sample_rate = sample_rate
        self.video_window_frames = video_window_frames
        self.expected_hw = expected_hw
        self.video_size = video_size
        self.video_fps = video_fps
        self.split = split
        self.eval_ratio = eval_ratio
        self.split_seed = split_seed

        # fixed metadata file (jsonl)
        metadata_path = Path(f"{source}/metadata.jsonl")
        self.records = FineVideoLoader._load_records(metadata_path)

        # filter records by 'train' and 'eval' split
        if self.eval_ratio > 0:
            filtered = []
            for record in self.records:
                is_holdout = self._is_holdout(record['chunk_id'])
                if (self.split == 'eval') == is_holdout:
                    filtered.append(record)
                self.records = filtered
        if not self.records:
            raise RuntimeError("No paired (audio, video, caption) found.")
        print(f"[DataLoader({self.split})] Found {len(self.records)} records.")
     
        rng = random.Random(seed)
        rng.shuffle(self.records)
        self.idx = 0

    @staticmethod
    def _load_records(metadata_path):
        path = Path(metadata_path)
        if not path.exists():
            return list()
        suffix = path.suffix.lower()
        if suffix == ".jsonl":
            records = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    records.append(json.loads(line))
            return records
        if suffix == ".json":
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                for key in ("records", "items", "data"):
                    if key in data and isinstance(data[key], list):
                        return data[key]
            return list()
        return list()

    @staticmethod
    def _decode_video_audio(mp4_path, window_length, target_audio_sr, video_width, video_height, video_fps):
        
        with av.open(mp4_path) as container:
            video_stream = container.streams.video[0]
            audio_stream = container.streams.audio[0]

            native_fps = float(video_stream.average_rate)
            native_h = video_stream.height
            native_w = video_stream.width
            native_sr = audio_stream.sample_rate

            keep_every   = max(1, round(native_fps / video_fps)) if video_fps and video_fps < native_fps else 1
            video_needs_resize = (native_h != video_height or native_w != video_width)

            i = 0
            frames, chunks = [], []

            for packet in container.demux(video=0, audio=0):
                if packet.size == 0:
                    continue

                try:
                    # decode video stream
                    for frame in packet.decode():
                        if isinstance(frame, av.VideoFrame):
                            if i % keep_every == 0:
                                arr = frame.to_ndarray(format="rgb24")
                                frames.append(torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0)
                            i += 1

                        # decode audio stream
                        elif isinstance(frame, av.AudioFrame):
                            arr = frame.to_ndarray()
                            if arr.dtype != np.float32:
                                arr = arr.astype(np.float32) / np.iinfo(arr.dtype).max
                            chunks.append(torch.from_numpy(arr))

                except Exception as e:
                    print("decode error:", mp4_path, repr(e))
                    raise
            
            if not frames:
                raise RuntimeError(f"No video frames found in {mp4_path}")
            if not chunks:
                raise RuntimeError(f"No audio stream found in {mp4_path}")

        video = torch.stack(frames)  # (T, C, H, W)
        audio = torch.cat(chunks, dim=-1)  # (C, T)

        if video_needs_resize:
            video = F.interpolate(
                video,
                size=(video_height, video_width),
                mode="bilinear",
                align_corners=False,
            )

        if native_sr != target_audio_sr:
            audio = Resample(native_sr, target_audio_sr)(audio)

        # pad/trim video
        expected_frames = int(video_fps * window_length) 
        video = video[:expected_frames]
        if video.shape[0] < expected_frames:
            pad = torch.zeros(expected_frames - video.shape[0], *video.shape[1:], dtype=video.dtype)
            video = torch.cat([video, pad], dim=0)

        # pad/trim audio
        expected_samples = int(target_audio_sr * window_length)
        audio = audio[..., :expected_samples]
        if audio.shape[-1] < expected_samples:
            pad = torch.zeros(*audio.shape[:-1], expected_samples - audio.shape[-1], dtype=audio.dtype)
            audio = torch.cat([audio, pad], dim=-1)

        # convert mono → stereo
        if audio.shape[0] == 1:
            audio = audio.repeat(2, 1)

        return video, audio, native_sr

    def _is_holdout(self, chunk_path):
        if self.eval_ratio <= 0:
            return False
        key = f"{self.split_seed}:{str(chunk_path)}".encode("utf-8")
        h = hashlib.sha1(key).hexdigest()
        v = int(h[:8], 16) / 0xFFFFFFFF
        return v < self.eval_ratio

    def __iter__(self):
        return self

    def __next__(self):
        idx = self.idx % len(self.records)
        self.idx += 1

        record = self.records[idx]
        try:
            video, audio, nativr_sr = FineVideoLoader._decode_video_audio(
                record["chunk_video_path"], self.window_length, 
                self.sample_rate, self.video_size[0], 
                self.video_size[1], self.video_fps,
            )
        except Exception as e:
            raise RuntimeError(str(e))
        
        caption = record.get("caption", "") 
        return audio, video, caption


class FineVideoDataset(IterableDataset):
    def __init__(self, source, seed=0,
                 window_length=10.0, sample_rate=44100,
                 video_window_frames=75, expected_hw=(16, 32), 
                 video_size=(640, 320), split='train', 
                 video_fps=30.0, eval_ratio=0.1, split_seed=123):
        super().__init__()
        self.source = source
        self.seed = seed
        self.window_length = window_length
        self.sample_rate = sample_rate
        self.video_window_frames = video_window_frames
        self.expected_hw = expected_hw
        self.video_size = video_size
        self.video_fps = video_fps
        self.split = split
        self.eval_ratio = eval_ratio
        self.split_seed = split_seed

    def __iter__(self):
        info = get_worker_info()
        wid = info.id if info else 0
        wseed = (torch.initial_seed() + self.seed + wid) % (2**32)
        finevid = FineVideoLoader(
            self.source, seed=int(wseed),
            window_length=self.window_length,
            sample_rate=self.sample_rate,
            video_window_frames=self.video_window_frames,
            expected_hw=self.expected_hw,
            video_size=self.video_size,
            video_fps=self.video_fps,
            split=self.split,
            eval_ratio=self.eval_ratio,
            split_seed=self.seed # fixed at 123
        )
        for audio, video, caption in finevid:
            yield audio.bfloat16(), video.bfloat16(), caption


def get_loader(batch_size, **data_kwargs):
    if "seed" not in data_kwargs:
        data_kwargs["seed"] = 123
    ds = FineVideoDataset(**data_kwargs)
    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
        drop_last=True,
        multiprocessing_context="spawn",
    )

def sanity_check():
    # Sanity check: verify train and eval splits are disjoint
    train_loader = FineVideoLoader(source='/workspace/dataset/source', seed=123, split='train', eval_ratio=0.1)
    eval_loader = FineVideoLoader(source='/workspace/dataset/source', seed=123, split='eval', eval_ratio=0.1)
    
    train_ids = {record['chunk_id'] for record in train_loader.records}
    eval_ids = {record['chunk_id'] for record in eval_loader.records}
    
    intersection = train_ids & eval_ids
    print(f"Train split size: {len(train_ids)}")
    print(f"Eval split size: {len(eval_ids)}")
    print(f"Intersection size: {len(intersection)}")
    
    if intersection:
        print(f"ERROR: Splits are not disjoint! Overlapping IDs: {intersection}")
    else:
        print("✓ Train and eval splits are properly disjoint")

if __name__ == '__main__':
    # sanity_check()
    import time

    loader = get_loader(
        2,
        source="/workspace/dataset/source",
        window_length=10.0,
        sample_rate=44100,
        video_window_frames=75,
    )

    loader_iter = iter(loader)
    total_time = 0.0
    for i in range(100):
        t0 = time.time()
        audio, video, caption = next(loader_iter)
        t1 = time.time()
        elapsed = t1 - t0
        total_time += elapsed
        if i == 0:
            print(f"Audio shape : {tuple(audio.shape)}, dtype={audio.dtype}")
            print(f"Video shape : {tuple(video.shape)}, dtype={video.dtype}")
        print(f"Batch {i+1:2d}: {elapsed:.3f}s")

    print(f"\nAverage over 10 batches: {total_time / 10:.3f}s")





