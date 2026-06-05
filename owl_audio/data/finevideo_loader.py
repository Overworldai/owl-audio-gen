import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


def _load_jsonl(path: Path) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def _decode_video_audio(mp4_path, target_audio_sr, video_height, video_width, frames_per_second):
    import torch
    from torchvision.io import VideoReader
    from torchaudio.transforms import Resample

    reader = VideoReader(mp4_path, "video")
    meta = reader.get_metadata()
    fps = (meta.get("video", {}).get("fps", [25]) or [25])[0] or 25.0
    keep_every = max(1, round(fps / frames_per_second)) if frames_per_second and frames_per_second < fps else 1

    reader.set_current_stream("video")
    frames = [f["data"].float() / 255.0 for i, f in enumerate(reader) if i % keep_every == 0]
    video = torch.stack(frames) if frames else torch.zeros(1, 3, video_height, video_width)

    native_sr = int((meta.get("audio", {}).get("framerate", [target_audio_sr]) or [target_audio_sr])[0])
    reader.set_current_stream("audio")
    chunks = [c["data"] for c in reader]
    if chunks:
        import torch as _t
        audio = _t.cat(chunks, dim=-1)
    else:
        import torch as _t
        audio = _t.zeros(1, int(len(frames) / fps * target_audio_sr))
        native_sr = target_audio_sr

    if native_sr != target_audio_sr:
        audio = Resample(native_sr, target_audio_sr)(audio)

    return video, audio, native_sr


class FineVideoDataset:
    """
    PyTorch-compatible dataset for FineVideo chunks.

    Each item is a dict with keys:
        video [T,C,H,W], audio [C,N], text, chunk_id, video_id,
        start_s, end_s, category, fine_category, activities, tts_segments
    """

    def __init__(
        self,
        metadata_path: str,
        audio_sr: int = 44_100,
        video_height: int = 320,
        video_width: int = 568,
        frames_per_second: Optional[float] = None,
        categories: Optional[List[str]] = None,
        require_caption: bool = False,
    ):
        self.audio_sr = audio_sr
        self.video_height = video_height
        self.video_width = video_width
        self.frames_per_second = frames_per_second

        records = _load_jsonl(Path(metadata_path))

        if categories:
            cats = [c.lower() for c in categories]
            records = [r for r in records if any(
                c in r.get("content_parent_category", "").lower() or
                c in r.get("content_fine_category", "").lower()
                for c in cats
            )]

        if require_caption:
            records = [r for r in records if r.get("caption", "").strip()]

        self.records = [r for r in records if Path(r["chunk_video_path"]).exists()]
        print(F"FineVideoDataset: {len(self.records)} chunks loaded")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        import torch
        rec = self.records[idx]
        try:
            video, audio, _ = _decode_video_audio(
                rec["chunk_video_path"], self.audio_sr,
                self.video_height, self.video_width, self.frames_per_second,
            )
        except Exception as e:
            T = max(1, int((rec["end_s"] - rec["start_s"]) * (self.frames_per_second or 25)))
            video = torch.zeros(T, 3, self.video_height, self.video_width)
            audio = torch.zeros(2, int((rec["end_s"] - rec["start_s"]) * self.audio_sr))

        caption = rec.get("caption", "")

        return {
            "video": video,
            "audio":         audio,
            "text":          caption,
            "chunk_id":      rec["chunk_id"],
            "video_id":      rec["video_id"],
            "start_s":       rec["start_s"],
            "end_s":         rec["end_s"],
            "category":      rec.get("content_parent_category", ""),
            "fine_category": rec.get("content_fine_category", ""),
            "activities":    rec.get("activities", []),
            "tts_segments":  rec.get("tts_segments", []),
        }

    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        import torch
        max_t   = max(item["video"].shape[0] for item in batch)
        C, H, W = batch[0]["video"].shape[1:]
        video   = torch.zeros(len(batch), max_t, C, H, W)
        for i, item in enumerate(batch):
            video[i, :item["video"].shape[0]] = item["video"]

        max_n = max(item["audio"].shape[-1] for item in batch)
        Ca    = batch[0]["audio"].shape[0]
        audio = torch.zeros(len(batch), Ca, max_n)
        for i, item in enumerate(batch):
            audio[i, :, :item["audio"].shape[-1]] = item["audio"]

        keys = ["text", "chunk_id", "video_id", "start_s", "end_s",
                "category", "fine_category", "activities"]
        return {"video": video, "audio": audio, **{k: [item[k] for item in batch] for k in keys}}


# ── Pipeline ──────────────────────────────────────────────────────────────────

def run_pipeline(
    categories,
    root_dir="./finevideo_data",
    window_length=10.0,
    audio_sr=44_100,
    video_height=320,
    max_videos=None,
    hf_token=None,
    skip_download=False,
    skip_chunking=False,
) -> Path:
    if not skip_download:
        download_finevideo(categories, root_dir, max_videos, hf_token)

    if not skip_chunking:
        return chunk_finevideo(root_dir, window_length, audio_sr, video_height)

    return Path(root_dir) / "chunks_metadata.jsonl"


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--categories",     nargs="+", required=True)
    p.add_argument("--root_dir",       default="./finevideo_data")
    p.add_argument("--max_videos",     type=int,   default=None)
    p.add_argument("--hf_token",       default=None)
    p.add_argument("--window_length",  type=float, default=10.0)
    p.add_argument("--audio_sr",       type=int,   default=44_100)
    p.add_argument("--video_height",   type=int,   default=320)
    p.add_argument("--skip_download",  action="store_true")
    p.add_argument("--skip_chunking",  action="store_true")
    args = p.parse_args()

    run_pipeline(
        categories    = args.categories,
        root_dir      = args.root_dir,
        window_length = args.window_length,
        audio_sr      = args.audio_sr,
        video_height  = args.video_height,
        max_videos    = args.max_videos,
        hf_token      = args.hf_token,
        skip_download = args.skip_download,
        skip_chunking = args.skip_chunking,
    )