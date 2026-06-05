import json
import os
import subprocess
from pathlib import Path
from datasets import load_dataset


def download_finevideo(categories, root_dir="./finevideo_data", max_videos=None, hf_token=None):
    token = hf_token or os.environ.get("HF_TOKEN")
    raw_dir = Path(root_dir) / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("HuggingFaceFV/finevideo", split="train", streaming=True, token=token)

    cats_lower = [c.lower() for c in categories]
    count = 0
    for sample in ds:
        if max_videos and count >= max_videos:
            break
        parent = sample.get("content_parent_category", "").lower()
        fine = sample.get("content_fine_category", "").lower()
        if not any(c in parent or c in fine for c in cats_lower):
            continue

        vid_id  = sample["video_id"]
        out_path = raw_dir / f"{vid_id}.mp4"
        if out_path.exists():
            count += 1
            continue

        video_bytes = sample.get("video", {}).get("bytes")
        if not video_bytes:
            continue
        out_path.write_bytes(video_bytes)

        meta_path = raw_dir / f"{vid_id}.json"
        meta_path.write_text(json.dumps({k: v for k, v in sample.items() if k != "video"}, default=str))

        count += 1
    print(f"Download complete: {count} videos in {raw_dir}")

def _ffprobe_duration(mp4: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(mp4)],
        capture_output=True, text=True,
    )
    try:
        return float(result.stdout.strip())
    except ValueError:
        return 0.0

def _align_captions(activities: list, start_s: float, end_s: float) -> list:
    out = []
    for act in activities:
        a_start = act.get("start_time", 0)
        a_end   = act.get("end_time", 0)
        if a_end <= start_s or a_start >= end_s:
            continue
        clipped = dict(act)
        clipped["start_time"] = max(0.0, a_start - start_s)
        clipped["end_time"]   = min(end_s - start_s, a_end - start_s)
        out.append(clipped)
    return out

def _build_caption(activities: list) -> str:
    return " ".join(a.get("caption", "") for a in activities if a.get("caption", "")).strip()

def chunk_finevideo(root_dir="./finevideo_data", window_length=10.0, audio_sr=44_100, video_height=320):
    raw_dir    = Path(root_dir) / "raw"
    chunks_dir = Path(root_dir) / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    meta_path  = Path(root_dir) / "chunks_metadata.jsonl"

    mp4s = sorted(raw_dir.glob("*.mp4"))
    print("Chunking %d videos → %s", len(mp4s), chunks_dir)

    with open(meta_path, "a", encoding="utf-8") as meta_f:
        for mp4 in mp4s:
            vid_id    = mp4.stem
            meta_file = raw_dir / f"{vid_id}.json"
            meta      = json.loads(meta_file.read_text()) if meta_file.exists() else {}
            duration  = _ffprobe_duration(mp4)
            activities = meta.get("activities", [])

            start = 0.0
            chunk_idx = 0
            while start < duration:
                end      = min(start + window_length, duration)
                chunk_id = f"{vid_id}_chunk{chunk_idx:04d}"
                out_mp4  = chunks_dir / f"{chunk_id}.mp4"

                if not out_mp4.exists():
                    vf = f"scale=-2:{video_height}"
                    subprocess.run([
                        "ffmpeg", "-y", "-ss", str(start), "-t", str(end - start),
                        "-i", str(mp4),
                        "-vf", vf, "-r", "25",
                        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                        "-c:a", "aac", "-ar", str(audio_sr), "-ac", "2",
                        str(out_mp4),
                    ], capture_output=True)

                chunk_acts = _align_captions(activities, start, end)
                record = {
                    "chunk_id": chunk_id,
                    "video_id": vid_id,
                    "chunk_video_path": str(out_mp4),
                    "start_s":  round(start, 3),
                    "end_s":    round(end, 3),
                    "caption":  _build_caption(chunk_acts),
                    "activities":   chunk_acts,
                    "tts_segments": meta.get("tts_segments", []),
                    "content_parent_category": meta.get("content_parent_category", ""),
                    "content_fine_category":    meta.get("content_fine_category", ""),
                }
                meta_f.write(json.dumps(record) + "\n")

                start   += window_length
                chunk_idx   += 1

            mp4.unlink()
            print(f"Chunked and removed raw: {vid_id}")

    print(f"Chunking complete. Metadata: {meta_path}")
    return meta_path
