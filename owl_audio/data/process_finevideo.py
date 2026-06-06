import json
import os
import subprocess
import threading
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm
from datasets import load_dataset
from ..configs import Config


def download_finevideo(categories=[], root_dir="./finevideo_data", max_videos=None, hf_token=None):
    token = hf_token or os.environ.get("HF_TOKEN")
    categories = [c.lower() for c in categories]

    def is_desired_category(sample):
        if not categories: return True
        parent = sample['json']['content_parent_category'].lower()
        fine = sample['json']['content_fine_category'].lower()
        return any(
            c in parent or c in fine
            for c in categories
        )
    
    ds = load_dataset("HuggingFaceFV/finevideo", split="train", streaming=True, token=token)
    filtered_ds = filter(is_desired_category, ds)

    os.makedirs(f"{root_dir}/videos", exist_ok=True)
    os.makedirs(f"{root_dir}/metadata", exist_ok=True)
    
    count = 0
    for idx, sample in tqdm(enumerate(filtered_ds), desc="Downloading Finevideo", unit="video"):
        if max_videos and count >= max_videos:
            break

        video_filename = f"{root_dir}/videos/sample_{idx:06d}.mp4"
        with open(video_filename, 'wb') as video_file:
            video_file.write(sample['mp4'])

        json_filename = f"{root_dir}/metadata/sample_{idx:06d}.json"
        with open(json_filename, 'w') as json_file:
          json.dump(sample['json'], json_file)

        count += 1
    print(f"Download complete: {count} videos in {root_dir}/videos")

def parse_time(ts: str) -> float:
    """Convert 'HH:MM:SS.mmm' or 'MM:SS.mmm' to seconds."""
    parts = ts.strip().split(':')
    if len(parts) == 3:
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + float(s)
    elif len(parts) == 2:
        m, s = parts
        return int(m) * 60 + float(s)
    return float(parts[0])

def create_chunk_caption(video_metadata, t_start, t_end, 
                         min_overlap_ratio=0.4, audioVisualCorrelation=0.5):
    tts_timed = video_metadata['timecoded_text_to_speech']
    video_scenes = video_metadata['content_metadata']['scenes']

    # video description
    video_discription_comp = video_metadata['content_metadata']['description']

    # add visual description
    activities_comp = ""
    for scene in video_scenes:
        activities = scene['activities']
        av_corr = scene['audioVisualCorrelation']
        # only keep scenes with high audio-visual correlation
        if av_corr < audioVisualCorrelation:
            continue
        for activity in activities:
            start = parse_time(activity['timestamp']['start_timestamp'])
            end = parse_time(activity['timestamp']['end_timestamp'])

            # add if activity lies entirely in winow
            if start >= t_start and end <= t_end:
                activities_comp += activity['description']
                continue

            t_overlap = max(0, min(t_end, end) - max(t_start, start))
            if t_overlap/(t_end-t_start) < min_overlap_ratio:
                continue
            activities_comp += activity['description']

    # add speech
    speech_comp = ""
    for entry in tts_timed:
        start = parse_time(entry['start'])
        end = parse_time(entry['end'])

        # add if tts lies entirely in winow
        if start >= t_start and end <= t_end:
            speech_comp += entry['text']
            continue

        t_overlap = max(0, min(t_end, end) - max(t_start, start))
        if t_overlap/(t_end-t_start) >= min_overlap_ratio:
            speech_comp += entry['text']

    caption = (
        f"{video_discription_comp}\n"
        f"Speaker says, <S>{speech_comp}<E>.\n"
        f"<VIDCAP>{activities_comp}<VIDAUDCAP>"
    )
    return caption

def chunk_single_video(mp4: Path, chunks_dir: Path, window_length: float,
                     sample_rate: int, video_size: tuple, video_fps: int,
                     write_lock: threading.Lock) -> list[dict]:
    """Chunk a single video file with ffmpeg and return a list of metadata records."""
    video_id = mp4.stem
    meta_file = f"{mp4.parent.parent}/metadata/{video_id}.json"

    with open(meta_file, 'r') as f:
        metadata = json.load(f)

    duration = metadata['duration_seconds']
    category = {
        'parent': metadata['content_parent_category'],
        'fine': metadata['content_fine_category']
    }
    
    records = []
    start = 0.0
    chunk_idx = 0
    
    # drop last chunk if dur < window_length
    while start + window_length <= duration:
        end = min(start + window_length, duration)
        chunk_id = f"{video_id}_chunk{chunk_idx:04d}"
        out_mp4 = Path(f"{chunks_dir}/{chunk_id}.mp4") 
 
        if not out_mp4.exists():
            vf = f"scale={video_size[0]}:{video_size[1]}"
            result = subprocess.run(
                [
                    "ffmpeg", "-y", "-i", str(mp4), "-ss", str(start), "-t", str(end - start),
                    "-i", str(mp4),
                    "-vf", vf, "-r", str(video_fps),
                    "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                    "-c:a", "aac", "-ar", str(sample_rate), "-ac", "2",
                    str(out_mp4),
                ],
                capture_output=True,
                timeout=600
            )

            if result.returncode != 0:
                raise RuntimeError(f"ffmpeg failed for {mp4}:\n{result.stderr.decode()}")
 
        records.append({
            "chunk_id": chunk_id,
            "video_id": video_id,
            "chunk_video_path": str(out_mp4),
            "t_start": round(start, 3),
            "t_end": round(end, 3),
            "caption": create_chunk_caption(metadata, start, end),
            'category': category
        })
 
        start += window_length
        chunk_idx += 1
    
    # delete raw mp4 file
    with write_lock:
        mp4.unlink()

    return records
 
 
def chunk_finevideo(root_dir="./finevideo_data", window_length=10.0, 
                    sample_rate=44_100, video_size=(640, 320), 
                    video_fps=30, max_workers=8):
    """Split source videos into fixed-length chunks and save into JSONL metadata."""
    video_dir = Path(f"{root_dir}/videos") 
    meta_path = Path(f"{root_dir}/metadata.jsonl")

    chunks_dir = Path(f"{root_dir}/chunks") 
    chunks_dir.mkdir(parents=True, exist_ok=True)

    mp4s = sorted(video_dir.glob("*.mp4"))[:2]
    print(f"Chunking {len(mp4s)} videos → {chunks_dir}")

    print(mp4s)

    # guards mp4.unlink() and metadata file writes
    write_lock = threading.Lock()   

    with (ThreadPoolExecutor(max_workers=max_workers) as pool,
        open(meta_path, "a", encoding="utf-8") as meta_f):

        future_to_mp4 = {
            pool.submit(chunk_single_video, mp4, chunks_dir, 
                        window_length, sample_rate, 
                        video_size, video_fps, write_lock)
            for mp4 in mp4s
        }
 
        for future in tqdm(as_completed(future_to_mp4), total=len(future_to_mp4), desc="Chunking videos", unit="video"):
            try:
                records = future.result()
                with write_lock:
                    for rec in records:
                        # Save the metadata as jsonl
                        meta_f.write(json.dumps(rec) + "\n")
            except Exception as e:
                print(e)
                continue 

    print(f"Chunking complete. Metadata: {meta_path}")
    return meta_path

def process_finevideo(
    categories,
    source,
    window_length=10.0,
    sample_rate=44_100,
    video_size=(640, 320),
    video_fps=30.0,
    max_videos=None,
    hf_token=None,
    skip_download=False,
    skip_chunking=False,
):
    if not skip_download:
        download_finevideo(categories, source, max_videos, hf_token)

    if not skip_chunking:
        return chunk_finevideo(source, window_length, sample_rate, video_size, video_fps)
    return Path(source) / "chunks_metadata.json"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--config_path", type=str, help="Path to config YAML file")
    args = parser.parse_args()
    cfg = Config.from_yaml(args.config_path)

    process_finevideo(
        categories=[],
        source=cfg.train.data_kwargs.source,
        window_length=cfg.train.data_kwargs.window_length,
        sample_rate=cfg.train.data_kwargs.sample_rate,
        video_size=cfg.train.video_size,
        video_fps=cfg.train.video_fps,
        max_videos=1000,
        skip_download=True
    )
