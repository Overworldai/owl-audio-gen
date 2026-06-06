import json
import os
import subprocess
import threading
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import traceback

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
    for idx, sample in tqdm(enumerate(filtered_ds), desc="Downloading Finevideo", unit="videos"):
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

def get_hw_encoder() -> str | None:
    """Detect available hardware encoder, return None if only software available."""
    hw_encoders = [
        ("h264_nvenc", "Nvidia"),
        ("h264_videotoolbox", "Apple"),
        ("h264_qsv", "Intel"),
    ]
    for encoder, name in hw_encoders:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-f", "lavfi", "-i", "nullsrc",
             "-t", "0.1", "-c:v", encoder, "-f", "null", "-"],
            capture_output=True
        )
        if result.returncode == 0:
            print(f"[encoder] Using hardware encoder: {name} ({encoder})")
            return encoder
    print("[encoder] No hardware encoder found, falling back to libx264")
    return None

def get_hw_encoder() -> str | None:
    """Detect available hardware encoder, return None if only software available."""
    hw_encoders = [
        ("h264_nvenc", "Nvidia"),
        ("h264_videotoolbox", "Apple"),
        ("h264_qsv", "Intel"),
    ]
    for encoder, name in hw_encoders:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-f", "lavfi", "-i", "nullsrc",
             "-t", "0.1", "-c:v", encoder, "-f", "null", "-"],
            capture_output=True
        )
        if result.returncode == 0:
            print(f"[encoder] Hardware encoder available: {name} ({encoder})")
            return encoder
    print("[encoder] No hardware encoder found, using libx264")
    return None

def probe_duration(path: Path) -> float:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True
    )
    return float(r.stdout.strip())

def chunk_single_video(
    mp4: Path,
    chunks_dir: Path,
    window_length: float,
    sample_rate: int,
    video_size: tuple,
    video_fps: int,
) -> list[dict]:
    """Chunk a single video file with ffmpeg and return a list of metadata records."""
    video_id = mp4.stem
    meta_file = Path(f"{mp4.parent.parent}/metadata/{video_id}.json")

    with open(meta_file, "r") as f:
        metadata = json.load(f)

    category = {
        "parent": metadata["content_parent_category"],
        "fine": metadata["content_fine_category"],
    }

    os.makedirs(chunks_dir, exist_ok=True)
    video_window_frames = int(window_length * video_fps)

    vf = (
        f"fps={video_fps},"
        f"scale={video_size[0]}:{video_size[1]}:force_original_aspect_ratio=increase,"
        f"crop={video_size[0]}:{video_size[1]}"
    )

    # always use libx264 — nvenc cannot control GOP size on this machine
    encoder_args = [
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "28",
        "-g", str(video_window_frames),
        "-keyint_min", str(video_window_frames),
        "-sc_threshold", "0",
        "-x264-params", "open_gop=0",
    ]

    out_pattern = str(chunks_dir / f"{video_id}_chunk%04d.mp4")

    result = subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", str(mp4),
            "-vf", vf,
            "-r", str(video_fps),
            "-fps_mode", "cfr",
            "-pix_fmt", "yuv420p",
            "-video_track_timescale", str(video_fps * 1000),
            *encoder_args,
            "-c:a", "aac", "-ar", str(sample_rate), "-ac", "2",
            "-f", "segment",
            "-segment_time", str(window_length),
            "-reset_timestamps", "1",
            "-segment_start_number", "0",
            out_pattern,
        ],
        capture_output=True, timeout=7200,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for {mp4}:\n{result.stderr.decode()}")

    # chunk_files = sorted(chunks_dir.glob(f"{video_id}_chunk*.mp4"))
    # verify_chunk_durations(chunk_files, window_length)

    mp4.unlink()
    meta_file.unlink()

    # build metadata records for all the chunks 
    chunk_files = sorted([f for f in chunks_dir.glob(f"{video_id}_chunk*.mp4")])
    records = []
    for i, chunk_path in enumerate(chunk_files):
        chunk_idx = int(chunk_path.stem.split("chunk")[-1])
        start = chunk_idx * window_length
       
        is_last = i == len(chunk_files) - 1
        if is_last:
            chunk_dur = probe_duration(chunk_path)
            end = start + chunk_dur
        else:
            end = start + window_length

        records.append({
            "chunk_id": chunk_path.stem,
            "video_id": video_id,
            "chunk_video_path": str(chunk_path),
            "t_start": round(start, 3),
            "t_end": round(end, 3),
            "caption": create_chunk_caption(metadata, start, end),
            'category': category
        })

    return records
 
def chunk_finevideo(root_dir="./finevideo_data", window_length=10.0, 
                    sample_rate=44_100, video_size=(640, 320), 
                    video_fps=30, max_workers=32):
    """Split source videos into fixed-length chunks and save into JSONL metadata."""
    video_dir = Path(f"{root_dir}/videos") 
    meta_path = Path(f"{root_dir}/metadata.jsonl")
    max_workers = min(os.cpu_count(), 64) # limit to 64 workers

    chunks_dir = Path(f"{root_dir}/chunks") 
    chunks_dir.mkdir(parents=True, exist_ok=True)

    mp4s = sorted(video_dir.glob("*.mp4"))
    print(f"Chunking {len(mp4s)} videos → {chunks_dir}")

    # guards mp4.unlink() and metadata file writes
    write_lock = threading.Lock()   
    HW_ENCODER = get_hw_encoder()

    with (ThreadPoolExecutor(max_workers=max_workers) as pool,
        open(meta_path, "w", encoding="utf-8") as meta_f):

        future_to_mp4 = {
            pool.submit(
                chunk_single_video, mp4, chunks_dir, 
                window_length, sample_rate, 
                video_size, video_fps
            ) for mp4 in mp4s
        }
 
        for future in tqdm(as_completed(future_to_mp4), total=len(future_to_mp4), 
                           desc="Chunking videos", unit="video"):
            try:
                records = future.result()
                with write_lock:
                    for rec in records:
                        meta_f.write(json.dumps(rec) + "\n")
                        meta_f.flush()
            except Exception as e:
                print(f"[error] {e}")
                print(traceback.format_exc()) 
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
    
    # perform sanity check at last
    sanity_check(source=cfg.train.data_kwargs.source)


def sanity_check(source):
    """Verify that number of chunk mp4 files matches number of records in metadata.jsonl.
    Returns True if match, False otherwise and prints a short report.
    """
    chunks_dir = Path(f"{source}chunks")
    meta_path = Path(f"{source}/metadata.jsonl")

    if not chunks_dir.exists():
        print(f"chunks dir missing: {chunks_dir}")
        return False
    if not meta_path.exists():
        print(f"metadata file missing: {meta_path}")
        return False

    mp4s = sorted(chunks_dir.glob("*.mp4"))
    n_mp4 = len(mp4s)

    # count non-empty JSON lines
    n_meta = 0
    with open(meta_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                n_meta += 1

    print(f"chunks: {n_mp4}, metadata records: {n_meta}")
    if n_mp4 != n_meta:
        print("Mismatch: counts differ")
        return False
    print("OK: counts match")
    return True

def verify_chunk_durations(
    chunk_files: list[Path],
    window_length: float,
    tolerance: float = 0.1,  # 100ms
) -> None:
    """Probe all chunk durations and warn on any that are off."""
    issues = []
    for chunk in chunk_files:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(chunk)],
            capture_output=True, text=True
        )
        try:
            dur = float(r.stdout.strip())
        except ValueError:
            issues.append(f"  {chunk.name}: could not probe duration")
            continue

        expected = window_length
        delta = abs(dur - expected)
        status = "✓" if delta <= tolerance else "✗"
        if delta > tolerance:
            issues.append(f"  {chunk.name}: {dur:.6f}s (expected {expected:.3f}s, delta {delta:.6f}s)")
        else:
            print(f"  [{status}] {chunk.name}: {dur:.6f}s")

    if issues:
        print(f"\n[verify] WARNING — {len(issues)} chunk(s) with duration issues:")
        for issue in issues:
            print(issue)
    else:
        print(f"\n[verify] All {len(chunk_files)} chunks within {tolerance*1000:.0f}ms tolerance ✓")


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
        video_size=cfg.train.data_kwargs.video_size,
        video_fps=cfg.train.video_fps,
    )

    sanity_check(cfg.train.data_kwargs.source)