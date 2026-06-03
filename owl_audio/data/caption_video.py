import os, glob, argparse, json
from pathlib import Path

import numpy as np
import time
import av
import torch
import torch.multiprocessing as mp
from tqdm import tqdm

from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

from owl_audio.configs import Config


CAPTION_PROMPT = (
    "Listen to this audio clip from a video game. "
    "Describe every sound event you hear: what produces the sound, "
    "its intensity (soft / medium / loud), approximate timing within the clip, "
    "and whether it is foreground or background. "
    "Focus only on sounds, not visuals."
)

def load_qwen_audio(model_id="Qwen/Qwen2-Audio-7B-Instruct", device="cuda"):
    processor = AutoProcessor.from_pretrained(
        model_id,
        trust_remote_code=True,
    )
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    ).eval()
    return processor, model


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
    filtered = [x for x in paths if x.lower().endswith(".mp4") and "_" not in Path(x).stem]
    mp4s = [Path(x) for x in sorted(set(filtered))]
    print(f"Found {len(mp4s)} mp4s under {source}")
    return mp4s


def iter_audio_windows(mp4_path, window_length, target_sr):
    samples_per_window = int(window_length * target_sr)

    resampler = av.AudioResampler(
        format="fltp",
        layout="mono",
        rate=target_sr,
    )

    buffer = np.empty(0, dtype=np.float32)

    with av.open(mp4_path) as container:
        a_stream = next((s for s in container.streams if s.type == "audio"), None)
        if a_stream is None:
            raise ValueError(f"No audio stream in {mp4_path}")

        for packet in container.demux(a_stream):
            for frame in packet.decode():
                for rf in resampler.resample(frame):
                    arr = rf.to_ndarray()
                    if arr.ndim == 2:
                        arr = arr.mean(axis=0)
                    arr = arr.astype(np.float32)
                    buffer = np.concatenate([buffer, arr])

                    while len(buffer) >= samples_per_window:
                        waveform = buffer[:samples_per_window]
                        peak = np.abs(waveform).max()
                        if peak > 1e-6:
                            waveform = waveform / peak
                        yield waveform
                        buffer = buffer[samples_per_window:]

        for rf in resampler.resample(None):
            arr = rf.to_ndarray()
            if arr.ndim == 2:
                arr = arr.mean(axis=0)
            arr = arr.astype(np.float32)
            buffer = np.concatenate([buffer, arr])

    while len(buffer) >= samples_per_window:
        waveform = buffer[:samples_per_window]
        peak = np.abs(waveform).max()
        if peak > 1e-6:
            waveform = waveform / peak
        yield waveform
        buffer = buffer[samples_per_window:]


@torch.no_grad()
def _caption_batch(
    waveforms,
    sample_rate,
    processor,
    model,
    device,
    max_new_tokens=256,
):
    conversations = [
        [{
            "role": "user",
            "content": [
                {"type": "audio"},
                {"type": "text", "text": CAPTION_PROMPT},
            ],
        }]
        for _ in waveforms
    ]

    texts = [
        processor.apply_chat_template(
            conv,
            add_generation_prompt=True,
            tokenize=False,
        )
        for conv in conversations
    ]

    inputs = processor(
        text=texts,
        audio=waveforms,
        sampling_rate=sample_rate,
        padding=True,
        return_tensors="pt",
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )

    generated_ids = outputs[:, inputs["input_ids"].shape[1]:]

    captions = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return [c.strip() for c in captions]

def _worker(rank, world_size, cfg, batch_size=48, max_new_tokens=64):
    device = f"cuda:{rank}"
    torch.cuda.set_device(rank)

    sample_rate = 16000  # Qwen2Audio @ 16 kHz
    window_length = cfg.train.data_kwargs.window_length
    encoded_dir = Path(cfg.train.data_kwargs.encoded)
    partial_path = encoded_dir / f"captions_rank{rank}.json"

    mp4s = _find_mp4s(cfg.train.data_kwargs.source)

    # Each rank processes its own slice of mp4s
    shard_mp4s = mp4s[rank::world_size]
    print(
        f"[rank {rank}] Shard: {len(shard_mp4s)} mp4s "
        f"(total={len(mp4s)}, world={world_size})",
        flush=True,
    )

    processor, model = load_qwen_audio(device=device)

    results = []
    pbar = tqdm(desc=f"[rank {rank}] windows captioned", unit="win", dynamic_ncols=True)

    for mp4_idx, mp4 in enumerate(shard_mp4s):
        pbar.set_postfix(mp4=mp4.name, refresh=False)

        batch_waveforms = []
        batch_entries = []

        for win_idx, waveform in enumerate(
            iter_audio_windows(str(mp4), window_length, sample_rate)
        ):
            batch_waveforms.append(waveform)
            batch_entries.append({
                "mp4": str(mp4),
                "mp4_idx": mp4_idx + rank * len(shard_mp4s),  # globally unique
                "win_idx": win_idx,
                "start_time": win_idx * window_length,
                "end_time": (win_idx + 1) * window_length,
            })

            if len(batch_waveforms) == batch_size:
                captions = _caption_batch(
                    batch_waveforms, sample_rate, processor, model, device, max_new_tokens
                )
                for entry, caption in zip(batch_entries, captions):
                    results.append({**entry, "caption": caption})
                pbar.update(len(batch_waveforms))
                batch_waveforms.clear()
                batch_entries.clear()

        if batch_waveforms:
            captions = _caption_batch(
                batch_waveforms, sample_rate, processor, model, device, max_new_tokens
            )
            for entry, caption in zip(batch_entries, captions):
                results.append({**entry, "caption": caption})
            pbar.update(len(batch_waveforms))
            batch_waveforms.clear()
            batch_entries.clear()

    pbar.close()

    with open(partial_path, "w") as f:
        json.dump(results, f, indent=2)

    torch.cuda.empty_cache()

    if rank == 0:
        print("Rank 0 waiting for all partial files ...", flush=True)
        for r in range(world_size):
            p = encoded_dir / f"captions_rank{r}.json"
            while not p.exists():
                time.sleep(2)

        all_results = []
        for r in range(world_size):
            p = encoded_dir / f"captions_rank{r}.json"
            with open(p) as f:
                all_results.extend(json.load(f))
            p.unlink()

        all_results.sort(key=lambda x: (x["mp4"], x["win_idx"]))

        out_path = encoded_dir / "captions.json"
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"Merged {len(all_results)} captions -> {out_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config_path)
    world_size = torch.cuda.device_count()

    if world_size > 1:
        mp.spawn(_worker, args=(world_size, cfg), nprocs=world_size)
    else:
        _worker(rank=0, world_size=1, cfg=cfg)