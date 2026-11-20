# /// script
# requires-python = ">=3.11"
# dependencies = ["transformers==5.23.0", "torch", "torchvision", "pillow", "numpy"]
# ///
"""Run TimeLens2 on local footage. Each input JSONL row has video, start, end, query.

    uv run --offline evals/local_grounding.py data/models/timelens2-2b cases.jsonl

Video paths are relative to the working directory. Output spans use the video's clock.
"""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import time

os.environ.update(HF_HUB_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1", TOKENIZERS_PARALLELISM="false")

import numpy as np
import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


def local_frames(video, start, end):
    path = Path(video).resolve(strict=True)
    if not path.is_file() or not 0 <= start < end:
        raise ValueError("Expected a local video file and a positive time range")
    raw = subprocess.check_output([
        "ffmpeg", "-v", "error", "-nostdin", "-protocol_whitelist", "file,pipe",
        "-ss", str(start), "-i", str(path), "-t", str(end - start),
        "-vf", "fps=2,scale=448:256:force_original_aspect_ratio=decrease,pad=448:256:(ow-iw)/2:(oh-ih)/2",
        "-an", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
    ])
    frames = np.frombuffer(raw, dtype=np.uint8).copy().reshape(-1, 256, 448, 3)
    if not len(frames) or len(frames) < math.floor(2 * (end - start)):
        raise ValueError("Requested interval exceeds the available footage")
    return frames


def absolute_spans(answer, start, end):
    try:
        pairs = json.loads(answer)
        if not isinstance(pairs, list):
            return None
        result = []
        for pair in pairs:
            if not isinstance(pair, list) or len(pair) != 2:
                return None
            a, b = pair
            if any(type(t) not in (int, float) for t in pair) or not 0 <= a < b <= end - start:
                return None
            result.append([round(start + a, 3), round(start + b, 3)])
        return result
    except ValueError:
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("weights", type=Path)
    parser.add_argument("cases", type=Path)
    args = parser.parse_args()
    weights = args.weights.resolve(strict=True)
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(4)
    processor = AutoProcessor.from_pretrained(weights, local_files_only=True, trust_remote_code=False)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        weights, local_files_only=True, trust_remote_code=False, dtype=torch.float32 if device == "cpu" else torch.float16,
        attn_implementation="sdpa",
    ).to(device).eval()
    with args.cases.open() as cases, torch.inference_mode():
        for line in cases:
            case = json.loads(line)
            video, start, end, query = (case[k] for k in ("video", "start", "end", "query"))
            tic = time.monotonic()
            frames = local_frames(video, start, end)
            prompt = (f'Given the query: "{query}", return ALL time spans (in seconds) where the query is relevant.\n'
                      "Output format MUST be a JSON array of [start, end] pairs. Return [] if the event is absent.")
            messages = [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": prompt}]}]
            formatted = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(
                text=[formatted], videos=[frames], do_sample_frames=False, return_tensors="pt",
                video_metadata=[{"fps": 2.0, "total_num_frames": len(frames), "frames_indices": list(range(len(frames)))}],
            ).to(device)
            ids = model.generate(**inputs, do_sample=False, max_new_tokens=100)
            answer = processor.decode(ids[0, inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
            print(json.dumps({**case, "device": device, "frames": len(frames), "raw": answer,
                              "spans": absolute_spans(answer, start, end), "seconds": round(time.monotonic()-tic, 2)}), flush=True)


if __name__ == "__main__":
    main()
