# /// script
# dependencies = ["numpy", "onnxruntime", "pillow"]
# ///
"""Encode sampled thumbnails locally for the optional visual search endpoint.

    python pipeline/visual_index.py           # all indexed matches
    python pipeline/visual_index.py clip2     # one match
"""
import argparse
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.visual import DATA, MODEL, REVISION, session


def build(clip_dir, encoder, config):
    index = json.loads((clip_dir / "index.json").read_text())
    names = index["frames"]
    if not names:
        raise ValueError(f"No sampled frames in {clip_dir}")
    size = config["crop_size"]
    mean = np.asarray(config["image_mean"], dtype=np.float32)
    std = np.asarray(config["image_std"], dtype=np.float32)
    encoded = []
    for offset in range(0, len(names), 8):
        batch = []
        for name in names[offset:offset + 8]:
            with Image.open(clip_dir / name) as img:
                crop = ImageOps.fit(img.convert("RGB"), (size["width"], size["height"]), method=config["resample"])
                pixels = np.asarray(crop, dtype=np.float32) * config["rescale_factor"]
                batch.append(((pixels - mean) / std).transpose(2, 0, 1))
        vectors, = encoder.run(["image_embeds"], {"pixel_values": np.stack(batch)})
        encoded.append(vectors)
    vectors = np.concatenate(encoded)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or (norms == 0).any():
        raise ValueError("Visual encoder produced invalid frame vectors")
    with tempfile.NamedTemporaryFile(dir=clip_dir, suffix=".npz", delete=False) as tmp:
        np.savez(tmp, frames=np.asarray(names), revision=REVISION, vectors=vectors / norms)
    output = clip_dir / "visual.npz"
    Path(tmp.name).chmod(0o644)
    Path(tmp.name).replace(output)
    print(f"{output}: {len(vectors)} frame vectors", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("clip", nargs="?")
    args = parser.parse_args()
    indexes = sorted((DATA / "clips").glob("*/index.json"))
    if args.clip:
        indexes = [p for p in indexes if p.parent.name == args.clip]
    if not indexes:
        parser.error("No matching clip index")
    config = json.loads((MODEL / "preprocessor_config.json").read_text())
    encoder = session(MODEL / "onnx" / "vision_model_quantized.onnx")
    for p in indexes:
        build(p.parent, encoder, config)


if __name__ == "__main__":
    main()
