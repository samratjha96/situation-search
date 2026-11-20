# /// script
# requires-python = ">=3.11"
# dependencies = ["torch", "torchvision", "timm==1.0.11", "numpy", "tqdm"]
# ///
"""Print the peak score for each SoccerNet action over 8 seconds of local video."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

os.environ.update(HF_HUB_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")

import numpy as np
import timm
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("assets", type=Path, help="SoccerNet model source, config, class list, and checkpoint")
    parser.add_argument("video", type=Path)
    parser.add_argument("start", type=float, nargs="+", help="Window starts in seconds")
    args = parser.parse_args()
    assets = args.assets.resolve(strict=True)
    video = args.video.resolve(strict=True)
    sys.path.insert(0, str(assets))
    from model.model import TDEEDModel
    from model.modules import FC2Layers

    torch.set_num_threads(3)
    config = SimpleNamespace(**json.loads((assets / "SoccerNetBall_baseline.json").read_text()))
    config.crop_dim = None  # upstream converts -1 to None
    factory = timm.create_model
    # The full checkpoint already contains the backbone. Skip the ImageNet download.
    with patch.object(timm, "create_model", side_effect=lambda name, **_: factory(name, pretrained=False)):
        model = TDEEDModel(device="cpu", args=config)
    # Upstream update_pred_head() hardcodes .cuda(). Use the same joint head on CPU.
    model._model._pred_fine = FC2Layers(model._model._feat_dim, [13, 18])
    model._model._double_head = True
    state = torch.load(assets / "checkpoint_best.pt", map_location="cpu", weights_only=True)
    model._model.load_state_dict(state, strict=True)
    labels = (assets / "class.txt").read_text().splitlines()

    for start in args.start:
        if start < 0:
            parser.error("window start must be nonnegative")
        raw = subprocess.check_output([
            "ffmpeg", "-v", "error", "-nostdin", "-protocol_whitelist", "file,pipe", "-ss", str(start),
            "-i", str(video), "-vf", r"fps=25,select=not(mod(n\,2)),scale=796:448",
            "-fps_mode", "passthrough", "-frames:v", "100", "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1",
        ])
        frames = np.frombuffer(raw, dtype=np.uint8).reshape(100, 448, 796, 3).copy()
        with torch.inference_mode():
            _, scores = model.predict(torch.from_numpy(frames).permute(0, 3, 1, 2), use_amp=False)
        # Sum the two team-side scores to measure each event regardless of team.
        by_action = scores[0, :, 1:].reshape(100, len(labels), 2).sum(axis=-1)
        peaks = []
        for i in np.argsort(-by_action.max(axis=0)):
            frame = int(by_action[:, i].argmax())
            peaks.append([labels[i], round(start + frame / 12.5, 2), round(float(by_action[frame, i]), 4)])
        print(json.dumps({"video": str(args.video), "start": start, "peaks": peaks}), flush=True)


if __name__ == "__main__":
    main()
