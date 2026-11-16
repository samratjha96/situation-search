# /// script
# dependencies = ["ultralytics", "supervision==0.25.1", "numpy<2.3", "opencv-python-headless"]
# ///
"""Detect and track every player, keeper, referee and the ball in live play.

A football-trained detector (Roboflow's YOLOv8 player model, from
github.com/roboflow/sports) boxes people and the ball on every STRIDE-th frame;
ByteTrack links boxes into tracks. Tracking runs only inside the live-play windows
of vision.json and restarts at each gap, so camera cuts and replays never join two
players into one track. For each box the mean colour of the torso is stored, so
pipeline/process.py can assign teams and split tracks that jump between players.

Runs locally (Apple GPU via MPS, else CPU), capped at 4 CPU threads.
Writes data/clips/<clip>/tracks.json.

    CLIP_VIDEO=data/clips/clip2/clip2.mp4 uv run pipeline/track.py
"""
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
import torch
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import CLIP_DIR, DATA, VIDEO

DETECTOR = Path(os.environ.get("PLAYER_DETECTOR", DATA / "models" / "football-player-detection.pt"))
STRIDE = 2      # every 2nd frame (12.5 fps at 25 fps) is plenty for tracking
LIVE = ("live_wide", "live_close")
torch.set_num_threads(4)


def live_segments(vision):
    """Union of live-play windows, as [start, end] seconds."""
    segs = []
    for w in vision["windows"]:
        if w["shot_type"] not in LIVE:
            continue
        a, b = w["start"], w["start"] + vision["win_s"]
        if segs and a <= segs[-1][1]:
            segs[-1][1] = max(segs[-1][1], b)
        else:
            segs.append([a, b])
    return segs


def torso_colour(frame, box):
    """Mean BGR of the shirt area: the middle of the upper half of the box."""
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    patch = frame[int(y0 + 0.15 * h):int(y0 + 0.5 * h), int(x0 + 0.25 * w):int(x1 - 0.25 * w)]
    return [round(float(c)) for c in patch.reshape(-1, 3).mean(0)] if patch.size else None


def main():
    vision = json.loads((CLIP_DIR / "vision.json").read_text())
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = YOLO(str(DETECTOR)).to(device)
    names = model.names
    info = sv.VideoInfo.from_video_path(str(VIDEO))
    frames, t0, n = [], time.time(), 0
    for k, (a, b) in enumerate(live_segments(vision)):
        tracker = sv.ByteTrack(minimum_consecutive_frames=3, frame_rate=round(info.fps / STRIDE))
        gen = sv.get_video_frames_generator(str(VIDEO), start=int(a * info.fps), end=int(b * info.fps), stride=STRIDE)
        for i, frame in enumerate(gen):
            t = round(a + i * STRIDE / info.fps, 2)
            d = sv.Detections.from_ultralytics(model(frame, imgsz=1280, verbose=False)[0])
            ball = d[np.array([names[c] == "ball" for c in d.class_id], dtype=bool)]
            people = d[np.array([names[c] != "ball" for c in d.class_id], dtype=bool)]
            people = tracker.update_with_detections(people.with_nms(threshold=0.5, class_agnostic=True))
            frames.append({"t": t, "seg": k,
                           "ball": [round(float(x)) for x in ball.xyxy[0]] if len(ball) else None,
                           "people": [{"id": k * 100000 + int(tid), "cls": names[int(c)],
                                       "box": [round(float(x)) for x in bx], "bgr": torso_colour(frame, bx)}
                                      for bx, c, tid in zip(people.xyxy, people.class_id, people.tracker_id)]})
            n += 1
        print(f"segment {k}: {a}-{b}s, {n} frames so far, {n / (time.time() - t0):.1f} fps", flush=True)
    (CLIP_DIR / "tracks.json").write_text(json.dumps({"fps": info.fps, "stride": STRIDE,
                                                       "width": info.width, "height": info.height,
                                                       "frames": frames}))
    print(f"{n} frames -> {CLIP_DIR / 'tracks.json'}")


if __name__ == "__main__":
    main()
