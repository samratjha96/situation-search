"""Single source of truth for the pipeline's paths and settings.

The video to process is CLIP_VIDEO (env var), defaulting to data/seed.mp4.
Everything else derives from it, so the whole pipeline runs on ANY clip:

    CLIP_VIDEO=data/clip2.mp4 python pipeline/build.py
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

VIDEO = Path(os.environ.get("CLIP_VIDEO", DATA / "seed.mp4")).resolve()
VIDEO_NAME = VIDEO.name                      # what the UI plays via /media/<name>

FRAMES = DATA / "frames"                     # sampled frames
AUDIO = DATA / "audio.wav"                   # extracted commentary audio
TRANSCRIPT = DATA / "transcript" / "audio.json"   # whisper transcript
CHUNKS = DATA / "chunks"                      # video-event chunk frames
OUT = DATA / "index.json"                     # the index the app reads

INTERVAL_S = 3                                # frame sampling interval (seconds)
