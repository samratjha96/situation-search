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
VIDEO_NAME = VIDEO.name                      # what the UI plays via /media/clips/<clip>/<name>

# Everything the app serves for a clip lives in data/clips/<clip>/ (the server's layout).
CLIP_DIR = DATA / "clips" / VIDEO.stem
FRAMES = CLIP_DIR / "frames"                 # sampled frames
OUT = CLIP_DIR / "index.json"                # the index the app reads

# Working state, not served.
CHUNKS = DATA / "chunks" / VIDEO.stem         # video windows cut for the vision model

INTERVAL_S = 3                                # frame sampling interval (seconds)

# Inference endpoint (OpenRouter-style). The real URL + key live in .env, never
# in the code — set OPENROUTER_URL / OPENROUTER_API_KEY in the environment.
OPENROUTER_URL = os.environ.get("OPENROUTER_URL", "https://openrouter.ai/api/v1")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")

# Model routes also come from the environment, so no provider-specific model ids
# are committed. The defaults are OpenRouter-style placeholders — set the real
# routes for your endpoint in .env (see .env.example).
VIDEO_MODEL = os.environ.get("VIDEO_MODEL", "google/gemini-2.5-flash")    # vision: video + stills
OCR_MODEL = os.environ.get("OCR_MODEL", "google/gemini-2.5-flash")        # scoreboard OCR
