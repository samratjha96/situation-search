"""Build the index for any soccer footage (broadcast, highlights, or training).

    CLIP_VIDEO=data/clip2.mp4 python pipeline/build.py

Vision models do all the perceiving; the rest is fixed processing.
  1. frames      ffmpeg samples a thumbnail every INTERVAL_S seconds
  2. vision      team survey + action log from the video (VIDEO_MODEL)
  3. scoreboard  OCR of the top of each frame (OCR_MODEL); optional by nature:
                 footage without a scoreboard just yields no scores
  4. track       football player detector + ByteTrack on live play (local GPU/CPU)
  5. process     roster from shirts, dedup, goals, set pieces, patterns
  6. identity    zoomed crops read who did each action (VIDEO_MODEL)
  7. process     with the crop reads; matches each action to a player track
  8. track_read  reads the number along each unnumbered acting track (VIDEO_MODEL)
  9. process     with the track reads
 10. embed       embeddings of the action descriptions (EMBED_MODEL)
 11. calibrate   per-clip "no match" cutoff for free-text search
Model stages are resumable; re-running skips finished work.
"""
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg

PIPE = Path(__file__).resolve().parent


def uv(deps, script):
    cmd = ["uv", "run", "-q"]
    for d in deps:
        cmd += ["--with", d]
    subprocess.run(cmd + ["python", str(PIPE / script)], check=True)


def main():
    src = cfg.VIDEO
    if not src.exists():
        sys.exit(f"CLIP_VIDEO not found: {src}")
    cfg.CLIP_DIR.mkdir(parents=True, exist_ok=True)
    dest = cfg.CLIP_DIR / cfg.VIDEO_NAME
    if src != dest.resolve():
        shutil.copy(src, dest)

    if not any(cfg.FRAMES.glob("*.jpg")):
        print("== 1. frames ==")
        cfg.FRAMES.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(dest), "-vf",
                        f"fps=1/{cfg.INTERVAL_S},scale=960:-1", str(cfg.FRAMES / "f_%04d.jpg")], check=True)
    steps = [("2. vision", ["openai", "json-repair"], "vision.py"),
             ("3. scoreboard", ["openai"], "scoreboard.py"),
             ("4. track", None, "track.py"),
             ("5. process", [], "process.py"),
             ("6. identity", ["openai", "json-repair"], "identity.py"),
             ("7. process", [], "process.py"),
             ("8. track_read", ["openai", "json-repair"], "track_read.py"),
             ("9. process", [], "process.py"),
             ("10. embed", ["openai"], "embed.py"),
             ("11. calibrate", ["fastapi", "uvicorn", "numpy", "openai", "httpx"], "calibrate.py")]
    for title, deps, script in steps:
        print(f"== {title} ==", flush=True)
        if deps is None:  # declares its own dependencies (PEP 723)
            subprocess.run(["nice", "-n", "10", "uv", "run", "-q", str(PIPE / script)], check=True)
        else:
            uv(deps, script)
    print("\nDONE ->", cfg.OUT, "(the server picks it up on restart)")


if __name__ == "__main__":
    main()
