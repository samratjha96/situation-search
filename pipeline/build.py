"""Build the index for any soccer footage (broadcast, highlights, or training).

    CLIP_VIDEO=data/clip2.mp4 python pipeline/build.py

Vision models do all the perceiving; the rest is fixed processing.
  1. frames      ffmpeg samples a thumbnail every INTERVAL_S seconds
  2. vision      team survey + action log from the video (VIDEO_MODEL)
  3. scoreboard  OCR of the top of each frame (OCR_MODEL); optional by nature:
                 footage without a scoreboard just yields no scores
  4. process     roster from shirts, dedup, goals, set pieces, patterns
  5. identity    zoomed crops read who did each action (VIDEO_MODEL)
  6. process     again, now with the crop reads
  7. embed       embeddings of the action descriptions (EMBED_MODEL)
  8. calibrate   per-clip "no match" cutoff for free-text search
Stages 2, 3 and 5 are resumable; re-running skips finished work.
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
             ("4. process", [], "process.py"),
             ("5. identity", ["openai", "json-repair"], "identity.py"),
             ("6. process", [], "process.py"),
             ("7. embed", ["openai"], "embed.py"),
             ("8. calibrate", ["fastapi", "uvicorn", "numpy", "openai", "httpx"], "calibrate.py")]
    for title, deps, script in steps:
        print(f"== {title} ==", flush=True)
        uv(deps, script)
    print("\nDONE ->", cfg.OUT, "(the server picks it up on restart)")


if __name__ == "__main__":
    main()
