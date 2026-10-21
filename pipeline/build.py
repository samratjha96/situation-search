"""Build the index for ANY clip, end to end.

    CLIP_VIDEO=data/clip2.mp4 python pipeline/build.py

Stages (each run in its own env so deps never clash):
  1. frames   - sample a frame every INTERVAL_S seconds
  2. audio    - extract mono 16k wav
  3. asr      - local Whisper transcript (optional; skipped if it fails)
  4. caption  - per-moment visual-first captions (3-frame bursts)
  5. events   - commentary+scorebug event attribution
  6. video    - video-native event timeline (8s chunks sent as video)
Nothing here is specific to a particular match — it reads only the clip.
"""
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg

PIPE = Path(__file__).resolve().parent


def sh(*args):
    subprocess.run([str(a) for a in args], check=True)


def uv(deps, *args):
    cmd = ["uv", "run"]
    for d in deps:
        cmd += ["--with", d]
    subprocess.run(cmd + [str(a) for a in args], check=True)


def main():
    src = cfg.VIDEO
    if not src.exists():
        sys.exit(f"CLIP_VIDEO not found: {src}")
    cfg.CLIP_DIR.mkdir(parents=True, exist_ok=True)
    dest = cfg.CLIP_DIR / cfg.VIDEO_NAME   # the app serves the clip from data/clips/<clip>/
    if src != dest.resolve():
        shutil.copy(src, dest)

    # fresh working state
    for d in (cfg.FRAMES, cfg.CHUNKS, cfg.TRANSCRIPT.parent):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
    if cfg.OUT.exists():
        cfg.OUT.unlink()

    print("== 1. frames ==")
    sh("ffmpeg", "-hide_banner", "-loglevel", "error", "-i", dest,
       "-vf", f"fps=1/{cfg.INTERVAL_S},scale=960:-1", cfg.FRAMES / "f_%04d.jpg")
    print("   frames:", len(list(cfg.FRAMES.glob("*.jpg"))))

    print("== 2. audio ==")
    sh("ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", dest,
       "-vn", "-ac", "1", "-ar", "16000", cfg.AUDIO)

    print("== 3. transcript (local whisper) ==")
    try:
        uv(["mlx-whisper"], "mlx_whisper", cfg.AUDIO,
           "--model", "mlx-community/whisper-large-v3-turbo", "--language", "en",
           "--output-format", "json", "--output-dir", cfg.TRANSCRIPT.parent)
        # mlx_whisper names it <audiostem>.json -> audio.json == TRANSCRIPT
    except subprocess.CalledProcessError:
        print("   (transcript failed — continuing without commentary)")

    print("== 4. caption moments ==")
    uv(["openai", "json-repair"], "python", PIPE / "caption.py")
    print("== 5. attribute events ==")
    uv(["openai"], "python", PIPE / "events.py")
    print("== 6. video-native events ==")
    uv(["openai", "json-repair"], "python", PIPE / "video_events.py")

    print("\nDONE ->", cfg.OUT)
    print(f"Register the clip in CLIP_CONF (app/server.py) as {cfg.VIDEO.stem!r} to serve it.")


if __name__ == "__main__":
    main()
