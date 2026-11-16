"""Read the on-screen score from every sampled frame with an OCR model.

The top TOP_BAND of each frame (where broadcasters put the scoreboard) goes to
OCR_MODEL; the text and the first "N-N" in it are saved per frame to
data/clips/<clip>/scoreboard.json. pipeline/process.py turns score increments
into goals. Replays do not change the score, so they cannot add goals.

    CLIP_VIDEO=data/clips/clip2/clip2.mp4 python pipeline/scoreboard.py
"""
import base64
import json
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import CLIP_DIR, FRAMES, INTERVAL_S, OCR_MODEL, OPENROUTER_API_KEY, OPENROUTER_URL

TOP_BAND = 0.15
client = OpenAI(base_url=OPENROUTER_URL, api_key=OPENROUTER_API_KEY)


def read(i, path):
    band = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(path), "-vf",
                           f"crop=iw:ih*{TOP_BAND}:0:0", "-f", "image2", "-c:v", "mjpeg", "-"],
                          capture_output=True, check=True).stdout
    try:
        r = client.chat.completions.create(model=OCR_MODEL, temperature=0.0, max_tokens=120, messages=[
            {"role": "user", "content": [{"type": "text", "text": "Read all text."}, {"type": "image_url",
             "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(band).decode()}}]}])
        text = r.choices[0].message.content.strip()
    except Exception as e:
        text = f"ERROR {e}"
    m = re.search(r"\b(\d{1,2})\s*[-–:]\s*(\d{1,2})\b", text)
    return {"t": i * INTERVAL_S, "text": text, "score": [int(m[1]), int(m[2])] if m else None}


def main():
    frames = sorted(FRAMES.glob("*.jpg"))
    with ThreadPoolExecutor(max_workers=12) as ex:
        reads = list(ex.map(read, range(len(frames)), frames))
    (CLIP_DIR / "scoreboard.json").write_text(json.dumps(reads, indent=1))
    print(f"{sum(1 for r in reads if r['score'])}/{len(reads)} frames with a score")


if __name__ == "__main__":
    main()
