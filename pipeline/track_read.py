"""Read the shirt number of each acting player along his whole track.

A number is legible only when a player shows his back, which may be a second
before or after the action. For every track that pipeline/process.py matched to an
action and could not number yet, this step crops the player on up to CROPS frames
spread over the track (largest boxes first), enlarges them, and asks the vision
model to read number and shirt colour in one call. Raw reads go to
data/clips/<clip>/track_reads.json; process.py votes them per track.

    CLIP_VIDEO=data/clips/clip2/clip2.mp4 python pipeline/track_read.py
"""
import base64
import json
import subprocess
import sys
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from json_repair import repair_json
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import CLIP_DIR, OPENROUTER_API_KEY, OPENROUTER_URL, VIDEO, VIDEO_MODEL

CROPS, ZOOM, PAD, MIN_H = 6, 5, 6, 30
OUT = CLIP_DIR / "track_reads.json"
client = OpenAI(base_url=OPENROUTER_URL, api_key=OPENROUTER_API_KEY)


def crop(t, box):
    x0, y0, x1, y1 = box
    x, y, w, h = max(x0 - PAD, 0), max(y0 - PAD, 0), x1 - x0 + 2 * PAD, y1 - y0 + 2 * PAD
    return subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", f"{t:.2f}", "-i", str(VIDEO),
                           "-frames:v", "1", "-vf", f"crop={w}:{h}:{x}:{y},scale=iw*{ZOOM}:ih*{ZOOM}:flags=lanczos",
                           "-f", "image2", "-c:v", "mjpeg", "-q:v", "2", "-"], capture_output=True).stdout


def read_track(samples):
    imgs = [c for c in (crop(t, b) for t, b in samples) if c]
    if not imgs:
        return []
    r = client.chat.completions.create(model=VIDEO_MODEL, temperature=0.0, max_tokens=800, messages=[{
        "role": "user", "content": [{"type": "text", "text":
            "Enlarged crops of soccer players. For EACH crop, the shirt number and printed name you can "
            "ACTUALLY read, and the shirt colour. Return ONLY JSON {\"reads\": [{\"shirt_number\": digits or "
            "null, \"shirt_name\": text or null, \"shirt_colour\": text}]}"}] +
        [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(c).decode()}}
         for c in imgs]}])
    return json.loads(repair_json(r.choices[0].message.content)).get("reads") or []


def main():
    idx = json.loads((CLIP_DIR / "index.json").read_text())
    tracks = json.loads((CLIP_DIR / "tracks.json").read_text())
    want = {a["track"] for a in idx["actions"] if a.get("track") and a["identity_source"] in ("none", "carried")}
    boxes = defaultdict(list)  # split track ids are "<id>.<part>"; process.py assigns parts the same way
    for f in tracks["frames"]:
        for p in f["people"]:
            boxes[p["id"]].append((f["t"], p["box"]))
    done = json.loads(OUT.read_text()) if OUT.exists() else {}
    todo = [tr for tr in want if tr not in done]
    print(f"{len(todo)} tracks to read", flush=True)
    lock = threading.Lock()

    def job(tr):
        tid = tr.split(".")[0]
        span = idx["track_parts"][tr]  # the split part's time span
        cand = [(t, b) for t, b in boxes[int(tid)] if b[3] - b[1] >= MIN_H and span[0] <= t <= span[1]]
        cand.sort(key=lambda x: -(x[1][3] - x[1][1]))
        picked = []
        for t, b in cand:  # largest boxes, at least 0.4 s apart
            if all(abs(t - u) >= 0.4 for u, _ in picked):
                picked.append((t, b))
            if len(picked) == CROPS:
                break
        return read_track(picked)

    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(job, tr): tr for tr in todo}
        for f in as_completed(futs):
            tr = futs[f]
            try:
                reads = f.result()
            except Exception as e:
                print(f"[{tr}] FAILED: {e}", file=sys.stderr, flush=True)
                continue
            with lock:
                done[tr] = reads
                OUT.write_text(json.dumps(done))
            print(f"[{len(done)}] track {tr}: {sorted({str(r.get('shirt_number')) for r in reads if r.get('shirt_number')})}",
                  flush=True)


if __name__ == "__main__":
    main()
