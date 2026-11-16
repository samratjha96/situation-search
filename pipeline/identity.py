"""Read who did each action from zoomed crops of full-resolution frames.

In the broadcast video players are ~20 px tall, so the action log rarely reads a
shirt number. For each action in index.json this step:
  1. grabs 5 full-resolution frames around the action time,
  2. asks the vision model for the box of the player doing that action in each,
  3. crops each box, enlarges it 6x, and asks the vision model to read the shirt
     number and the kit colour from the crops alone.
The raw reads go to data/clips/<clip>/identity.json keyed by "t|action|team";
pipeline/process.py decides the number (readable reads with the team's kit must
agree) and turns it into a name with the vision-built roster.

    CLIP_VIDEO=data/clips/clip2/clip2.mp4 python pipeline/identity.py
"""
import base64
import json
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from json_repair import repair_json
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import CLIP_DIR, OPENROUTER_API_KEY, OPENROUTER_URL, VIDEO, VIDEO_MODEL

OFFSETS = (-1.0, -0.5, 0.0, 0.5, 1.0)  # action times are only good to about a second
ZOOM, PAD = 6, 6
OUT = CLIP_DIR / "identity.json"
client = OpenAI(base_url=OPENROUTER_URL, api_key=OPENROUTER_API_KEY)


def jpg(args):
    return subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", *args, "-f", "image2",
                           "-c:v", "mjpeg", "-q:v", "2", "-"], capture_output=True, check=True).stdout


def part(b):
    return {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(b).decode()}}


def ask(content, mt):
    r = client.chat.completions.create(model=VIDEO_MODEL, temperature=0.0, max_tokens=mt,
                                       messages=[{"role": "user", "content": content}])
    return json.loads(repair_json(r.choices[0].message.content))


def read_action(a, kits, w, h):
    frames = [jpg(["-ss", f"{max(a['t'] + o, 0):.2f}", "-i", str(VIDEO), "-frames:v", "1"]) for o in OFFSETS]
    loc = ask([{"type": "text", "text":
                f"{len(frames)} consecutive soccer frames, 0.5 s apart. Find the {a['team']} player "
                f"({kits[a['team']]}) doing this: {a['action']} - {a['detail']} "
                "Return ONLY JSON {\"boxes\": [box_2d or null per frame]} with box_2d = [ymin, xmin, ymax, xmax] "
                "normalized 0-1000; null if he is not visible in that frame."}] + [part(f) for f in frames], 800)
    crops = []
    for f, box in zip(frames, loc.get("boxes") or []):
        if not box or len(box) != 4:
            continue
        y0, x0, y1, x1 = box
        x, y = max(int(x0 * w / 1000) - PAD, 0), max(int(y0 * h / 1000) - PAD, 0)
        cw, ch = int((x1 - x0) * w / 1000) + 2 * PAD, int((y1 - y0) * h / 1000) + 2 * PAD
        if cw < 8 or ch < 8:
            continue
        crops.append(subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "image2pipe", "-i", "-",
             "-vf", f"crop={cw}:{ch}:{x}:{y},scale=iw*{ZOOM}:ih*{ZOOM}:flags=lanczos",
             "-f", "image2", "-c:v", "mjpeg", "-q:v", "2", "-"], input=f, capture_output=True).stdout)
    crops = [c for c in crops if c]
    if not crops:
        return {"reads": [], "boxes": loc.get("boxes"), "offsets": list(OFFSETS)}
    rd = ask([{"type": "text", "text":
               "Enlarged crops, each meant to show one soccer player. For EACH crop, the shirt number and "
               "printed name you can ACTUALLY read, and his shirt colour. Return ONLY JSON {\"reads\": "
               "[{\"shirt_number\": digits or null, \"shirt_name\": text or null, \"shirt_colour\": text}]}"}]
             + [part(c) for c in crops], 800).get("reads") or []
    return {"reads": rd, "boxes": loc.get("boxes"), "offsets": list(OFFSETS)}


def main():
    idx = json.loads((CLIP_DIR / "index.json").read_text())
    kits = {t["name"]: t["kit"] for t in idx["teams"]}
    w, h = map(int, subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v", "-show_entries",
                                    "stream=width,height", "-of", "csv=p=0", str(VIDEO)],
                                   capture_output=True, text=True).stdout.strip().split(","))
    done = json.loads(OUT.read_text()) if OUT.exists() else {}
    def finished(k):  # read with the current frame offsets, or already legible
        r = done.get(k)
        return r is not None and (r.get("offsets") == list(OFFSETS) or
                                  any(x.get("shirt_number") for x in r.get("reads", [])))
    todo = [a for a in idx["actions"] if a["team"] and not finished(f"{a['t']}|{a['action']}|{a['team']}")]
    print(f"{len(todo)} actions to read", flush=True)
    lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(read_action, a, kits, w, h): a for a in todo}
        for f in as_completed(futs):
            a = futs[f]
            try:
                r = f.result()
            except Exception as e:
                print(f"[{a['timecode']} {a['action']}] FAILED: {e}", file=sys.stderr, flush=True)
                continue
            with lock:
                done[f"{a['t']}|{a['action']}|{a['team']}"] = r
                OUT.with_suffix(".tmp").write_text(json.dumps(done))
                OUT.with_suffix(".tmp").replace(OUT)
            got = sorted({str(x.get("shirt_number")) for x in r["reads"] if x.get("shirt_number")})
            print(f"[{len(done)}] {a['timecode']} {a['team']} {a['action']}: {got}", flush=True)


if __name__ == "__main__":
    main()
