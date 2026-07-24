"""Video-native event detection — works on ANY footage (incl. training clips).

IMPORTANT: this gateway (OpenAI-compatible proxy) silently DROPS video sent via
video_url — Gemini then confabulates a generic "buildup->shot->goal" from priors
(verified: it invented scorelines and player names). So we send a BURST of real
sampled frames as images (which the gateway does process) and let the model read
the motion across them. A calibrated prompt lets it answer "nothing notable" and
forbids assuming a shot/goal. Identity is best-effort: jersey number or
shirt-colour + position, never a guessed name.

Adds "video_chunks" and a flattened "video_events" list to data/index.json.
Parallel, resumable, incremental.
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

ROOT = Path(__file__).resolve().parent.parent
VIDEO = ROOT / "data" / "seed.mp4"
CHUNKS = ROOT / "data" / "chunks"
OUT = ROOT / "data" / "index.json"
CHUNK_S = 8
N_FRAMES = 5  # real frames sampled across each chunk (sent as images)
MODEL = "google/gemini-2.5-flash"

PROMPT = (
    f"These are {N_FRAMES} frames sampled evenly across a ~{CHUNK_S}-second soccer clip "
    f"(frame 1 = start, frame {N_FRAMES} = end). Read the MOTION across them. "
    "Report ONLY events you can CLEARLY see in these frames. Most clips are routine "
    "possession — if nothing notable happens, return an EMPTY events list. Do NOT assume "
    "a shot or goal unless you actually see the ball struck at goal or entering the net; "
    "never invent a buildup-to-goal story. Identify players ONLY by a legible jersey "
    "NUMBER or shirt-colour + position (e.g. 'red-#9', 'blue-goalkeeper'); NEVER guess a name.\n"
    "Return ONLY JSON with keys events and summary, where events is a list of objects with "
    "keys: frame_index (integer 1-N), type (pass|shot|cross|dribble|tackle|duel|save|"
    "clearance|interception|foul|goal|corner|free_kick|throw_in|out_of_play|other), "
    "key_player, confidence_0_1."
)

client = OpenAI(base_url="https://openrouter.ai/api/v1",
                api_key=__import__("os").environ["OPENROUTER_API_KEY"])


def tc(s):
    return f"{int(s)//60:02d}:{int(s)%60:02d}"


def duration():
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "default=nk=1:nw=1", str(VIDEO)],
                         capture_output=True, text=True).stdout
    return float(out.strip())


def frame_uri(start, k):
    """Extract one frame at start + k*step and return an image data URI."""
    CHUNKS.mkdir(exist_ok=True)
    t = start + k * (CHUNK_S / (N_FRAMES - 1))
    tmp = CHUNKS / f"f_{start:04d}_{k}.jpg"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", str(t),
                    "-i", str(VIDEO), "-frames:v", "1", "-vf", "scale=854:-1", str(tmp)],
                   check=True)
    return "data:image/jpeg;base64," + base64.b64encode(tmp.read_bytes()).decode()


def analyze(start):
    imgs = [{"type": "image_url", "image_url": {"url": frame_uri(start, k)}}
            for k in range(N_FRAMES)]
    r = client.chat.completions.create(
        model=MODEL, temperature=0.0, max_tokens=6000,
        messages=[{"role": "user", "content": [{"type": "text", "text": PROMPT}] + imgs}])
    d = json.loads(repair_json(r.choices[0].message.content))
    step = CHUNK_S / (N_FRAMES - 1)
    evs = []
    for e in d.get("events", []):
        fi = e.get("frame_index")
        fi = int(fi) if isinstance(fi, (int, float)) or str(fi).isdigit() else 1
        t = start + (fi - 1) * step
        evs.append({"t_seconds": round(t, 1), "timecode": tc(t), "type": e.get("type"),
                    "key_player": e.get("key_player"), "confidence": e.get("confidence_0_1"),
                    "chunk_start": start})
    return {"start": start, "summary": d.get("summary"), "events": evs}


def main():
    starts = list(range(0, int(duration()), CHUNK_S))
    data = json.loads(OUT.read_text()) if OUT.exists() else {}
    results = {c["start"]: c for c in data.get("video_chunks", [])}
    todo = [s for s in starts if s not in results]
    print(f"{len(starts)} chunks ({CHUNK_S}s), {len(todo)} to do")
    lock = threading.Lock()

    def save():
        data["video_chunks"] = [results[k] for k in sorted(results)]
        data["video_events"] = sorted((e for c in data["video_chunks"] for e in c["events"]),
                                       key=lambda x: x["t_seconds"])
        tmp = OUT.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2))
        tmp.replace(OUT)

    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(analyze, s): s for s in todo}
        for f in as_completed(futs):
            s = futs[f]
            try:
                r = f.result()
                with lock:
                    results[s] = r
                    save()
                line = " ".join(f"{e['timecode']}:{e['type']}" for e in r["events"])
                print(f"[{len(results)}/{len(starts)}] {tc(s)}  {line}", flush=True)
            except Exception as e:
                print(f"[chunk {s}] FAILED: {e}", file=sys.stderr, flush=True)
    total = sum(len(c["events"]) for c in results.values())
    print(f"\n{total} video-native events across {len(results)} chunks -> data/index.json")


if __name__ == "__main__":
    main()
