"""Video-native event detection — works on ANY footage (incl. training clips).

Chunks are 8 s long and start every 4 s, so each moment is seen by two windows
and an event on a chunk boundary (a shot whose goal lands in the next chunk) is
whole in one of them. Each chunk is cut to a short mp4 and sent as the video
itself, so the model sees real motion. It must go as a `file` content part: the endpoint silently DROPS
video sent via video_url, and the model then confabulates a generic
"buildup->shot->goal" from priors (verified: it invented scorelines and player
names). A calibrated prompt lets it answer "nothing notable" and forbids
assuming a shot/goal. Identity is best-effort: jersey number or shirt-colour +
position, never a guessed name.

Adds "video_chunks" and a flattened "video_events" list to data/clips/<clip>/index.json.
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import CHUNKS, OPENROUTER_API_KEY, OPENROUTER_URL, OUT, VIDEO, VIDEO_MODEL

CHUNK_S = 8
STRIDE_S = 4  # overlapping windows; the server de-dups the double detections

PROMPT = (
    f"This is a ~{CHUNK_S}-second soccer clip. Describe what you LITERALLY see; do not infer.\n"
    "First classify shot_type: wide_live | close_up | replay | celebration | stoppage. "
    "If it is close_up / replay / celebration / stoppage, return an EMPTY events list "
    "(these are not live play — do NOT infer a goal from a celebration).\n"
    "Call it a 'goal' ONLY if you clearly SEE the ball fully cross the goal line into the "
    "net during live play. A shot that is saved, blocked, or misses is a 'shot', not a "
    "'goal'. Identify players ONLY by a legible jersey NUMBER or shirt-colour + position "
    "(e.g. 'red-#9', 'blue-goalkeeper'); NEVER guess a real name.\n"
    "For each clearly-visible on-ball event give a RICH factual detail: which players "
    "(by number/colour) do what, and where the ball goes.\n"
    "Return ONLY JSON with keys: shot_type, summary, and events (list of objects with keys: "
    f"t_offset_s (seconds from clip start, 0-{CHUNK_S}), type (pass|shot|goal|cross|dribble|"
    "tackle|duel|save|clearance|interception|foul|corner|free_kick|throw_in|other), key_player, "
    "detail (one grounded sentence), confidence_0_1)."
)


client = OpenAI(base_url=OPENROUTER_URL, api_key=OPENROUTER_API_KEY)


def tc(s):
    return f"{int(s)//60:02d}:{int(s)%60:02d}"


def duration():
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "default=nk=1:nw=1", str(VIDEO)],
                         capture_output=True, text=True).stdout
    return float(out.strip())


def chunk_uri(start):
    """Cut [start, start+CHUNK_S) to a small silent mp4 and return a video data URI."""
    CHUNKS.mkdir(parents=True, exist_ok=True)
    tmp = CHUNKS / f"c_{start:04d}.mp4"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", str(start),
                    "-t", str(CHUNK_S), "-i", str(VIDEO), "-an", "-vf", "scale=854:-2",
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "28", str(tmp)],
                   check=True)
    return "data:video/mp4;base64," + base64.b64encode(tmp.read_bytes()).decode()


def analyze(start):
    clip = {"type": "file", "file": {"file_data": chunk_uri(start), "format": "video/mp4"}}
    r = client.chat.completions.create(
        model=VIDEO_MODEL, temperature=0.0, max_tokens=6000,
        messages=[{"role": "user", "content": [{"type": "text", "text": PROMPT}, clip]}])
    d = json.loads(repair_json(r.choices[0].message.content))
    shot_type = (d.get("shot_type") or "wide_live").lower()
    evs = []
    # only trust events from live play; replays/celebrations/close-ups are flagged, not mined
    if shot_type in ("wide_live", "live"):
        for e in d.get("events", []):
            try:
                off = min(max(float(e.get("t_offset_s")), 0.0), CHUNK_S)
            except (TypeError, ValueError):
                off = 0.0
            t = start + off
            evs.append({"t_seconds": round(t, 1), "timecode": tc(t), "type": e.get("type"),
                        "key_player": e.get("key_player"), "detail": e.get("detail"),
                        "confidence": e.get("confidence_0_1"), "chunk_start": start})
    return {"start": start, "shot_type": shot_type, "scoreboard": d.get("scoreboard"),
            "summary": d.get("summary"), "events": evs}


def main():
    starts = list(range(0, int(duration()), STRIDE_S))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(OUT.read_text()) if OUT.exists() else {}
    results = {c["start"]: c for c in data.get("video_chunks", [])}
    todo = [s for s in starts if s not in results]
    print(f"{len(starts)} chunks ({CHUNK_S}s every {STRIDE_S}s), {len(todo)} to do")
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
    if not results:
        sys.exit("No chunks analysed — check OPENROUTER_URL / OPENROUTER_API_KEY and VIDEO_MODEL.")
    total = sum(len(c["events"]) for c in results.values())
    print(f"\n{total} video-native events across {len(results)} chunks -> {OUT}")


if __name__ == "__main__":
    main()
