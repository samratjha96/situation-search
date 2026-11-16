"""Ball receptions with body shape — from the video, not the commentary.

Each 6 s window (stride 3 s) is cut, slowed to half speed so the model's ~1 fps
sampling sees ~2 fps of real play, and sent as a video file. The model lists every
clear reception in live play: receiver (team + legible shirt number), body shape on
the first touch, first-touch direction, pressure and zone. Names come from the
clip's squad sheet (data/clips/<clip>/lineup.json: kit + number -> name), never
from a guess or the commentator. Overlapping windows are de-duplicated.

    CLIP_VIDEO=data/clips/clip2/clip2.mp4 python pipeline/receptions.py

Adds "receptions" to data/clips/<clip>/index.json. Resumable.
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
from config import CHUNKS, CLIP_DIR, OPENROUTER_API_KEY, OPENROUTER_URL, OUT, VIDEO, VIDEO_MODEL

WIN_S, STRIDE_S, SLOW = 6, 3, 2.0
BODY = ("half_turn", "open", "back_to_goal", "side_on", "unclear")
_LP = CLIP_DIR / "lineup.json"  # optional: without it receivers stay "team #number"
LINEUP = json.loads(_LP.read_text()) if _LP.exists() else {"teams": []}
TEAMS = ("; ".join(f"{t['name']} wear {t['kit']} (goalkeeper {t['goalkeeper_kit']})"
                   for t in LINEUP["teams"])
         or "unknown; name each team by the scoreboard if shown, else by shirt colour")

PROMPT = f"""This is {WIN_S} seconds of a soccer broadcast slowed to {1 / SLOW:g}x speed
(so it plays for {WIN_S * SLOW:g} s). Teams: {TEAMS}.
First classify shot_type: wide_live | close_up | replay | stoppage. If it is not
wide_live, return an empty receptions list.
List EVERY clear moment in live play where a player receives the ball (first touch of
a pass, or controls a loose ball). For each:
- t_offset_s: seconds into the SLOWED video when the first touch happens
- team: exact team name above, from shirt colour
- number: the shirt number ONLY if you can actually read it, else null
- body_shape on the first touch, relative to the opponent's goal:
  half_turn = side-on, hips angled so he can play forward AND back (receives on the back foot facing diagonally);
  open = already facing the opponent's goal; back_to_goal = facing his own goal;
  side_on = facing the touchline; unclear = cannot tell
- first_touch: forward | backward | sideways | none (one-touch pass) | unclear
- pressure: none | light | tight (opponent within ~2 m)
- zone: def_third | mid_third | att_third, and side: left | center | right
- detail: one factual sentence of what you see
- confidence_0_1 for the body_shape judgement
Do not invent receptions. Return ONLY JSON: {{"shot_type": ..., "receptions": [...]}}"""

client = OpenAI(base_url=OPENROUTER_URL, api_key=OPENROUTER_API_KEY)


def tc(s):
    return f"{int(s) // 60:02d}:{int(s) % 60:02d}"


def duration():
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "default=nk=1:nw=1", str(VIDEO)], capture_output=True, text=True)
    return float(out.stdout.strip())


def window_uri(start):
    CHUNKS.mkdir(parents=True, exist_ok=True)
    tmp = CHUNKS / f"r_{start:04d}.mp4"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", str(start),
                    "-t", str(WIN_S), "-i", str(VIDEO), "-an",
                    "-vf", f"setpts={SLOW}*PTS,scale=1280:-2", "-c:v", "libx264",
                    "-preset", "veryfast", "-crf", "26", str(tmp)], check=True)
    return "data:video/mp4;base64," + base64.b64encode(tmp.read_bytes()).decode()


def player_name(team, number):
    for t in LINEUP["teams"]:
        if t["name"] == team and number is not None:
            return t["numbers"].get(str(number).lstrip("#"))
    return None


def analyze(start):
    clip = {"type": "file", "file": {"file_data": window_uri(start), "format": "video/mp4"}}
    r = client.chat.completions.create(
        model=VIDEO_MODEL, temperature=0.0, max_tokens=8000,
        messages=[{"role": "user", "content": [{"type": "text", "text": PROMPT}, clip]}])
    d = json.loads(repair_json(r.choices[0].message.content))
    out = []
    if (d.get("shot_type") or "").lower() == "wide_live":
        for e in d.get("receptions") or []:
            try:
                off = min(max(float(e.get("t_offset_s")) / SLOW, 0.0), WIN_S)
            except (TypeError, ValueError):
                continue
            t = round(start + off, 1)
            num = e.get("number")
            out.append({"t_seconds": t, "timecode": tc(t), "team": e.get("team"),
                        "number": None if num in (None, "", "null") else str(num).lstrip("#"),
                        "player": player_name(e.get("team"), num),
                        "body_shape": e.get("body_shape") if e.get("body_shape") in BODY else "unclear",
                        "first_touch": e.get("first_touch"), "pressure": e.get("pressure"),
                        "zone": e.get("zone"), "side": e.get("side"), "detail": e.get("detail"),
                        "confidence": e.get("confidence_0_1"), "window": start})
    return {"start": start, "shot_type": d.get("shot_type"), "receptions": out}


def dedup(recs):
    """Overlapping windows see each reception twice: keep the more confident one."""
    kept = []
    for r in sorted(recs, key=lambda x: -(x["confidence"] or 0)):
        if not any(abs(r["t_seconds"] - k["t_seconds"]) <= 1.5 and r["team"] == k["team"]
                   and (r["number"] == k["number"] or None in (r["number"], k["number"]))
                   for k in kept):
            kept.append(r)
    return sorted(kept, key=lambda x: x["t_seconds"])


def main():
    data = json.loads(OUT.read_text())
    done = {w["start"]: w for w in data.get("reception_windows", [])}
    starts = [s for s in range(0, int(duration()) - 1, STRIDE_S) if s not in done]
    print(f"{len(starts)} windows to do with {VIDEO_MODEL}", flush=True)
    lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(analyze, s): s for s in starts}
        for f in as_completed(futs):
            try:
                w = f.result()
            except Exception as e:
                print(f"[{tc(futs[f])}] FAILED: {e}", file=sys.stderr, flush=True)
                continue
            with lock:
                done[w["start"]] = w
                data["reception_windows"] = [done[k] for k in sorted(done)]
                data["receptions"] = dedup([r for x in data["reception_windows"] for r in x["receptions"]])
                OUT.with_suffix(".tmp").write_text(json.dumps(data))
                OUT.with_suffix(".tmp").replace(OUT)
            print(f"[{len(done)}] {tc(w['start'])} {w['shot_type']}: " + ", ".join(
                f"{r['timecode']} {r['player'] or r['team']} {r['body_shape']}" for r in w["receptions"]), flush=True)
    print(f"{len(data['receptions'])} receptions -> {OUT}")


if __name__ == "__main__":
    main()
