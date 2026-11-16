"""Vision pass: the only step that looks at the match. Everything after it is
deterministic processing of what this step records (pipeline/process.py).

1. Team survey. A few frames spread over the clip go to the vision model, which
   names both sides (from the scoreboard if there is one, else by kit or bib
   colour) and describes each kit.
2. Action log. The clip is cut into WIN_S-second windows every STRIDE_S seconds,
   slowed to 1/SLOW speed (the model samples ~1 fps, so it sees ~2 fps of play),
   and sent as video. For each window the model records the shot type and every
   on-ball action: time, action, team, what is READ on the player's shirt (number
   and/or printed name, never guessed), body shape on receptions, start/end
   zones, pass length, outcome, pressure. It also records every shirt it can
   read (team + number + printed name), including in replays and close-ups,
   which is what later turns numbers into names.

No commentary, no captions, no squad sheet. Raw output is saved per window in
data/clips/<clip>/vision.json. Resumable: finished windows are skipped.

    CLIP_VIDEO=data/clips/clip2/clip2.mp4 python pipeline/vision.py
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
from config import CHUNKS, CLIP_DIR, OPENROUTER_API_KEY, OPENROUTER_URL, VIDEO, VIDEO_MODEL

WIN_S, STRIDE_S, SLOW = 6, 3, 2.0
SURVEY_FRAMES = 8
OUT = CLIP_DIR / "vision.json"

ACTIONS = ("receive|pass|cross|shot|header|dribble|tackle|interception|clearance|save|"
           "foul|corner|free_kick|throw_in|goal_kick|penalty|goal")

SURVEY_PROMPT = """These are frames from one soccer video: a broadcast, a training session, or any other footage.
Identify the two sides playing. If an on-screen scoreboard names them, use those names (fullest spelling you see)
and list them in scoreboard order. Otherwise name each side by what it wears, e.g. "Red bibs" and "No bibs".
Describe each side's outfield kit and goalkeeper kit (null if there is no distinct goalkeeper kit).
Return ONLY JSON: {"teams": [{"name": ..., "scoreboard_abbr": ... or null, "kit": ..., "goalkeeper_kit": ...}, {...}]}"""

ACTION_PROMPT = """This is {win} seconds of soccer footage slowed to {speed:g}x (it plays for {dur:g} s).
Teams (name: kit): {teams}.
Report ONLY what you can see. Never name a player from memory or from where he plays.

shot_type: live_wide | live_close | replay | stoppage | celebration.

actions: only if shot_type is live_wide or live_close. Every on-ball action, each with:
- t_offset_s: seconds into the SLOWED video
- action: one of {actions}
- team: exact team name above, from the kit
- shirt_number: digits you can actually read on this player's shirt, else null
- shirt_name: printed name you can actually read on this player's shirt, else null
- body_shape (receive only): half_turn | open | back_to_goal | side_on | unclear
  (half_turn = hips angled so he can play forward and back; open = facing the opponent's goal)
- start: {{"third": def|mid|att, "side": left|center|right}} where the action starts, from the
  acting team's point of view (att = the third nearest the goal they attack)
- end: same format, where the ball ends up (pass, cross, shot, clearance), else null
- length: short | medium | long (pass and cross only), else null
- outcome: completed | intercepted | out | on_target | off_target | blocked | saved | goal | won | lost | none
- pressure: none | light | tight
- confidence_0_1
- detail: one factual sentence of what you see

shirts: EVERY shirt in the clip (any shot type, replays included) whose number or printed name
you can read: {{"team": ..., "shirt_number": ... or null, "shirt_name": ... or null}}.

Return ONLY JSON: {{"shot_type": ..., "actions": [...], "shirts": [...]}}"""

client = OpenAI(base_url=OPENROUTER_URL, api_key=OPENROUTER_API_KEY)


def duration():
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "default=nk=1:nw=1", str(VIDEO)], capture_output=True, text=True)
    return float(out.stdout.strip())


def ask(content, max_tokens):
    r = client.chat.completions.create(model=VIDEO_MODEL, temperature=0.0, max_tokens=max_tokens,
                                       messages=[{"role": "user", "content": content}])
    return json.loads(repair_json(r.choices[0].message.content))


def survey(dur):
    frames = []
    for k in range(SURVEY_FRAMES):
        t = dur * (k + 0.5) / SURVEY_FRAMES
        jpg = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", f"{t:.1f}",
                              "-i", str(VIDEO), "-frames:v", "1", "-f", "image2", "-c:v", "mjpeg", "-"],
                             capture_output=True, check=True).stdout
        frames.append({"type": "image_url", "image_url": {
            "url": "data:image/jpeg;base64," + base64.b64encode(jpg).decode()}})
    return ask([{"type": "text", "text": SURVEY_PROMPT}] + frames, 2000)["teams"]


def window(start, teams):
    CHUNKS.mkdir(parents=True, exist_ok=True)
    tmp = CHUNKS / f"v_{start:04d}.mp4"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", str(start),
                    "-t", str(WIN_S), "-i", str(VIDEO), "-an", "-vf", f"setpts={SLOW}*PTS",
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "24", str(tmp)], check=True)
    uri = "data:video/mp4;base64," + base64.b64encode(tmp.read_bytes()).decode()
    tdesc = "; ".join(f"{t['name']}: {t['kit']} (goalkeeper {t['goalkeeper_kit']})" for t in teams)
    text = ACTION_PROMPT.format(win=WIN_S, speed=1 / SLOW, dur=WIN_S * SLOW, teams=tdesc, actions=ACTIONS)
    d = ask([{"type": "text", "text": text}, {"type": "file", "file": {"file_data": uri, "format": "video/mp4"}}],
            10000)
    return {"start": start, "shot_type": d.get("shot_type"), "actions": d.get("actions") or [],
            "shirts": d.get("shirts") or []}


def main():
    dur = duration()
    data = json.loads(OUT.read_text()) if OUT.exists() else {}
    data.update(video=VIDEO.name, duration=dur, model=VIDEO_MODEL, win_s=WIN_S, stride_s=STRIDE_S, slow=SLOW)
    if not data.get("teams"):
        data["teams"] = survey(dur)
    print("teams:", "; ".join(f"{t['name']} ({t['kit']})" for t in data["teams"]), flush=True)
    done = {w["start"]: w for w in data.get("windows", [])}
    todo = [s for s in range(0, int(dur) - 1, STRIDE_S) if s not in done]
    print(f"{len(todo)} windows to do", flush=True)
    lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(window, s, data["teams"]): s for s in todo}
        for f in as_completed(futs):
            try:
                w = f.result()
            except Exception as e:
                print(f"[{futs[f]}s] FAILED: {e}", file=sys.stderr, flush=True)
                continue
            with lock:
                done[w["start"]] = w
                data["windows"] = [done[k] for k in sorted(done)]
                OUT.with_suffix(".tmp").write_text(json.dumps(data))
                OUT.with_suffix(".tmp").replace(OUT)
            print(f"[{len(done)}] {w['start']}s {w['shot_type']}: {len(w['actions'])} actions, "
                  f"{len(w['shirts'])} shirts", flush=True)
    print(f"{len(done)} windows -> {OUT}")


if __name__ == "__main__":
    main()
