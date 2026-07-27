"""Event-level attribution — the accurate layer.

Per-frame attribution double-counts (a goal spans many frames) and guesses
scorers from narrow windows. This module attributes at the EVENT level from
two grounded signals:

  - OCR scorebug score timeline  -> WHEN/WHETHER a goal happened (objective;
    the score increments exactly once per goal, so no replay double-count).
  - Full commentary transcript   -> WHO (one holistic pass names each scorer
    once). Robust line-delimited output; a strong model for completeness.

Goals = score increments, each attributed to the scorer the commentary names
near that time. Other events (cards, shots, saves, fouls, subs) come from the
commentary event list, de-duplicated. Each event is aligned to the nearest
frame so the UI can show the clip. Degrades gracefully: no scorebug -> use
commentary goals; no commentary -> score-change goals with no name.

Reads/writes data/index.json (adds a top-level "events" list).
"""
import json
import os
import re
import sys
from pathlib import Path

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import OUT as INDEX, TRANSCRIPT  # noqa: E402
EXTRACT_MODEL = "google/gemini-2.5-pro"  # reasoning model: complete + accurate

client = OpenAI(base_url="https://openrouter.ai/api/v1",
                api_key=os.environ["OPENROUTER_API_KEY"])

EVENT_PROMPT = (
    "You are a football analyst. Below is the FULL time-stamped commentary of a clip. "
    "Names may be misspelled (auto-transcribed) — normalize phonetically to the real player. "
    "List EVERY notable event across the ENTIRE clip. De-duplicate replays (a goal shown "
    "then replayed is ONE event). Attribute only names the commentary states.\n"
    "Output ONE event per line, pipe-delimited, NO header, NO preamble:\n"
    "TIME|TYPE|PLAYER|ASSIST|TEAM|CONFIDENCE\n"
    "TIME as mm:ss. TYPE in: goal shot save foul yellow_card red_card penalty corner "
    "substitution big_chance other. Use '-' for empty fields.\n\nCOMMENTARY:\n"
)


def mmss_to_s(t):
    m = re.match(r"\s*(\d+):(\d+)", str(t))
    return int(m[1]) * 60 + int(m[2]) if m else None


def parse_score(s):
    m = re.search(r"(\d+)\s*-\s*(\d+)", str(s or ""))
    return (int(m[1]), int(m[2])) if m else None


def load_transcript_text():
    if not TRANSCRIPT.exists():
        return ""
    segs = json.loads(TRANSCRIPT.read_text())["segments"]
    def tc(x): return f"{int(x)//60:02d}:{int(x)%60:02d}"
    return "\n".join(f"[{tc(s['start'])}] {s['text'].strip()}" for s in segs)


def extract_commentary_events(transcript):
    """One holistic pass -> list of {t, type, player, assist, team, conf}."""
    if not transcript:
        return []
    r = client.chat.completions.create(
        model=EXTRACT_MODEL, temperature=0.0, max_tokens=9000,
        messages=[{"role": "user", "content": EVENT_PROMPT + transcript}])
    events = []
    for line in r.choices[0].message.content.splitlines():
        if line.count("|") < 4:
            continue
        p = [x.strip() for x in line.split("|")]
        t = mmss_to_s(p[0])
        if t is None:
            continue
        events.append({"t": t, "type": p[1].lower(),
                       "player": None if p[2] in ("-", "") else p[2],
                       "assist": None if p[3] in ("-", "") else p[3],
                       "team": None if p[4] in ("-", "") else p[4],
                       "conf": p[5] if len(p) > 5 else None})
    return events


def score_change_goals(moments):
    """Objective goal timestamps from the OCR scorebug (total score increments)."""
    goals, last_total = [], None
    for m in sorted(moments, key=lambda x: x["t_seconds"]):
        sc = parse_score(m.get("score"))
        if not sc:
            continue
        total = sc[0] + sc[1]
        if last_total is not None and total > last_total:
            goals.append({"t": m["t_seconds"], "score": f"{sc[0]}-{sc[1]}"})
        if last_total is None or total >= last_total:  # ignore transient OCR drops
            last_total = total
    return goals


def nearest_moment(moments, t):
    return min(moments, key=lambda m: abs(m["t_seconds"] - t))


def nearest_named(events, t, types, window=30):
    """Closest commentary event of a given type with a named player, within window."""
    cands = [e for e in events if e["type"] in types and e["player"]
             and abs(e["t"] - t) <= window]
    return min(cands, key=lambda e: abs(e["t"] - t)) if cands else None


def build_events(moments, commentary):
    events = []

    # 1) Goals: driven by the objective scorebug, identity from commentary.
    goal_times = score_change_goals(moments)
    if goal_times:
        for g in goal_times:
            src = nearest_named(commentary, g["t"], {"goal", "shot", "other", "big_chance"})
            fr = nearest_moment(moments, g["t"])
            events.append({
                "type": "goal", "t_seconds": g["t"], "timecode": fr["timecode"],
                "score": g["score"], "frame_id": fr["id"], "image": fr["image"],
                "player": src["player"] if src else None,
                "assist": src["assist"] if src else None,
                "confirmed": True, "source": "scorebug+commentary" if src else "scorebug",
            })
    else:
        # No scorebug goals detected -> fall back to commentary goals.
        for e in commentary:
            if e["type"] == "goal":
                fr = nearest_moment(moments, e["t"])
                events.append({
                    "type": "goal", "t_seconds": e["t"], "timecode": fr["timecode"],
                    "score": fr.get("score", "unknown"), "frame_id": fr["id"],
                    "image": fr["image"], "player": e["player"], "assist": e["assist"],
                    "confirmed": False, "source": "commentary"})

    # 2) Non-goal events straight from the (de-duplicated) commentary list.
    for e in commentary:
        if e["type"] in ("goal",):
            continue
        fr = nearest_moment(moments, e["t"])
        events.append({
            "type": e["type"], "t_seconds": e["t"], "timecode": fr["timecode"],
            "score": fr.get("score", "unknown"), "frame_id": fr["id"], "image": fr["image"],
            "player": e["player"], "assist": e["assist"], "confirmed": None,
            "source": "commentary"})

    events.sort(key=lambda e: e["t_seconds"])
    return events


def main():
    data = json.loads(INDEX.read_text())
    moments = data["moments"]
    transcript = load_transcript_text()
    commentary = extract_commentary_events(transcript)
    print(f"commentary events: {len(commentary)}  | scorebug goals: {len(score_change_goals(moments))}")
    events = build_events(moments, commentary)
    data["events"] = events
    INDEX.write_text(json.dumps(data, indent=2))
    print(f"\n{len(events)} attributed events -> data/index.json\n")
    for e in events:
        extra = f" (assist {e['assist']})" if e.get("assist") else ""
        conf = "" if e["confirmed"] is None else (" [confirmed]" if e["confirmed"] else " [unconfirmed]")
        print(f"  {e['timecode']}  {e['type']:<13} {str(e['player']):<20}{extra}{conf}")


if __name__ == "__main__":
    main()
