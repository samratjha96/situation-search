"""Index a match video's frames into searchable 'moments' — clip-grounded.

Designed for ARBITRARY analyst clips: assume nothing about the match. Ground
everything in the clip itself:
  1. OCR (PaddleOCR-VL) reads the on-screen text — scorebug (teams, score,
     clock), name graphics, cards, captions. The scorebug is the reliable
     structured signal and detects goals via score change.
  2. Commentary (local Whisper) is used IF present, else ignored.
  3. A VLM classifies the sports EVENT (shot, foul, card, attack, goal, ...)
     and only names a player when the OCR text or commentary supports it, or a
     jersey number is legible — otherwise it uses '#<number> <shirt colour>'.
     It NEVER guesses names from general football knowledge.

Output: data/index.json
"""
import base64
import json
import os
import re
import sys
from pathlib import Path

from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent
FRAMES = ROOT / "data" / "frames"
TRANSCRIPT = ROOT / "data" / "transcript" / "seed_audio.json"
OUT = ROOT / "data" / "index.json"
INTERVAL_S = 3   # dense sampling so events land on the right second

VLM_MODEL = "google/gemini-2.5-flash"
OCR_MODEL = "google/gemini-2.5-flash"
EMBED_MODEL = "openai/text-embedding-3-small"
WIN_BEFORE, WIN_AFTER = 5, 5  # tight: commentary is only a name hint, not the story

# General sports event taxonomy (soccer).
EVENTS = ("goal|shot|save|foul|yellow_card|red_card|penalty|corner|free_kick|"
          "throw_in|offside|tackle|duel|cross|attack|buildup|transition|"
          "celebration|substitution|injury|kickoff|out_of_play|other")

# VISUAL-FIRST: the frame is the ground truth. Commentary lags (it narrates the
# past), so it must NOT drive what the moment is or when — it only supplies names
# for players actually visible in the frame.
PROMPT = (
    "You are a sports video analyst. Describe ONLY what is VISIBLE in this single "
    "broadcast frame — the frame is the ground truth. Commentary LAGS and narrates the "
    "past, so NEVER describe an action just because commentary mentions it; include an "
    "action ONLY if you can actually SEE it in this frame.\n"
    "You also get on-screen OCR text and (optionally) nearby commentary. Use commentary "
    "ONLY to attach a name to a player you can clearly SEE — never to decide what is "
    "happening or when.\n"
    "GROUNDING RULES:\n"
    "- event: the primary on-ball action VISIBLE right now. If no clear action is visible "
    "(players jogging, a wide shot, a close-up, a replay), use a truthful visible label "
    "(buildup, transition, duel, out_of_play, ...). Do NOT import a shot / goal / switch / "
    "save from commentary if it is not on screen.\n"
    "- teams and score: from the OCR scorebug if present, else 'unknown'.\n"
    "- player identity: name a player ONLY if a name is on screen (OCR graphic), a jersey "
    "number is clearly legible, or the commentary clearly names a player you can SEE. "
    "Otherwise use '#<number> <shirt colour>' or shirt colour. NEVER guess from football "
    "knowledge.\n"
    "- identity_source: name_graphic | jersey_number | commentary | none.\n"
    f"- event is one of: {EVENTS}.\n"
    "Return ONLY a JSON object with keys: event, event_confidence (0-1), "
    "phase (buildup|progression|final_third|transition|set_piece|out_of_play), "
    "zone_third (def_third|mid_third|att_third), zone_side (left|center|right), "
    "pressure (none|low|medium|high), teams, score, card (none|yellow|red), card_player, "
    "players, scorer, assist, shot_by, identity_source, "
    "prose (ONE factual sentence describing ONLY what is visible in the frame). No extra text."
)

client = OpenAI(base_url="https://openrouter.ai/api/v1",
                api_key=os.environ["OPENROUTER_API_KEY"])

SEGS = json.loads(TRANSCRIPT.read_text())["segments"] if TRANSCRIPT.exists() else []


def commentary_window(t: int) -> str:
    a, b = t - WIN_BEFORE, t + WIN_AFTER
    return " ".join(s["text"].strip() for s in SEGS if s["end"] >= a and s["start"] <= b)


def img_uri(path: Path) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(path.read_bytes()).decode()


def ocr_text(path: Path) -> str:
    """Read all on-screen text with a dedicated OCR model. '' on failure."""
    try:
        r = client.chat.completions.create(
            model=OCR_MODEL, temperature=0.0, max_tokens=200,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": "Extract all on-screen text verbatim, one item per line. If none, output NONE."},
                {"type": "image_url", "image_url": {"url": img_uri(path)}},
            ]}])
        txt = r.choices[0].message.content.strip()
        return "" if txt.upper() == "NONE" else txt
    except Exception:
        return ""


def parse_json(text: str) -> dict:
    i = text.find("{")
    if i == -1:
        raise ValueError(f"no JSON in: {text[:120]}")
    try:
        obj, _ = json.JSONDecoder().raw_decode(text[i:])
        return obj
    except Exception:
        from json_repair import repair_json  # fixes truncated/malformed LLM JSON
        return json.loads(repair_json(text[i:]))


def caption(path: Path, commentary: str, ocr: str) -> dict:
    user = (PROMPT + f"\n\nOCR TEXT (on screen): {ocr or '(none)'}"
            f"\nCOMMENTARY (around this moment): {commentary or '(none)'}")
    content = [
        {"type": "text", "text": user},
        {"type": "image_url", "image_url": {"url": img_uri(path)}},
    ]
    last = None
    for temp in (0.0, 0.4, 0.8):  # temp 0 is deterministic, so escalate on parse failure
        kw = dict(model=VLM_MODEL, temperature=temp, max_tokens=3500,
                  messages=[{"role": "user", "content": content}])
        try:
            r = client.chat.completions.create(**kw, response_format={"type": "json_object"})
        except Exception:
            r = client.chat.completions.create(**kw)
        try:
            return parse_json(r.choices[0].message.content)
        except Exception as e:
            last = e
    raise last


def embed(text: str) -> list[float]:
    r = client.embeddings.create(model=EMBED_MODEL, input=[text],
                                 encoding_format="float",
                                 extra_body={"input_type": "passage"})
    return r.data[0].embedding


def timecode(s: int) -> str:
    return f"{s // 60:02d}:{s % 60:02d}"


def save(moments):
    # preserve other index sections (events, video_events) built by other passes
    d = json.loads(OUT.read_text()) if OUT.exists() else {}
    d["video"] = "seed.mp4"
    d["moments"] = moments
    tmp = OUT.with_suffix(".tmp")
    tmp.write_text(json.dumps(d, indent=2))
    tmp.replace(OUT)


def player_str(p):
    """Coerce a player field to a string (the model sometimes returns a dict)."""
    if p is None:
        return None
    if isinstance(p, dict):
        n = p.get("name") or p.get("desc") or p.get("player")
        if n:
            return str(n)
        num = p.get("shirt_number") or p.get("number")
        col = p.get("shirt_colour") or p.get("colour") or p.get("team")
        return (f"#{num} {col}".strip() if (num or col) else "unknown")
    return str(p)


def build_moment(i, fp):
    t = i * INTERVAL_S
    comm = commentary_window(t)
    ocr = ocr_text(fp)
    cap = caption(fp, comm, ocr)
    players = [player_str(p) for p in (cap.get("players") or [])]
    prose = cap.get("prose", "")
    # embed the VISUAL description (+ visible players + on-screen text) — NOT the
    # commentary, so search matches what is on screen, not the narration.
    embed_text = " ".join([prose, " ".join(map(str, players)), ocr]).strip()
    vec = embed(embed_text) if embed_text else []
    return {
        "id": i, "t_seconds": t, "timecode": timecode(t),
        "caption_mode": "visual_first",
        "image": f"frames/{fp.name}",
        "event": cap.get("event", "other"),
        "event_confidence": cap.get("event_confidence"),
        "phase": cap.get("phase", "unknown"),
        "zone_third": cap.get("zone_third", "unknown"),
        "zone_side": cap.get("zone_side", "unknown"),
        "pressure": cap.get("pressure", "unknown"),
        "teams": cap.get("teams", "unknown"),
        "score": cap.get("score", "unknown"),
        "card": cap.get("card", "none"),
        "card_player": cap.get("card_player"),
        "players": players,
        "scorer": player_str(cap.get("scorer")),
        "assist": player_str(cap.get("assist")),
        "shot_by": player_str(cap.get("shot_by")),
        "identity_source": cap.get("identity_source"),
        "ocr_text": ocr,
        "prose": prose,
        "commentary": comm,
        "embedding": vec,
    }


def main():
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading
    frames = sorted(FRAMES.glob("*.jpg"))
    by_id = {}
    if OUT.exists():
        for m in json.loads(OUT.read_text()).get("moments", []):
            if m.get("caption_mode") == "visual_first":  # only reuse visual-first moments
                by_id[m["id"]] = m
    todo = [(i, fp) for i, fp in enumerate(frames) if i not in by_id]
    lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(build_moment, i, fp): (i, fp) for i, fp in todo}
        for fut in as_completed(futs):
            i, fp = futs[fut]
            try:
                m = fut.result()
                with lock:
                    by_id[i] = m
                    save([by_id[k] for k in sorted(by_id)])
                who = m["scorer"] or m["shot_by"] or (m["players"][0] if m["players"] else "")
                print(f"[{len(by_id)}/{len(frames)}] {m['timecode']}  {m['event']:<12} {str(who):<16} {m['prose'][:46]}", flush=True)
            except Exception as e:
                print(f"[frame {i}] FAILED: {e}", file=sys.stderr, flush=True)
    print(f"\nWrote {len(by_id)} moments -> {OUT}")


if __name__ == "__main__":
    main()
