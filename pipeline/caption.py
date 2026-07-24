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
INTERVAL_S = 10

VLM_MODEL = "google/gemini-2.5-flash"
OCR_MODEL = "google/gemini-2.5-flash"
EMBED_MODEL = "openai/text-embedding-3-small"
WIN_BEFORE, WIN_AFTER = 12, 8

# General sports event taxonomy (soccer).
EVENTS = ("goal|shot|save|foul|yellow_card|red_card|penalty|corner|free_kick|"
          "throw_in|offside|tackle|duel|cross|attack|buildup|transition|"
          "celebration|substitution|injury|kickoff|out_of_play|other")

PROMPT = (
    "You are a sports video analyst. You get ONE broadcast frame, the on-screen "
    "TEXT read by OCR, and (optionally) the TV COMMENTARY around this moment. "
    "Assume NOTHING about which match this is — use only what the clip shows or says.\n"
    "GROUNDING RULES (critical):\n"
    "- Teams and score: take from the OCR scorebug text if present, else 'unknown'.\n"
    "- Player identity: name a player ONLY if a name appears in the OCR text (e.g. a "
    "name graphic) OR the commentary names them, OR a jersey number is clearly legible. "
    "Otherwise refer to the player as '#<number> <shirt colour>' or by shirt colour "
    "alone. NEVER guess a name from general football knowledge.\n"
    "- Set identity_source to how you got the main player's identity: "
    "name_graphic | commentary | jersey_number | none.\n"
    f"Classify the primary EVENT as one of: {EVENTS}.\n"
    "Return ONLY a JSON object with keys: "
    "event (one of the list above), event_confidence (0-1), "
    "phase (buildup|progression|final_third|transition|set_piece|out_of_play), "
    "zone_third (def_third|mid_third|att_third), zone_side (left|center|right), "
    "pressure (none|low|medium|high), "
    "teams (from OCR, or 'unknown'), score (from OCR scorebug, or 'unknown'), "
    "card (none|yellow|red), card_player (name/desc or null), "
    "players (array of grounded names or '#7 red' style descriptors, [] if none), "
    "scorer (grounded name/desc or null), assist (grounded name/desc or null), "
    "shot_by (grounded name/desc or null), identity_source, "
    "prose (one factual sentence). No extra text."
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
    obj, _ = json.JSONDecoder().raw_decode(text[i:])
    return obj


def caption(path: Path, commentary: str, ocr: str) -> dict:
    user = (PROMPT + f"\n\nOCR TEXT (on screen): {ocr or '(none)'}"
            f"\nCOMMENTARY (around this moment): {commentary or '(none)'}")
    content = [
        {"type": "text", "text": user},
        {"type": "image_url", "image_url": {"url": img_uri(path)}},
    ]
    last = None
    for temp in (0.0, 0.4, 0.8):  # temp 0 is deterministic, so escalate on parse failure
        kw = dict(model=VLM_MODEL, temperature=temp, max_tokens=2000,
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
    tmp = OUT.with_suffix(".tmp")
    tmp.write_text(json.dumps({"video": "seed.mp4", "moments": moments}, indent=2))
    tmp.replace(OUT)


def build_moment(i, fp):
    t = i * INTERVAL_S
    comm = commentary_window(t)
    ocr = ocr_text(fp)
    cap = caption(fp, comm, ocr)
    players = cap.get("players") or []
    prose = cap.get("prose", "")
    embed_text = " ".join([prose, " ".join(map(str, players)), ocr, comm]).strip()
    vec = embed(embed_text) if embed_text else []
    return {
        "id": i, "t_seconds": t, "timecode": timecode(t),
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
        "scorer": cap.get("scorer"),
        "assist": cap.get("assist"),
        "shot_by": cap.get("shot_by"),
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
            if "identity_source" in m:  # only reuse moments from THIS schema
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
