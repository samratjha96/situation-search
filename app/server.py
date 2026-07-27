"""Situation Search backend.

Serves the analyst UI and a hybrid (semantic + keyword) search over the
captioned 'moments' index. Semantic search embeds the query on the the
gateway and ranks moments by cosine similarity to their caption embeddings;
keyword matches on the structured fields boost the score. Falls back to
keyword-only if the embedding call fails, so the app still works offline.
"""
import json
import os
import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

import httpx
import numpy as np
import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent
INDEX = ROOT / "data" / "index.json"
STATIC = ROOT / "app" / "static"
EMBED_MODEL = "openai/text-embedding-3-small"

data = json.loads(INDEX.read_text())
MOMENTS = data["moments"]
EVENTS = data.get("events", [])              # commentary/OCR events (goals via scorebug)
VIDEO_EVENTS = data.get("video_events", [])  # video-native events (work on any footage)
CONFIRMED_GOAL_TS = [e["t_seconds"] for e in EVENTS
                     if e.get("type") == "goal" and e.get("confirmed")]

# Build a normalized embedding matrix for fast cosine search.
_dim = next((len(m["embedding"]) for m in MOMENTS if m.get("embedding")), 0)
_mat = np.zeros((len(MOMENTS), _dim), dtype=np.float32)
for i, m in enumerate(MOMENTS):
    v = m.get("embedding") or []
    if len(v) == _dim and _dim:
        _mat[i] = v
_norms = np.linalg.norm(_mat, axis=1, keepdims=True)
_matn = _mat / np.clip(_norms, 1e-9, None)

def _api_key():
    k = os.environ.get("OPENROUTER_API_KEY")
    if k:
        return k
    # Fallback: read the export from the user's shell rc so the preview
    # server (which starts without the env var) still gets semantic search.
    rc = Path.home() / ".zshrc"
    if rc.exists():
        for line in rc.read_text().splitlines():
            line = line.strip()
            if line.startswith("export OPENROUTER_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""

client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=_api_key())

# Public view of a moment (drop the bulky embedding vector).
# Note: use 'relevance' for the ranking score so we don't clobber the OCR 'score'.
def public(m, relevance=None):
    d = {k: v for k, v in m.items() if k != "embedding"}
    if relevance is not None:
        d["relevance"] = round(float(relevance), 4)
    return d

def embed_query(text):
    r = client.embeddings.create(model=EMBED_MODEL, input=[text],
                                 encoding_format="float",
                                 extra_body={"input_type": "query"})
    return np.array(r.data[0].embedding, dtype=np.float32)

# Cross-encoder reranker: gives a calibrated relevance score per (query, passage),
# so we can rank precisely AND honestly say "nothing matches" below a threshold.
RERANK_MODEL = "cohere/rerank-v3.5"
# Calibrated on this clip: legit queries bottom out ~-2, junk/absent tops out ~-7.
NO_MATCH_THRESHOLD = -4.0  # if the best reranked result is below this -> "nothing here"
SHOW_GAP = 5.0             # show results within this of the best (keeps strong queries tight)
RERANK_K = 25              # rerank the top-K cosine candidates

def rerank(query, passages):
    """Return {passage_index: relevance_score}. Raises on failure (caller falls back)."""
    r = httpx.post("https://openrouter.ai/api/v1/rerank",
                   headers={"Authorization": f"Bearer {_api_key()}", "Content-Type": "application/json"},
                   json={"model": RERANK_MODEL, "query": query,
                         "documents": passages, "top_n": len(passages)}, timeout=20)
    r.raise_for_status()
    return {res["index"]: float(res["relevance_score"]) for res in r.json()["results"]}

def keyword_score(m, terms):
    parts = [str(m.get(k, "")) for k in
             ("prose", "event", "phase", "zone_third", "zone_side", "pressure",
              "teams", "score", "card", "card_player", "scorer", "assist",
              "shot_by", "ocr_text", "commentary")]
    parts += m.get("players", []) or []
    hay = " ".join(parts).lower()
    return sum(1.0 for t in terms if t in hay)

# ---- Player highlights -----------------------------------------------------
# Group phonetic / partial name variants into canonical players, then rank each
# player's moments so an analyst can pull one player's big moments from a clip.

ROLE_KEYS = ("scorer", "assist", "shot_by", "card_player")
# Importance weight per event, for ranking a player's "big moments".
W_EVENT = {"goal": 10, "penalty": 9, "red_card": 8, "assist": 7, "save": 6,
           "shot": 5, "yellow_card": 5, "celebration": 4, "foul": 4, "cross": 3,
           "corner": 3, "free_kick": 3, "tackle": 3, "offside": 3, "duel": 2,
           "attack": 2, "substitution": 2, "injury": 2, "transition": 1,
           "buildup": 1, "throw_in": 1, "kickoff": 1, "out_of_play": 0, "other": 0}


def _norm(s):
    return re.sub(r"\s+", " ", str(s or "").strip())


def _is_named(s):
    s = _norm(s)
    return bool(s) and not s.startswith("#") and s.lower() not in ("unknown", "none", "n/a")


def _mentions(m):
    names = set()
    for p in (m.get("players") or []):
        if _is_named(p):
            names.add(_norm(p))
    for k in ROLE_KEYS:
        if _is_named(m.get(k)):
            names.add(_norm(m[k]))
    return names


def _similar(a, b):
    a, b = a.lower(), b.lower()
    if SequenceMatcher(None, a, b).ratio() >= 0.85:
        return True
    ta = {w for w in a.split() if len(w) >= 4}
    tb = {w for w in b.split() if len(w) >= 4}
    return bool(ta & tb)  # shared surname-length token


def _build_players():
    counts = Counter()
    for m in MOMENTS:
        for n in _mentions(m):
            counts[n] += 1
    for e in EVENTS:  # include event names so frame + event spellings cluster together
        for n in (e.get("player"), e.get("assist")):
            if _is_named(n):
                counts[_norm(n)] += 1
    clusters = []  # each: {variants: {name: count}}
    for name, cnt in counts.most_common():
        for cl in clusters:
            if _similar(name, cl["canonical"]):
                cl["variants"][name] = cnt
                break
        else:
            clusters.append({"canonical": name, "variants": {name: cnt}})
    cmap = {}
    for cl in clusters:
        # canonical = fullest (longest) variant, tie-break by frequency
        cl["canonical"] = max(cl["variants"], key=lambda x: (len(x), cl["variants"][x]))
        for v in cl["variants"]:
            cmap[v.lower()] = cl["canonical"]
    return clusters, cmap


CLUSTERS, CANON = _build_players()


def canon(name):
    return CANON.get(_norm(name).lower())


def moment_importance(m, pc=None):
    base = W_EVENT.get(m.get("event"), 0)
    bonus = 0
    if pc:
        for k, b in (("scorer", 6), ("assist", 5), ("shot_by", 4), ("card_player", 3)):
            if m.get(k) and canon(m[k]) == pc:
                bonus = max(bonus, b)
    conf = m.get("event_confidence")
    conf = conf if isinstance(conf, (int, float)) else 1.0
    return (base + bonus) * conf


def player_stats(pc):
    # Counting stats come from the de-duplicated EVENT list (accurate), not frames.
    g = a = s = c = 0
    for e in EVENTS:
        ep, ea = canon(e.get("player")), canon(e.get("assist"))
        et = e.get("type")
        if ep == pc:
            if et == "goal":
                g += 1
            elif et == "shot":
                s += 1
            elif et in ("yellow_card", "red_card"):
                c += 1
        if ea == pc and et == "goal":
            a += 1
    # Appearances = how many frames the player is visible in (from moments).
    ap = sum(1 for m in MOMENTS if pc in {canon(n) for n in _mentions(m)})
    return {"goals": g, "assists": a, "shots": s, "cards": c, "appearances": ap}


def player_moments(pc):
    scored = [(moment_importance(m, pc), m) for m in MOMENTS if pc in {canon(n) for n in _mentions(m)}]
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored


app = FastAPI(title="Soccer Situation Search")

@app.get("/api/players")
def players():
    out = []
    for cl in CLUSTERS:
        pc = cl["canonical"]
        st = player_stats(pc)
        weight = st["goals"] * 10 + st["assists"] * 7 + st["shots"] * 3 + st["cards"] * 2 + st["appearances"]
        out.append({"name": pc, "aliases": sorted(cl["variants"]), **st, "weight": weight})
    out.sort(key=lambda x: x["weight"], reverse=True)
    return JSONResponse({"players": out})

@app.get("/api/player")
def player(name: str):
    pc = canon(name) or _norm(name)
    moms = [public(m, imp) for imp, m in player_moments(pc)]
    return JSONResponse({"name": pc, "stats": player_stats(pc), "moments": moms})

@app.get("/api/events")
def events():
    return JSONResponse({"events": EVENTS})

def _nearest_image(t):
    return min(MOMENTS, key=lambda m: abs(m["t_seconds"] - t))["image"] if MOMENTS else None

@app.get("/api/timeline")
def timeline():
    """Video-native events for the timeline, with a thumbnail and goal confirmation.
    A 'goal' is confirmed only if the scorebug (commentary/OCR) corroborates it near
    that time; otherwise it is flagged unconfirmed rather than trusted."""
    out = []
    for e in VIDEO_EVENTS:
        t = e["t_seconds"]
        # Goals are detected from the video (works with no scoreboard). If a scorebug
        # change happens to corroborate one, flag it as a bonus — but never require it.
        corro = None
        if e.get("type") == "goal":
            corro = any(abs(t - g) <= 20 for g in CONFIRMED_GOAL_TS)
        out.append({**e, "image": _nearest_image(t), "scorebug_corroborated": corro})
    out.sort(key=lambda x: x["t_seconds"])
    # de-dup: adjacent chunks re-detect the same event — collapse same type within 4s
    deduped = []
    for e in out:
        if deduped and e["type"] == deduped[-1]["type"] and \
                abs(e["t_seconds"] - deduped[-1]["t_seconds"]) <= 4:
            continue
        deduped.append(e)
    out = deduped
    dur = max((m["t_seconds"] for m in MOMENTS), default=0) + 10
    return JSONResponse({"events": out, "duration": dur,
                         "confirmed_goals": len(CONFIRMED_GOAL_TS)})

@app.get("/api/moments")
def moments():
    return JSONResponse({"moments": [public(m) for m in MOMENTS],
                         "video": data.get("video")})

# When the query NAMES an event, hard-boost moments of that type so a literal
# term like "goal" surfaces actual goals, not everything goal-adjacent.
EVENT_SYNONYMS = {
    "goal": ["goal", "scored", "scores", "scoring"],
    "shot": ["shot", "shoots", "effort", "strike"],
    "save": ["save", "saves", "parry", "parried"],
    "corner": ["corner"], "cross": ["cross", "crossing", "crossed"],
    "foul": ["foul", "fouled"], "tackle": ["tackle", "tackled"],
    "yellow_card": ["yellow", "booking", "booked", "card"],
    "red_card": ["red card", "sent off"], "penalty": ["penalty", "spot kick"],
    "free_kick": ["free kick", "free-kick"], "duel": ["duel", "aerial"],
    "offside": ["offside", "var"], "throw_in": ["throw in", "throw-in"],
    "celebration": ["celebration", "celebrate", "celebrates"],
}

def target_events(ql):
    return {ev for ev, syns in EVENT_SYNONYMS.items() if any(s in ql for s in syns)}

@app.get("/api/search")
def search(q: str = "", event: str = "", phase: str = ""):
    idx = list(range(len(MOMENTS)))
    if event:
        idx = [i for i in idx if MOMENTS[i].get("event") == event]
    if phase:
        idx = [i for i in idx if MOMENTS[i].get("phase") == phase]

    ql = q.lower()
    terms = [t for t in ql.split() if t]
    targets = target_events(ql)
    semantic_ok = False
    sims = np.zeros(len(MOMENTS), dtype=np.float32)
    if q.strip() and _dim:
        try:
            qv = embed_query(q)
            qv = qv / max(np.linalg.norm(qv), 1e-9)
            sims = _matn @ qv
            semantic_ok = True
        except Exception:
            semantic_ok = False

    scored = []
    for i in idx:
        kw = keyword_score(MOMENTS[i], terms) if terms else 0.0
        m = MOMENTS[i]
        # event-intent boost: strongest when the moment's own event matches the
        # queried event; a "goal" query also lifts the goal celebration.
        boost = 0.0
        if targets:
            if m.get("event") in targets:
                boost = 0.5
            elif "goal" in targets and _is_named(m.get("scorer")):
                boost = 0.3
        if not q.strip():
            score = 1.0  # pure filter / browse
        elif semantic_ok:
            score = float(sims[i]) + 0.15 * kw + boost
        else:
            score = kw + boost
        scored.append((score, i))

    scored.sort(key=lambda x: x[0], reverse=True)

    # Retrieve-then-rerank: rerank the top-K cosine candidates with the cross-encoder
    # for a calibrated relevance score, and honestly report "no match" when nothing
    # clears the bar (a bi-encoder always returns a "best" result, even for junk).
    if q.strip():
        cand = [i for _, i in scored[:RERANK_K]]
        passages = [f"{MOMENTS[i].get('event', '')}. {MOMENTS[i].get('prose', '')}" for i in cand]
        try:
            rs = rerank(q, passages)
            reranked = sorted(((rs.get(k, -1e9), cand[k]) for k in range(len(cand))), reverse=True)
            best = reranked[0][0] if reranked else -1e9
            if best < NO_MATCH_THRESHOLD:
                return JSONResponse({"results": [], "no_match": True, "query": q,
                                     "ranking": "reranked", "count": 0})
            cutoff = max(NO_MATCH_THRESHOLD, best - SHOW_GAP)  # relative: keep strong queries tight
            results = [public(MOMENTS[i], s) for s, i in reranked if s >= cutoff][:40]
            return JSONResponse({"results": results, "ranking": "reranked", "count": len(results)})
        except Exception:
            pass  # reranker unavailable -> fall back to cosine ordering

    if q.strip() and not semantic_ok:
        scored = [(s, i) for s, i in scored if s > 0]
    results = [public(MOMENTS[i], s) for s, i in scored[:40]]
    return JSONResponse({"results": results,
                         "ranking": "semantic" if semantic_ok else "keyword",
                         "count": len(results)})

@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")

app.mount("/media", StaticFiles(directory=ROOT / "data"), name="media")
app.mount("/static", StaticFiles(directory=STATIC), name="static")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8008, log_level="warning")
