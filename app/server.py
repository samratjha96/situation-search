"""Situation Search backend — natural-language search over captioned match moments.

Serves the analyst UI and a search API over the offline-built 'moments' index
(one captioned frame-burst every few seconds). A query flows through three steps:

  1. expand   — tactical phrases ("switch of play") get a plain-language gloss so
                 they match the descriptive captions.
  2. retrieve  — embed the query and rank moments by cosine similarity.
  3. rerank    — score the top-K candidates with a cross-encoder; anything below a
                 tuned threshold returns an honest "no match" instead of a
                 bi-encoder's always-present best guess.

The app can hold several clips at once. Each clip has its own index under
data/clips/<name>/; every endpoint takes a ?clip=<name> parameter and the UI picks
one from a dropdown. Also exposes per-player highlights and a video-native event
timeline. Embedding and reranking run on an OpenRouter-style endpoint (URL + key
from the env); if it is unreachable, search degrades to keyword-only.
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
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
STATIC = ROOT / "app" / "static"

# Inference endpoint (OpenRouter-style) — URL + key + model routes from the env.
OPENROUTER_URL = os.environ.get("OPENROUTER_URL", "https://openrouter.ai/api/v1")
OPENROUTER_KEY = os.environ.get("OPENROUTER_API_KEY", "")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "openai/text-embedding-3-small")
RERANK_MODEL = os.environ.get("RERANK_MODEL", "cohere/rerank-v3.5")
client = OpenAI(base_url=OPENROUTER_URL, api_key=OPENROUTER_KEY)

# Calibrated on these clips with NV-RerankQA 1B logits: legit queries bottom out ~-2,
# junk/absent tops out ~-7. A reranker with another score scale (e.g. 0-1) needs its own.
NO_MATCH_THRESHOLD = float(os.environ.get("NO_MATCH_THRESHOLD", "-4.0"))  # best below this -> "nothing here"
SHOW_GAP = float(os.environ.get("SHOW_GAP", "5.0"))  # show results within this of the best
RERANK_K = 40              # rerank the top-K cosine candidates
DEDUP_S = 6.0              # collapse adjacent same-event moments (dense 3s sampling)

ROLE_KEYS = ("scorer", "assist", "shot_by", "card_player")
# Importance weight per event, for ranking a player's "big moments".
W_EVENT = {"goal": 10, "penalty": 9, "red_card": 8, "assist": 7, "save": 6,
           "shot": 5, "yellow_card": 5, "celebration": 4, "foul": 4, "cross": 3,
           "corner": 3, "free_kick": 3, "tackle": 3, "offside": 3, "duel": 2,
           "attack": 2, "substitution": 2, "injury": 2, "transition": 1,
           "buildup": 1, "throw_in": 1, "kickoff": 1, "out_of_play": 0, "other": 0}


# ---- stateless helpers -----------------------------------------------------
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
    # same surname, allowing ASR/OCR misspellings ("Mbeumo" ~ "Bryan Mbuemo");
    # a shared first name alone ("Gabriel ...") is not the same player
    sa, sb = a.split()[-1], b.split()[-1]
    return min(len(sa), len(sb)) >= 4 and SequenceMatcher(None, sa, sb).ratio() >= 0.85


def public(m, relevance=None):
    """Public view of a moment (drop the bulky embedding vector)."""
    d = {k: v for k, v in m.items() if k != "embedding"}
    if relevance is not None:
        d["relevance"] = round(float(relevance), 4)
    return d


# ---- Clip: all per-clip state ---------------------------------------------
class Clip:
    """One indexed match clip: moments, events, embedding matrix, and players.

    Images and the video are served from data/ via /media, so their paths are
    prefixed with clips/<name>/ (each clip keeps its own frames + video)."""

    def __init__(self, name, label, video):
        self.name = name
        self.label = label
        self.prefix = f"clips/{name}/"
        self.video = self.prefix + video
        d = json.loads((DATA / "clips" / name / "index.json").read_text())
        self.moments = d["moments"]
        self.events = d.get("events", [])              # scorebug/commentary events (goals)
        self.video_events = d.get("video_events", [])  # video-native events (any footage)
        for m in self.moments:
            if m.get("image"):
                m["image"] = self.prefix + m["image"]
        for e in self.events:
            if e.get("image"):
                e["image"] = self.prefix + e["image"]
        self.confirmed_goal_ts = [e["t_seconds"] for e in self.events
                                  if e.get("type") == "goal" and e.get("confirmed")]
        # normalized embedding matrix for fast cosine search
        self.dim = next((len(m["embedding"]) for m in self.moments if m.get("embedding")), 0)
        mat = np.zeros((len(self.moments), self.dim), dtype=np.float32)
        for i, m in enumerate(self.moments):
            v = m.get("embedding") or []
            if self.dim and len(v) == self.dim:
                mat[i] = v
        self.matn = mat / np.clip(np.linalg.norm(mat, axis=1, keepdims=True), 1e-9, None)
        self.clusters, self._canon = self._build_players()

    def canon(self, name):
        return self._canon.get(_norm(name).lower())

    def _build_players(self):
        # Group phonetic / partial name variants into canonical players.
        counts = Counter()
        for m in self.moments:
            for n in _mentions(m):
                counts[n] += 1
        for e in self.events:  # include event names so frame + event spellings cluster
            for n in (e.get("player"), e.get("assist")):
                if _is_named(n):
                    counts[_norm(n)] += 1
        clusters = []
        for name, cnt in counts.most_common():
            for cl in clusters:
                if _similar(name, cl["canonical"]):
                    cl["variants"][name] = cnt
                    break
            else:
                clusters.append({"canonical": name, "variants": {name: cnt}})
        cmap = {}
        for cl in clusters:
            cl["canonical"] = max(cl["variants"], key=lambda x: (len(x), cl["variants"][x]))
            for v in cl["variants"]:
                cmap[v.lower()] = cl["canonical"]
        return clusters, cmap

    def moment_importance(self, m, pc=None):
        base = W_EVENT.get(m.get("event"), 0)
        bonus = 0
        if pc:
            for k, b in (("scorer", 6), ("assist", 5), ("shot_by", 4), ("card_player", 3)):
                if m.get(k) and self.canon(m[k]) == pc:
                    bonus = max(bonus, b)
        conf = m.get("event_confidence")
        conf = conf if isinstance(conf, (int, float)) else 1.0
        return (base + bonus) * conf

    def player_stats(self, pc):
        # Counting stats come from the de-duplicated EVENT list (accurate), not frames.
        g = a = s = c = 0
        for e in self.events:
            ep, ea = self.canon(e.get("player")), self.canon(e.get("assist"))
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
        ap = sum(1 for m in self.moments if pc in {self.canon(n) for n in _mentions(m)})
        return {"goals": g, "assists": a, "shots": s, "cards": c, "appearances": ap}

    def player_moments(self, pc):
        scored = [(self.moment_importance(m, pc), m) for m in self.moments
                  if pc in {self.canon(n) for n in _mentions(m)}]
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored

    def nearest_image(self, t):
        return min(self.moments, key=lambda m: abs(m["t_seconds"] - t))["image"] if self.moments else None


# Load every clip that has an index on disk (skip any not yet built).
CLIP_CONF = [
    ("clip2", "Arsenal v Manchester United", "clip2.mp4"),
    ("seed", "Manchester United v Manchester City", "seed.mp4"),
]
CLIPS = {}
for _name, _label, _vid in CLIP_CONF:
    try:
        CLIPS[_name] = Clip(_name, _label, _vid)
    except FileNotFoundError:
        pass
CLIP_ORDER = [n for n, _, _ in CLIP_CONF if n in CLIPS]
DEFAULT_CLIP = CLIP_ORDER[0] if CLIP_ORDER else None


def get_clip(name):
    c = CLIPS.get(name) or CLIPS.get(DEFAULT_CLIP)
    if c is None:
        raise HTTPException(404, "No clip index loaded (expected data/clips/<name>/index.json)")
    return c


# ---- search helpers (clip-independent) -------------------------------------
def embed_query(text):
    r = client.embeddings.create(model=EMBED_MODEL, input=[text],
                                 encoding_format="float",
                                 extra_body={"input_type": "query"})
    return np.array(r.data[0].embedding, dtype=np.float32)


def rerank(query, passages):
    """Return {passage_index: relevance_score}. Raises on failure (caller falls back)."""
    r = httpx.post(f"{OPENROUTER_URL}/rerank",
                   headers={"Authorization": f"Bearer {OPENROUTER_KEY}", "Content-Type": "application/json"},
                   json={"model": RERANK_MODEL, "query": query,
                         "documents": passages, "top_n": len(passages)}, timeout=20)
    r.raise_for_status()
    return {res["index"]: float(res["relevance_score"]) for res in r.json()["results"]}


def _has_phrase(text, phrase):
    """Whole-word match with plural/-ing forms: "goals" hits goal, "goalkeeper" does not."""
    return re.search(rf"\b{re.escape(phrase)}(?:s|es|ing)?\b", text) is not None


def _passage(m):
    """Reranker input: what is on screen plus who is named (prose often only says '#10')."""
    names = sorted(_mentions(m))
    who = f" Players: {', '.join(names)}." if names else ""
    return f"{m.get('event', '')}. {m.get('prose', '')}{who}"


def keyword_score(m, terms):
    parts = [str(m.get(k, "")) for k in
             ("prose", "event", "phase", "zone_third", "zone_side", "pressure",
              "teams", "score", "card", "card_player", "scorer", "assist",
              "shot_by", "ocr_text", "commentary")]
    parts += m.get("players", []) or []
    hay = " ".join(parts).lower()
    return sum(1.0 for t in terms if _has_phrase(hay, t))


# When the query NAMES an event, hard-boost moments of that type so a literal
# term like "goal" surfaces actual goals, not everything goal-adjacent.
EVENT_SYNONYMS = {
    "goal": ["goal", "scored", "scores", "scoring"],
    "shot": ["shot", "shoots", "effort", "strike"],
    "save": ["save", "saves", "parry", "parried"],
    "corner": ["corner"], "cross": ["cross", "crossing", "crossed"],
    "foul": ["foul", "fouled"], "tackle": ["tackle", "tackled"],
    "yellow_card": ["yellow", "booking", "booked"],
    "red_card": ["red card", "sent off"], "penalty": ["penalty", "spot kick"],
    "free_kick": ["free kick", "free-kick"], "duel": ["duel", "aerial"],
    "offside": ["offside", "var"], "throw_in": ["throw in", "throw-in"],
    "celebration": ["celebration", "celebrate", "celebrates"],
}


def target_events(ql):
    return {ev for ev, syns in EVENT_SYNONYMS.items() if any(_has_phrase(ql, s) for s in syns)}


# Tactical vocabulary -> the descriptive language the captions actually use.
# Analysts search in tactical terms; captions describe what's on screen. Bridge them
# by appending a descriptive gloss to the query before embedding/reranking.
TACTICAL_EXPANSIONS = {
    "high press": "winning the ball back high up the pitch, pressing and closing down the opponent near their own goal",
    "press high": "pressing high up the pitch to win the ball near the opponent's goal",
    "gegenpress": "immediately winning the ball back after losing it, pressing right after a turnover",
    "counter press": "immediately winning the ball back after losing it, pressing right after a turnover",
    "counterpress": "immediately winning the ball back after losing it, pressing right after a turnover",
    "low block": "defending deep in a compact defensive shape near their own goal with many players behind the ball",
    "sitting deep": "defending deep near their own goal in a compact shape",
    "switch of play": "a long diagonal pass across to the far side, switching play to the opposite flank",
    "switch the play": "a long diagonal pass across to the far side, switching play to the opposite flank",
    "overload": "a numerical advantage with more attackers than defenders in a wide area",
    "line-breaking pass": "a forward pass that splits the defense and breaks through the defensive line",
    "line breaking": "a forward pass that splits the defense and breaks through the defensive line",
    "through ball": "a forward pass played through the defense into space behind the defenders",
    "build up from the back": "building play from the back, the goalkeeper and defenders passing out of their own third",
    "build-up from the back": "building play from the back, the goalkeeper and defenders passing out of their own third",
    "long ball": "a long ball played forward up the pitch",
    "cutback": "a ball pulled back from the byline across the penalty area",
    "pull back": "a ball pulled back from the byline across the penalty area",
    "one two": "a quick one-two passing combination",
    "give and go": "a quick one-two passing combination",
    "wing play": "attacking down the wing near the touchline",
    "down the wing": "attacking down the wing near the touchline",
    "third man run": "a runner arriving from deep behind the ball",
    "set piece": "a corner kick or free kick delivery into the box",
}


def expand_query(q):
    ql = q.lower()
    adds = [v for k, v in TACTICAL_EXPANSIONS.items() if _has_phrase(ql, k)]
    return (q + ". " + ". ".join(adds)) if adds else q


app = FastAPI(title="Soccer Situation Search")


@app.get("/api/clips")
def clips():
    return JSONResponse({
        "clips": [{"name": CLIPS[n].name, "label": CLIPS[n].label, "video": CLIPS[n].video}
                  for n in CLIP_ORDER],
        "default": DEFAULT_CLIP,
    })


@app.get("/api/players")
def players(clip: str = ""):
    c = get_clip(clip)
    out = []
    for cl in c.clusters:
        pc = cl["canonical"]
        st = c.player_stats(pc)
        weight = st["goals"] * 10 + st["assists"] * 7 + st["shots"] * 3 + st["cards"] * 2 + st["appearances"]
        out.append({"name": pc, "aliases": sorted(cl["variants"]), **st, "weight": weight})
    out.sort(key=lambda x: x["weight"], reverse=True)
    return JSONResponse({"players": out})


@app.get("/api/player")
def player(name: str, clip: str = ""):
    c = get_clip(clip)
    pc = c.canon(name) or _norm(name)
    moms = [public(m, imp) for imp, m in c.player_moments(pc)]
    return JSONResponse({"name": pc, "stats": c.player_stats(pc), "moments": moms})


@app.get("/api/events")
def events(clip: str = ""):
    return JSONResponse({"events": get_clip(clip).events})


@app.get("/api/timeline")
def timeline(clip: str = ""):
    """Video-native events for the timeline, with a thumbnail and goal confirmation.
    A 'goal' is confirmed only if the scorebug (commentary/OCR) corroborates it near
    that time; otherwise it is flagged unconfirmed rather than trusted."""
    c = get_clip(clip)
    out = []
    for e in c.video_events:
        t = e["t_seconds"]
        # Goals are detected from the video (works with no scoreboard). When the clip
        # has a scorebug, a goal it does not corroborate is a replay or a disallowed
        # goal, so it is demoted to a shot.
        corro = None
        if e.get("type") == "goal":
            corro = any(abs(t - g) <= 20 for g in c.confirmed_goal_ts)
            if c.confirmed_goal_ts and not corro:
                e = {**e, "type": "shot"}
        out.append({**e, "image": c.nearest_image(t), "scorebug_corroborated": corro})
    out.sort(key=lambda x: x["t_seconds"])
    # de-dup: overlapping chunks re-detect the same event. Collapse same type within
    # 4s; a goal is seen by both windows a few seconds apart, so within 15s.
    deduped, last = [], {}
    for e in out:
        prev = last.get(e["type"])
        if prev is not None and e["t_seconds"] - prev <= (15 if e["type"] == "goal" else 4):
            continue
        last[e["type"]] = e["t_seconds"]
        deduped.append(e)
    dur = max((m["t_seconds"] for m in c.moments), default=0) + 10
    return JSONResponse({"events": deduped, "duration": dur,
                         "confirmed_goals": len(c.confirmed_goal_ts)})


@app.get("/api/moments")
def moments(clip: str = ""):
    c = get_clip(clip)
    return JSONResponse({"moments": [public(m) for m in c.moments], "video": c.video})


@app.get("/api/search")
def search(q: str = "", event: str = "", phase: str = "", clip: str = ""):
    c = get_clip(clip)
    M = c.moments
    idx = list(range(len(M)))
    if event:
        idx = [i for i in idx if M[i].get("event") == event]
    if phase:
        idx = [i for i in idx if M[i].get("phase") == phase]

    ql = q.lower()
    terms = [t for t in ql.split() if t]
    targets = target_events(ql)
    q_search = expand_query(q)  # append a descriptive gloss for tactical terms
    semantic_ok = False
    sims = np.zeros(len(M), dtype=np.float32)
    if q.strip() and c.dim:
        try:
            qv = embed_query(q_search)
            qv = qv / max(np.linalg.norm(qv), 1e-9)
            sims = c.matn @ qv
            semantic_ok = True
        except Exception:
            semantic_ok = False

    scored = []
    for i in idx:
        kw = keyword_score(M[i], terms) if terms else 0.0
        m = M[i]
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
        passages = [_passage(M[i]) for i in cand]
        try:
            rs = rerank(q_search, passages)
            reranked = sorted(((rs.get(k, -1e9), cand[k]) for k in range(len(cand))), reverse=True)
            best = reranked[0][0] if reranked else -1e9
            if best < NO_MATCH_THRESHOLD:
                return JSONResponse({"results": [], "no_match": True, "query": q,
                                     "ranking": "reranked", "count": 0})
            cutoff = max(NO_MATCH_THRESHOLD, best - SHOW_GAP)  # relative: keep strong queries tight
            kept = []
            for s, i in reranked:
                if s < cutoff:
                    continue
                t, ev = M[i]["t_seconds"], M[i].get("event")
                # dense sampling -> collapse a run of the same event into its best moment
                if any(ev == M[j].get("event") and abs(t - M[j]["t_seconds"]) < DEDUP_S
                       for _, j in kept):
                    continue
                kept.append((s, i))
            results = [public(M[i], s) for s, i in kept[:40]]
            return JSONResponse({"results": results, "ranking": "reranked", "count": len(results)})
        except Exception:
            pass  # reranker unavailable -> fall back to cosine ordering

    if q.strip() and not semantic_ok:
        scored = [(s, i) for s, i in scored if s > 0]
    results = [public(M[i], s) for s, i in scored[:40]]
    return JSONResponse({"results": results,
                         "ranking": "semantic" if semantic_ok else "keyword",
                         "count": len(results)})


@app.get("/")
def index():
    # no-store so the browser never serves a stale UI after we change it
    return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-store"})


# Serve only per-clip media; transcripts, audio and working files elsewhere in data/ stay private.
app.mount("/media/clips", StaticFiles(directory=DATA / "clips", check_dir=False), name="media")
app.mount("/static", StaticFiles(directory=STATIC), name="static")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8008, log_level="warning")
