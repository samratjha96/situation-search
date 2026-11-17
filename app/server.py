"""Situation Search backend. Serves search over a vision-built match index.

Everything the app knows about a match comes from vision models, processed by a
fixed pipeline (see pipeline/): an action log of every on-ball action, a roster
read off the shirts, goals from the scoreboard, and set pieces, switches of play
and counter-attacks derived from the actions. No commentary, no captions.

A query is answered in one of three ways:
  1. a player's name alone   -> his reel: every snippet where he is on the ball
  2. football vocabulary     -> structured filters on the action log (action,
                                set piece, pattern, goal, body shape, outcome,
                                team, player), ordered by the reranker
  3. anything else           -> semantic search over the vision model's action
                                descriptions, reranked; below the clip's
                                calibrated cutoff the answer is "no match"
"""
import json
import os
import re
from collections import Counter
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

OPENROUTER_URL = os.environ.get("OPENROUTER_URL", "https://openrouter.ai/api/v1")
OPENROUTER_KEY = os.environ.get("OPENROUTER_API_KEY", "")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "openai/text-embedding-3-small")
RERANK_MODEL = os.environ.get("RERANK_MODEL", "cohere/rerank-v3.5")
client = OpenAI(base_url=OPENROUTER_URL, api_key=OPENROUTER_KEY)

NO_MATCH_THRESHOLD = float(os.environ.get("NO_MATCH_THRESHOLD", "-4.0"))  # per-clip search.json overrides
SHOW_GAP = float(os.environ.get("SHOW_GAP", "5.0"))  # semantic results within this of the best
RERANK_K = 40
REEL_PAD = (3.0, 4.0)  # seconds before/after each of the player's actions (the finish lands after the touch)
REEL_GAP_S = 2.0       # snippets closer than this merge

# Query vocabulary -> structured filters. Longest phrases are matched first.
ACTION_TERMS = {
    "receive": ["receive", "receives", "receiving", "received", "reception", "first touch", "controls"],
    "pass": ["pass", "passes", "passing"], "cross": ["cross", "crosses", "crossing", "crossed"],
    "shot": ["shot", "shots", "shoot", "shoots", "shooting", "strike", "effort"],
    "header": ["header", "headers", "headed", "heading"],
    "dribble": ["dribble", "dribbles", "dribbling", "carry", "carries", "run with the ball"],
    "tackle": ["tackle", "tackles", "tackling"], "interception": ["interception", "intercept", "intercepts"],
    "clearance": ["clearance", "clearances", "clears", "cleared"],
    "save": ["save", "saves", "diving save", "goalkeeper save"], "foul": ["foul", "fouls", "fouled"],
    "throw_in": ["throw in", "throw-in", "throw-ins"], "goal_kick": ["goal kick", "goal kicks"],
}
SET_PIECE_TERMS = {"corner": ["corner", "corners", "corner kick"],
                   "free_kick": ["free kick", "free kicks", "free-kick"], "penalty": ["penalty", "penalties"]}
PATTERN_TERMS = {"switch_of_play": ["switch of play", "switch play", "switches of play", "switch the play"],
                 "counter_attack": ["counter attack", "counter-attack", "counterattack", "counter attacks",
                                    "counter-attacks", "on the break"]}
GOAL_TERMS = ["goal", "goals", "scores", "scored", "scoring"]
BODY_TERMS = {"half_turn": ["half turn", "half-turn", "half turned", "half-turned"],
              "open": ["open body", "facing forward", "facing goal", "opened up"],
              "back_to_goal": ["back to goal", "back to the goal", "with his back"]}
OUTCOME_TERMS = {"on_target": ["on target", "on goal"], "off_target": ["off target", "wide", "misses", "missed"],
                 "blocked": ["blocked"], "saved": ["saved"], "intercepted": ["intercepted"]}
LED_TO = {"goal": ["led to a goal", "lead to a goal", "leading to a goal", "resulted in a goal", "scored from"],
          "shot": ["led to a shot", "lead to a shot", "leading to a shot", "resulted in a shot"]}
STOPWORDS = set("a an the in on of to for from by at and or with into is are that this it show me find all any "
                "moment moments where when which who his her their ball player players team box area".split())
NOTABLE = {"goal", "shot", "header", "save", "corner", "free_kick", "penalty", "foul"}


def tc(s):
    return f"{int(s) // 60:02d}:{int(s) % 60:02d}"


def phrase(ql, p):
    return re.search(rf"(?<![a-z]){re.escape(p)}(?![a-z])", ql) is not None


def matched(ql, table):
    return {k for k, ps in table.items() if any(phrase(ql, p) for p in ps)}


class Clip:
    def __init__(self, name):
        d = DATA / "clips" / name
        idx = json.loads((d / "index.json").read_text())
        self.name, self.prefix = name, f"clips/{name}/"
        self.video = self.prefix + idx["video"]
        self.duration = idx["duration"]
        self.teams = [t["name"] for t in idx["teams"]]
        self.label = " v ".join(self.teams)
        self.actions = idx["actions"]
        self.by_id = {a["id"]: a for a in self.actions}
        self.episodes, self.patterns, self.goals = idx["episodes"], idx["patterns"], idx["goals"]
        self.frames = [self.prefix + f for f in idx["frames"]]
        self.interval = idx["frame_interval_s"]
        for a in self.actions:
            a["image"] = self.prefix + a["image"] if a.get("image") else None
        self.players = sorted({a["player"] for a in self.actions if a["player"]} |
                              {g["scorer"] for g in self.goals if g["scorer"]})
        ep = d / "embeddings.json"
        vecs = json.loads(ep.read_text())["vectors"] if ep.exists() else {}
        self.emb_ids = [a["id"] for a in self.actions if str(a["id"]) in vecs]
        m = np.array([vecs[str(i)] for i in self.emb_ids], dtype=np.float32).reshape(len(self.emb_ids), -1)
        self.emb = m / np.clip(np.linalg.norm(m, axis=1, keepdims=True), 1e-9, None) if len(m) else m
        sp = d / "search.json"
        self.no_match = json.loads(sp.read_text())["no_match_threshold"] if sp.exists() else NO_MATCH_THRESHOLD

    def image_at(self, t):
        return self.frames[min(max(int(t // self.interval), 0), len(self.frames) - 1)] if self.frames else None

    def player_in(self, ql):
        for p in self.players:
            if p.lower() in ql or any(len(w) >= 4 and phrase(ql, w.lower()) for w in p.split()):
                return p
        return None

    def team_in(self, ql):
        return next((t for t in self.teams if phrase(ql, t.lower())), None)

    def reel(self, player):
        ts = sorted({a["t"] for a in self.actions if a["player"] == player} |
                    {g["t"] for g in self.goals if g["scorer"] == player})
        segs = []
        for t in ts:
            a, b = max(t - REEL_PAD[0], 0.0), t + REEL_PAD[1]
            if segs and a <= segs[-1][1] + REEL_GAP_S:
                segs[-1][1] = b
            else:
                segs.append([a, b])
        return segs


CLIPS = {p.parent.name: Clip(p.parent.name) for p in sorted((DATA / "clips").glob("*/index.json"))
         if "actions" in json.loads(p.read_text())}
CLIP_ORDER = sorted(CLIPS, key=lambda n: CLIPS[n].label)
DEFAULT_CLIP = os.environ.get("DEFAULT_CLIP") or (CLIP_ORDER[0] if CLIP_ORDER else None)


def get_clip(name):
    c = CLIPS.get(name) or CLIPS.get(DEFAULT_CLIP)
    if c is None:
        raise HTTPException(404, "No clip index loaded")
    return c


def who(a):
    return a["player"] or (f"{a['team']} #{a['number']}" if a["number"] else f"{a['team'] or 'unknown'} player")


def action_text(a):
    return f"{a['action'].replace('_', ' ')} by {who(a)}. {a['detail']}"


def action_row(c, a, rel=None):
    extra = []
    if a["action"] == "receive" and a.get("body_shape") not in (None, "unclear"):
        extra.append(a["body_shape"].replace("_", "-"))
    if a.get("outcome") not in (None, "none"):
        extra.append(a["outcome"].replace("_", " "))
    return {"t_seconds": a["t"], "timecode": a["timecode"], "event": a["action"], "image": a["image"],
            "prose": f"{who(a)}{' (' + ', '.join(extra) + ')' if extra else ''}. {a['detail']}",
            "players": [a["player"]] if a["player"] else [], "identity_source": a["identity_source"],
            "relevance": rel}


def span_row(c, kind, start, end, prose, outcome=None, scorer=None):
    return {"t_seconds": start, "timecode": tc(start), "episode_end": end, "event": kind, "outcome": outcome,
            "image": c.image_at(start + 2), "prose": prose, "scorer": scorer}


def rerank(query, docs):
    r = httpx.post(f"{OPENROUTER_URL}/rerank", headers={"Authorization": f"Bearer {OPENROUTER_KEY}"},
                   json={"model": RERANK_MODEL, "query": query, "documents": docs, "top_n": len(docs)}, timeout=20)
    r.raise_for_status()
    return {x["index"]: float(x["relevance_score"]) for x in r.json()["results"]}


def order(q, rows, texts):
    """Order structured hits by reranker relevance; keep time order if it is down."""
    if len(rows) < 2:
        return rows
    try:
        rs = rerank(q, texts)
        return [rows[i] for i in sorted(range(len(rows)), key=lambda i: -rs.get(i, -1e9))]
    except Exception:
        return rows


def semantic(c, q):
    """Best-first [(score, action)] over the vision descriptions; raises if the endpoint is down."""
    if not len(c.emb):
        return []
    r = client.embeddings.create(model=EMBED_MODEL, input=[q], encoding_format="float",
                                 extra_body={"input_type": "query"})
    qv = np.array(r.data[0].embedding, dtype=np.float32)
    sims = c.emb @ (qv / max(np.linalg.norm(qv), 1e-9))
    cand = [c.by_id[c.emb_ids[i]] for i in np.argsort(-sims)[:RERANK_K]]
    rs = rerank(q, [action_text(a) for a in cand])
    return sorted(((rs.get(i, -1e9), a) for i, a in enumerate(cand)), key=lambda x: -x[0])


def respond(results, ranking, **kw):
    return JSONResponse({"results": results, "ranking": ranking, "count": len(results),
                         "no_match": not results, **kw})


def search_clip(c, q):
    ql = q.lower().strip()
    player, team = c.player_in(ql), c.team_in(ql)
    words = [w for w in re.findall(r"[a-z'-]+", ql) if w not in STOPWORDS]

    # 1. a name alone: the player's reel. Shirts carry surnames, so one extra word
    # (a first name: "Kobbie Mainoo") is allowed.
    vocab = (ACTION_TERMS, SET_PIECE_TERMS, PATTERN_TERMS, BODY_TERMS, OUTCOME_TERMS, LED_TO)
    others = [w for w in words if w not in player.lower().split()] if player else []
    if player and len(others) <= 1 and not any(matched(ql, t) for t in vocab) \
            and not any(phrase(ql, g) for g in GOAL_TERMS):
        segs = c.reel(player)
        rows = []
        for s, e in segs:
            acts = [a for a in c.actions if a["player"] == player and s <= a["t"] <= e]
            prose = " ".join(action_text(a) for a in acts[:3]) or f"{player} scores."
            rows.append(span_row(c, acts[0]["action"] if acts else "goal", s, e, prose))
        return respond(rows, "reel", reel=segs, player=player)

    sp, pat = matched(ql, SET_PIECE_TERMS), matched(ql, PATTERN_TERMS)
    acts, led = matched(ql, ACTION_TERMS), matched(ql, LED_TO)
    shapes, outcomes = matched(ql, BODY_TERMS), matched(ql, OUTCOME_TERMS)
    wants_goal = any(phrase(ql, g) for g in GOAL_TERMS) and not led and "goal_kick" not in acts \
        and not phrase(ql, "on goal")  # "shot on goal" = on target, not scored

    # 2a. set pieces as episodes, optionally filtered on what they led to
    if sp:
        eps = [e for e in c.episodes if e["type"] in sp and (not team or e["team"] == team)
               and (not led or e["outcome"] in led) and (not player or e["taker"] == player)]
        rows = [span_row(c, e["type"], e["start"], e["end"],
                         " ".join(action_text(c.by_id[i]) for i in e["action_ids"][:3]), e["outcome"])
                for e in eps]
        return respond(order(q, rows, [r["prose"] for r in rows]), "set pieces")

    # 2b. derived patterns
    if pat:
        ps = [p for p in c.patterns if p["type"] in pat and (not team or p["team"] == team)
              and (not player or any(c.by_id[i]["player"] == player for i in p["action_ids"]))]
        rows = [span_row(c, p["type"], p["start"], p["end"],
                         " ".join(action_text(c.by_id[i]) for i in p["action_ids"][:3])) for p in ps]
        return respond(order(q, rows, [r["prose"] for r in rows]), "patterns")

    # 2c. goals from the scoreboard
    if wants_goal and not acts:
        gs = [g for g in c.goals if (not team or g["team"] == team) and (not player or g["scorer"] == player)]
        rows = [span_row(c, "goal", max(g["t"] - 8, 0), g["t"] + 4,
                         f"{g['team']} goal" + (f" by {g['scorer']}. " if g["scorer"] else ". ") +
                         (action_text(c.by_id[g["action_id"]]) if g["action_id"] is not None else ""),
                         scorer=g["scorer"]) for g in gs]
        return respond(rows, "goals")

    # 2d. actions with modifiers
    if acts or shapes:
        if shapes:
            acts = acts | {"receive"}
        if wants_goal:
            outcomes = outcomes | {"goal"}
        if "on_target" in outcomes:  # a shot on target is saved, scores, or is logged as on target
            outcomes = outcomes | {"saved", "goal"}
        hits = [a for a in c.actions if a["action"] in acts and (not team or a["team"] == team)
                and (not player or a["player"] == player)
                and (not shapes or a.get("body_shape") in shapes)
                and (not outcomes or a.get("outcome") in outcomes)]
        rows = order(q, [action_row(c, a) for a in hits], [action_text(a) for a in hits])
        return respond(rows, "actions")

    # 3. free text (a named player or team still filters) over the vision descriptions
    try:
        ranked = semantic(c, q)
    except Exception:
        return respond([], "unavailable", error="search endpoint unreachable")
    if not ranked or ranked[0][0] < c.no_match:
        return respond([], "semantic")
    cutoff = max(c.no_match, ranked[0][0] - SHOW_GAP)
    rows = [action_row(c, a, s) for s, a in ranked if s >= cutoff and (not player or a["player"] == player)
            and (not team or a["team"] == team)]
    return respond(rows, "semantic")


app = FastAPI(title="Situation Search")


@app.get("/api/clips")
def clips():
    return JSONResponse({"clips": [{"name": n, "label": CLIPS[n].label, "video": CLIPS[n].video}
                                   for n in CLIP_ORDER], "default": DEFAULT_CLIP})


@app.get("/api/clip")
def clip(clip: str = ""):
    c = get_clip(clip)
    return JSONResponse({"video": c.video, "teams": c.teams})


@app.get("/api/timeline")
def timeline(clip: str = ""):
    c = get_clip(clip)
    ev = [{"t_seconds": a["t"], "timecode": a["timecode"], "type": a["action"], "key_player": who(a),
           "detail": a["detail"], "confidence": a["confidence"], "image": a["image"]}
          for a in c.actions if a["action"] in NOTABLE and a["action"] != "goal"]
    ev += [{"t_seconds": g["t"], "timecode": tc(g["t"]), "type": "goal", "key_player": g["scorer"] or g["team"],
            "detail": f"{g['team']} goal" + (f" by {g['scorer']}" if g["scorer"] else ""),
            "confidence": 1.0, "image": c.image_at(g["t"]), "scorebug_corroborated": True} for g in c.goals]
    ev.sort(key=lambda e: e["t_seconds"])
    return JSONResponse({"events": ev, "duration": c.duration, "confirmed_goals": len(c.goals)})


@app.get("/api/players")
def players(clip: str = ""):
    c = get_clip(clip)
    out = []
    for p in c.players:
        mine = [a for a in c.actions if a["player"] == p]
        st = {"goals": sum(1 for g in c.goals if g["scorer"] == p), "assists": 0,
              "shots": sum(1 for a in mine if a["action"] in ("shot", "header")), "cards": 0,
              "appearances": len(mine)}
        out.append({"name": p, **st, "weight": st["goals"] * 10 + st["shots"] * 3 + st["appearances"]})
    out.sort(key=lambda x: -x["weight"])
    return JSONResponse({"players": out})


@app.get("/api/player")
def player(name: str, clip: str = ""):
    c = get_clip(clip)
    res = json.loads(search_clip(c, name).body)
    st = next((p for p in json.loads(players(clip).body)["players"] if p["name"] == res.get("player")), {})
    return JSONResponse({"name": res.get("player") or name, "stats": st, "moments": res["results"],
                         "reel": res.get("reel", [])})


@app.get("/api/search")
def search(q: str = "", clip: str = ""):
    c = get_clip(clip)
    if not q.strip():
        return respond([action_row(c, a) for a in c.actions if a["action"] in NOTABLE], "browse")
    return search_clip(c, q)


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-store"})


app.mount("/media/clips", StaticFiles(directory=DATA / "clips", check_dir=False), name="media")
app.mount("/static", StaticFiles(directory=STATIC), name="static")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8008, log_level="warning")
