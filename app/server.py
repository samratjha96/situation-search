"""Local search over indexed football actions and timestamped passages."""
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from .query import FILLER, normalize, parse, vocabulary

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
STATIC = ROOT / "app" / "static"

REEL_PAD = (3.0, 4.0)  # seconds before/after each of the player's actions (the finish lands after the touch)
REEL_GAP_S = 2.0       # snippets closer than this merge

NOTABLE = {"goal", "shot", "header", "save", "corner", "free_kick", "penalty", "foul"}


def tc(s):
    return f"{int(s) // 60:02d}:{int(s) % 60:02d}"


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
        self.attempt_saves = {int(shot): save for shot, save in idx["attempt_saves"].items()}
        self.by_id = {a["id"]: a for a in self.actions}
        self.actions_by_type = defaultdict(list)
        for a in self.actions:
            self.actions_by_type[a["action"]].append(a)
        self.episodes, self.patterns, self.goals = idx["episodes"], idx["patterns"], idx["goals"]
        self.frames = [self.prefix + f for f in idx["frames"]]
        self.frame_files = set(self.frames)
        self.interval = idx["frame_interval_s"]
        for a in self.actions:
            a["image"] = self.prefix + a["image"] if a.get("image") else None
        self.players = sorted({a["player"] for a in self.actions if a["player"]} |
                              {g["scorer"] for g in self.goals if g["scorer"]})
        self.lexicon = vocabulary(self.teams, set(idx["roster"].values()) | set(self.players))
        self.descriptions = [(a, set(normalize(action_text(a)).split())) for a in self.actions]

    def image_at(self, t):
        return self.frames[min(max(int(t // self.interval), 0), len(self.frames) - 1)] if self.frames else None

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


def get_clip(name):
    c = CLIPS.get(name or DEFAULT_CLIP)
    if c is None:
        raise HTTPException(404, "No clip index loaded")
    return c


def who(a):
    return a["player"] or (f"{a['team']} #{a['number']}" if a["number"] else f"{a['team'] or 'unknown'} player")


def action_text(a):
    return f"{a['action'].replace('_', ' ')} by {who(a)}. {a['detail']}"


def action_row(c, a):
    extra = []
    if a["action"] == "receive" and a.get("body_shape") not in (None, "unclear"):
        extra.append(a["body_shape"].replace("_", "-"))
    if a.get("outcome") not in (None, "none"):
        extra.append(a["outcome"].replace("_", " "))
    return {"episode_end": min(c.duration, a["t"] + (3.5 if a["action"] == "cross" else 2)),
            "evidence": [{"t": a["t"], "detail": a["detail"]}], "t_seconds": a["t"], "timecode": a["timecode"], "event": a["action"], "image": a["image"],
            "prose": f"{who(a)}{' (' + ', '.join(extra) + ')' if extra else ''}. {a['detail']}",
            "players": [a["player"]] if a["player"] else [], "identity_source": a["identity_source"],
            "relevance": None}


def span_row(c, kind, start, end, prose, outcome=None, scorer=None):
    return {"t_seconds": start, "timecode": tc(start), "episode_end": end, "event": kind, "outcome": outcome,
            "image": c.image_at(start + 2), "prose": prose, "scorer": scorer}


def respond(results, ranking, **fields):
    status = fields.pop("status", "verified")
    return {"results": results, "ranking": ranking, "count": len(results), "status": status,
            "no_match": status == "verified" and not results, **fields}


def verified(rows, ranking, candidate_spans):
    return respond(rows, ranking, candidate_spans=candidate_spans)


def suggestions(c, q, plan, reason):
    # Description hits are reading aids. They never become verified matches.
    terms = set(plan.unknown or normalize(q).split()) - FILLER
    player = plan.one("player")
    team = plan.one("team")
    rows = []
    if terms:
        for action, words in c.descriptions:
            if terms <= words and (not team or team == action["team"]) and (not player or player == action["player"]):
                rows.append(action_row(c, action))
    return respond(rows, "description candidates", status="unverified", message=reason)


def player_reel(c, player):
    segs = c.reel(player)
    rows = []
    for start, end in segs:
        actions = [a for a in c.actions if a["player"] == player and start <= a["t"] <= end]
        prose = " ".join(action_text(a) for a in actions[:3]) or f"{player} scores."
        row = span_row(c, actions[0]["action"] if actions else "goal", start, min(end, c.duration), prose)
        row["evidence"] = [{"t": a["t"], "detail": action_text(a)} for a in actions]
        rows.append(row)
    return respond(rows, "reel", player=player, reel=segs)


def matches_fields(a, plan):
    for key in ("pressure", "length", "shape", "origin", "destination", "style"):
        value = plan.one(key)
        if not value:
            continue
        if key == "style" and not re.search(r"\bdiv(?:e|es|ed|ing)\b", a["detail"], re.I):
            return False
        if key == "shape" and a.get("body_shape") != value:
            return False
        if key == "length" and a.get("length") != value:
            return False
        if key == "pressure":
            pressures = {"tight", "light"} if value == "any" else {value}
            if a.get("pressure") not in pressures:
                return False
        start, end = a.get("start") or {}, a.get("end") or {}
        if key == "origin" and start.get("side") != value:
            return False
        if key == "destination":
            if value in {"box", "opposite_box"} and not re.search(
                    r"penalty (area|box)|six.yard|into the box|(?:near|far|back) post", a["detail"], re.I):
                return False
            if value == "opposite_box":
                if {start.get("side"), end.get("side")} != {"left", "right"} or end.get("third") != "att":
                    return False
            elif value != "box" and end.get("side") != value:
                return False
    return True


def outcome_matches(actual, wanted, negate=False, has_shot=False):
    if not wanted:
        return True
    if not actual or actual in {"unknown", "none", "unclear"}:
        return False
    if wanted == "shot":
        match = has_shot
    elif wanted == "on_target":
        match = actual in {"save", "saved", "on_target", "goal"}
    elif wanted == "saved":
        match = actual in {"save", "saved"}
    else:
        match = actual == wanted
    return not match if negate else match


def step_matches(e, step):
    if step in {"regain", "progression"}:
        return e.get("stage") == step
    if step == "shot":
        return e.get("stage") == "shot" or e.get("action") == "shot"
    return e.get("action") == step


def passage_observations(c, passage):
    observations = list(passage["evidence"])
    represented = {e.get("action_id") for e in observations}
    for action_id in passage["action_ids"]:
        if action_id not in represented:
            a = c.by_id[action_id]
            observations.append(dict(action_id=a["id"], **{k: a[k] for k in ("t", "action", "detail", "team", "player")}))
    return sorted(observations, key=lambda e: e["t"])


def step_witnesses(observations, team, steps, step_players, attempt_saves):
    # Select a linked pair together so an earlier, unrelated header cannot hide it.
    witnesses = []
    for i, step in enumerate(steps):
        last = witnesses[-1] if witnesses else None
        needs_save = step in {"header", "shot"} and steps[i+1:i+2] == ["save"]
        linked_to = (attempt_saves[last["action_id"]] if i and step == "save"
                     and steps[i-1] in {"header", "shot"} else None)
        witness = next((e for e in observations if (not last or e["t"] > last["t"])
                        and step_matches(e, step)
                        and (not needs_save or e.get("action_id") in attempt_saves)
                        and (linked_to is None or e.get("action_id") == linked_to)
                        and (step not in step_players or e.get("player") == step_players[step])
                        and (e.get("team") == team if step != "save" else e.get("team") and e["team"] != team)), None)
        if witness is None:
            return None
        witnesses.append(witness)
    return witnesses


def passage_search(c, q, plan):
    pattern, piece, steps = plan.one("pattern"), plan.one("piece"), plan.values("step")
    if pattern and piece:
        return suggestions(c, q, plan, "The index cannot verify a relationship between these two passages.")
    if piece:
        candidates = [p for p in c.episodes if piece in (p["type"], "set_piece")]
    elif pattern:
        candidates = [p for p in c.patterns if p["type"] == pattern]
    else:
        candidates = [p for p in c.patterns if p["type"] in {"regain_attack", "counter_attack"}]
    team, wanted = plan.one("team"), plan.one("result")
    rel = plan.values("relation")
    delivery_fields = any(plan.one(k) for k in ("shape", "pressure", "length", "origin", "destination", "style"))
    if "open_play" in rel or (delivery_fields and (not piece or steps or wanted)):
        return suggestions(c, q, plan, "The requested passage modifier needs additional evidence.")
    negate = "neg" in rel
    if negate:
        pos = plan.tags.index(("relation", "neg"))
        if any(k == "result" for k, _ in plan.tags[:pos]):
            return suggestions(c, q, plan, "The combination of a required outcome and a separate exclusion needs more evidence.")
        # Only an explicitly scoped terminal exclusion is supported.
        excluded = [(k, v) for k, v in plan.tags[pos+1:] if k in ("step", "result", "piece", "pattern", "player", "team", "saving_team")]
        if len(excluded) != 1 or excluded[0][0] not in ("step", "result") or excluded[0][1] not in ("goal", "shot", "save", "saved"):
            return suggestions(c, q, plan, "The scope of this exclusion could not be verified.")
        _, wanted = excluded[0]
        if wanted in steps:
            steps = steps.copy()
            steps.remove(wanted)
        wanted = "saved" if wanted == "save" else wanted
    ordered = [(k, v) for k, v in plan.tags if k in ("piece", "pattern", "step", "result", "relation")]
    # A shot after a regain reverses surface order. Normalize binary 'after'.
    if ("relation", "after") in ordered:
        split = ordered.index(("relation", "after"))
        ordered = ordered[split+1:] + ordered[:split]
    stages = [(k, v) for k, v in ordered if k in ("piece", "pattern", "step", "result")]
    if stages and (piece or pattern) and stages[0][0] not in ("piece", "pattern"):
        return suggestions(c, q, plan, "The requested ordering does not start with the selected passage.")
    if not piece and not pattern and ("step", "regain") in stages and stages[0] != ("step", "regain"):
        return suggestions(c, q, plan, "The indexed passage begins at the regain. Earlier actions need separate verification.")
    if not negate:
        steps = [v for k, v in stages if k == "step"]
        if stages[-1:] == [("step", "save")] and "terminal" in rel:
            wanted = "saved"
        if ("result", wanted) in stages and any(k == "step" for k, v in stages[stages.index(("result", wanted))+1:]):
            return suggestions(c, q, plan, "The requested events after the outcome need separate verification.")
    rows = []
    for p in candidates:
        if team and p["team"] != team:
            continue
        if delivery_fields and not matches_fields(c.by_id[p["action_ids"][0]], plan):
            continue
        observations = passage_observations(c, p)
        if plan.one("saving_team") and not any(e["action"] == "save" and e.get("team") == plan.one("saving_team")
                                               and e["team"] != p["team"] for e in observations):
            continue
        roles = {"taker": p.get("taker"), "shooter": p.get("shooter"), "scorer": p.get("scorer")}
        actors = {e.get("player") for e in observations if e.get("team") == p["team"]}
        keepers = {e.get("player") for e in observations if e["action"] == "save" and e.get("team") != p["team"]}
        participants = {role: {name} for role, name in roles.items()}
        participants.update(actor=actors, keeper=keepers)
        if any(name not in participants[role] for role, name in plan.role_players.items()):
            continue
        witnesses = step_witnesses(observations, p["team"], steps, plan.step_players, c.attempt_saves)
        if witnesses is None:
            continue
        has_shot = any(step_matches(e, "shot") and e.get("team") == p["team"] for e in observations)
        outcome = p.get("outcome")
        if outcome == "shot" and wanted in {"on_target", "off_target", "blocked"}:
            shot = next((e for e in p["evidence"] if e.get("stage") == "shot"), None)
            outcome = c.by_id[shot["action_id"]]["outcome"] if shot else None
        if negate and wanted == "saved" and "terminal" not in rel:
            if any(e["action"] == "save" and e.get("team") != p["team"] for e in observations):
                continue
        if not outcome_matches(outcome, wanted, negate, has_shot):
            continue
        row = span_row(c, p["type"], p["start"], p["end"], f"{p['team']} {p['type'].replace('_', ' ')}.",
                       outcome=p.get("outcome"), scorer=p.get("scorer"))
        row["evidence"] = sorted(p["evidence"] + [e for e in witnesses if e not in p["evidence"]], key=lambda e: e["t"])
        rows.append(row)
    return verified(rows, "passages", [[p["start"], p["end"]] for p in candidates])


def goal_search(c, q, plan):
    if any(role != "scorer" and role != "actor" for role in plan.role_players) or plan.values("step") or any(plan.one(k) for k in ("shape", "length", "pressure", "origin", "destination", "style")):
        return suggestions(c, q, plan, "The requested goal conditions need additional evidence.")
    team, player = plan.one("team"), plan.one("player")
    rows, candidates = [], []
    for g in c.goals:
        start, end = max(g["t"]-8, 0), min(g["t"]+4, c.duration)
        candidates.append([start, end])
        if team and g["team"] != team or player and g["scorer"] != player:
            continue
        if "open_play" in plan.values("relation"):
            # Verify a same-team on-ball build-up near the strike. Exclude linked restarts.
            if any(p["team"] == g["team"] and p["start"] <= g["t"] <= p["end"] for p in c.episodes):
                continue
            recent = [a for a in c.actions if start <= a["t"] < g["t"]]
            if not any(a["team"] == g["team"] and a["action"] in {"pass","dribble","receive"} for a in recent):
                continue
        action = c.by_id.get(g["action_id"])
        prose = f"{g['team']} goal" + (f" by {g['scorer']}." if g["scorer"] else ".")
        row = span_row(c, "goal", start, end, prose, "goal", g["scorer"])
        row["evidence"] = [{"t": g["t"], "detail": action_text(action) if action else prose}]
        if g.get("scoreboard_t") is not None:
            row["evidence"].append({"t": g["scoreboard_t"], "detail": "The scoreboard records the goal."})
        rows.append(row)
    return verified(rows, "goals", candidates)


def linked_save_search(c, plan, attempt):
    rows, candidates = [], []
    for shot_id, save_id in c.attempt_saves.items():
        shot, save = c.by_id[shot_id], c.by_id[save_id]
        if attempt == "header" and shot["action"] != "header":
            continue
        start, end = max(0, shot["t"] - 2), min(c.duration, save["t"] + 2)
        candidates.append([start, end])
        shooter = (plan.role_players.get("shooter") or plan.step_players.get(attempt)
                   or plan.role_players.get("actor"))
        if ((plan.one("team") and plan.one("team") != shot["team"]) or
                (plan.one("saving_team") and plan.one("saving_team") != save["team"]) or
                (shooter and shooter != shot["player"]) or
                (plan.role_players.get("keeper") and plan.role_players["keeper"] != save["player"])):
            continue
        row = span_row(c, "sequence", start, end, f'{save["team"]} saved the {shot["team"]} {shot["action"]}.',
                       outcome="save")
        row["evidence"] = [{k: a[k] for k in ("t", "detail", "team", "action", "player")}
                           for a in (shot, save)]
        rows.append(row)
    return verified(rows, "sequence", candidates)


def action_search(c, q, plan):
    steps = plan.values("step") or ["receive"]
    wanted = plan.one("result")
    team, player = plan.one("team"), plan.one("player")
    ordered = steps[::-1] if "after" in plan.values("relation") else steps
    saved_attempt = (ordered in (["header"], ["shot"]) and wanted == "saved") or (
        ordered in (["header", "save"], ["shot", "save"]) and not wanted)
    if saved_attempt:
        if any(plan.one(k) for k in ("shape", "length", "pressure", "origin", "destination", "style")):
            return suggestions(c, q, plan, "The requested modifiers need separate verification.")
        return linked_save_search(c, plan, ordered[0])
    if len(steps) > 1:
        if not plan.one("relation") or len(steps) != 2 or wanted or any(plan.one(k) for k in ("shape","length","pressure","origin","destination","style")):
            return suggestions(c, q, plan, "The requested action sequence needs additional evidence.")
        if "after" in plan.values("relation"):
            steps = steps[::-1]
        required_players = dict(plan.step_players)
        if "shooter" in plan.role_players:
            required_players["shot"] = plan.role_players["shooter"]
        if "keeper" in plan.role_players:
            required_players["save"] = plan.role_players["keeper"]
        if plan.one("saving_team"):
            return suggestions(c, q, plan, "The saving team needs an attempt linked to its save.")
        rows, candidates = [], []
        for a, b in zip(c.actions, c.actions[1:]):
            if a["action"] != steps[0] or b["action"] != steps[1] or not 0 < b["t"]-a["t"] <= 4:
                continue
            candidates.append([a["t"], b["t"] + 1])
            if not a["team"] or not b["team"] or team and team != a["team"]:
                continue
            if a["team"] != b["team"]:
                continue
            if any(required_players[s] != x["player"] for s, x in zip(steps, (a, b)) if s in required_players):
                continue
            if plan.role_players.get("actor") and not all(x["player"] == plan.role_players["actor"] for x in (a,b)):
                continue
            if steps[0] == "receive" and not ((a["player"] and a["player"] == b["player"]) or
                                               (a.get("track") and a["track"] == b.get("track"))):
                continue
            row = span_row(c, "sequence", a["t"], min(c.duration,b["t"]+1), f"{steps[0]} then {steps[1]}.")
            row["evidence"] = [{"t": x["t"], "detail": action_text(x)} for x in (a,b)]
            rows.append(row)
        return verified(rows, "sequence", candidates)
    rows, candidates = [], []
    previous = None
    for a in c.actions_by_type.get(steps[0], []):
        row = action_row(c, a)
        candidates.append([row["t_seconds"],row["episode_end"]])
        if team and a["team"] != team or player and a["player"] != player:
            continue
        if not matches_fields(a, plan) or not outcome_matches(a["outcome"], wanted):
            continue
        # The model sometimes describes one delivery twice in overlapping views.
        if previous and a["action"] == "cross" and 0 <= a["t"] - previous["t"] <= 1.5 and all(
                a[k] == previous[k] for k in ("team", "player", "start", "end")):
            continue
        rows.append(row)
        previous = a
    return verified(rows, "actions", candidates)


def search_clip(c, q):
    plan = parse(q, c.lexicon)
    if plan.problem or plan.unknown:
        reason = plan.problem or "The index cannot verify: " + ", ".join(dict.fromkeys(plan.unknown)) + "."
        return suggestions(c, q, plan, reason)
    if len(plan.tags) == 1 and plan.one("player"):
        return player_reel(c, plan.one("player"))
    if plan.one("piece") or plan.one("pattern") or "regain" in plan.values("step"):
        return passage_search(c, q, plan)
    if "neg" in plan.values("relation"):
        return suggestions(c, q, plan, "This exclusion needs a passage with an established outcome.")
    if plan.one("result") == "goal" and not plan.one("step"):
        return goal_search(c, q, plan)
    action_roles = {"shooter", "keeper"}
    if "open_play" in plan.values("relation") or set(plan.values("role")) - action_roles:
        return suggestions(c, q, plan, "The roles or phase of play could not be verified.")
    if plan.one("step") or plan.one("shape") or plan.one("result") == "saved":
        if plan.one("result") == "saved" and not plan.one("step"):
            plan.tags.append(("step", "shot"))
        return action_search(c, q, plan)
    return suggestions(c, q, plan, "No structured situation was recognized. These are possible description matches.")


CLIPS = {p.parent.name: Clip(p.parent.name) for p in sorted((DATA / "clips").glob("*/index.json"))
         if "actions" in json.loads(p.read_text())}
CLIP_ORDER = sorted(CLIPS, key=lambda n: CLIPS[n].label)
DEFAULT_CLIP = os.environ.get("DEFAULT_CLIP") or (CLIP_ORDER[0] if CLIP_ORDER else None)


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
            "confidence": 1.0 if g["scoreboard_t"] is not None else c.by_id[g["action_id"]]["confidence"],
            "image": c.image_at(g["t"]), "scorebug_corroborated": g["scoreboard_t"] is not None} for g in c.goals]
    ev.sort(key=lambda e: e["t_seconds"])
    return JSONResponse({"events": ev, "duration": c.duration, "confirmed_goals": sum(g["scoreboard_t"] is not None for g in c.goals)})


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
    res = search_clip(c, name)
    st = next((p for p in json.loads(players(clip).body)["players"] if p["name"] == res.get("player")), {})
    return JSONResponse({"name": res.get("player") or name, "stats": st, "moments": res["results"],
                         "reel": res.get("reel", [])})


@app.get("/api/visual-search")
def visual_search(q: str, clip: str = ""):
    from .visual import visual_candidates
    return visual_candidates(get_clip(clip), q)


@app.get("/api/search")
def search(q: str = "", clip: str = "", diagnostics: bool = False):
    c = get_clip(clip)
    if not q.strip():
        return respond([action_row(c, a) for a in c.actions if a["action"] in NOTABLE], "browse")
    result = search_clip(c, q)
    if not diagnostics:
        result.pop("candidate_spans", None)
    return JSONResponse(result)


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-store"})


@app.get("/media/clips/{clip}/{asset:path}")
def clip_media(clip: str, asset: str):
    c = get_clip(clip)
    relative = c.prefix + asset
    if relative != c.video and relative not in c.frame_files:
        raise HTTPException(404, "Unknown media file")
    return FileResponse(DATA / relative)



app.mount("/static", StaticFiles(directory=STATIC), name="static")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8008, log_level="warning")
