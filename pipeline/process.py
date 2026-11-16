"""Deterministic processing of the vision output. No model calls.

Reads data/clips/<clip>/vision.json (+ identity.json from pipeline/identity.py if
present) and writes data/clips/<clip>/index.json:

  teams     the two teams from the vision survey
  roster    (team, shirt number) -> printed shirt name, by majority vote over every
            shirt the vision model read. A pairing needs ROSTER_MIN_VOTES sightings
            and ROSTER_MIN_SHARE of that number's votes.
  actions   on-ball actions from live play, absolute times, duplicates from
            overlapping windows merged, each with a player when the shirt was read:
            identity_source = crop (identity.py read the number on a zoomed crop)
                            | shirt (the action log read the number or name)
                            | carried (same player as the receive/dribble next to it)
                            | none
  episodes  set pieces (corner, free kick, penalty) with the outcome of the
            following attack
  patterns  switches of play and counter-attacks derived from action geometry
  goals     one per scoreboard increment (scoreboard.json from pipeline/scoreboard.py)
            with the scoring team's last shot before it as the scorer; footage with
            no scoreboard (training) uses the goals the vision model saw go in

    CLIP_VIDEO=data/clips/clip2/clip2.mp4 python pipeline/process.py
"""
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import CLIP_DIR, FRAMES, INTERVAL_S

ROSTER_MIN_VOTES, ROSTER_MIN_SHARE = 2, 0.6
ROSTER_TRUSTED = 8    # with this many shirts read for a team, numbers off its roster are misreads
CARRY_S = 2.5         # receive -> next action by the same team this soon = the same player
CARRY_FROM = ("receive", "dribble", "interception", "tackle")  # he now has the ball
CARRY_NEXT = ("pass", "dribble", "shot", "header", "cross")
DEDUP_S = 1.5          # same action by the same team this close = one action seen twice
SET_PIECES = ("corner", "free_kick", "penalty")
OUTCOME_S = 10.0       # how long after a set piece its outcome is looked for
COUNTER_S = 12.0       # ball won -> shot by the same team within this = counter-attack
LIVE = ("live_wide", "live_close")
SCOREBOARD_MIN_SHARE = 0.5  # below this share of frames with a score, there is no scoreboard
GOAL_LOOKBACK_S = 45
REPLAY_S = 15          # how much of the build-up a replayed goal shows
REPLAY_WINDOW_S = 90   # replays of a goal come within this of its score change  # a highlight reel shows the goal, replays, then the new score


def fold(s):
    """'B. Fernandes' / 'ŠEŠKO' -> comparable key."""
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().upper()
    return re.sub(r"[^A-Z ]", "", s.split(".")[-1]).strip()


def num(v):
    m = re.search(r"\d+", str(v or ""))
    return m.group(0) if m else None


def tc(s):
    return f"{int(s) // 60:02d}:{int(s) % 60:02d}"


def team_of(raw, teams):
    r = fold(raw)
    for t in teams:
        if r and (r == fold(t["name"]) or r == fold(t.get("scoreboard_abbr")) or fold(t["name"]) in r):
            return t["name"]
    return None


def build_roster(windows, teams):
    votes = defaultdict(Counter)
    display = {}
    for w in windows:
        for s in w["shirts"]:
            team, n, name = team_of(s.get("team"), teams), num(s.get("shirt_number")), fold(s.get("shirt_name"))
            if team and n and name:
                votes[(team, n)][name] += 1
                display.setdefault(name, s["shirt_name"].split(".")[-1].strip().title())
    best = {}  # (team, name) -> (votes, number): a player wears one number
    for (team, n), c in votes.items():
        name, k = c.most_common(1)[0]
        if k >= ROSTER_MIN_VOTES and k / sum(c.values()) >= ROSTER_MIN_SHARE and k > best.get((team, name), (0,))[0]:
            best[(team, name)] = (k, n)
    return {f"{team}#{n}": display[name] for (team, name), (k, n) in best.items()}


def actions_from(vision, teams):
    out = []
    for w in vision["windows"]:
        if w["shot_type"] not in LIVE:
            continue
        for a in w["actions"]:
            try:
                t = w["start"] + float(a.get("t_offset_s")) / vision["slow"]
            except (TypeError, ValueError):
                continue
            out.append({"t": round(min(t, w["start"] + vision["win_s"]), 1), "action": a.get("action"),
                        "team": team_of(a.get("team"), teams), "number": num(a.get("shirt_number")),
                        "shirt_name": a.get("shirt_name"), "body_shape": a.get("body_shape"),
                        "start": a.get("start"), "end": a.get("end"), "length": a.get("length"),
                        "outcome": a.get("outcome"), "pressure": a.get("pressure"),
                        "confidence": a.get("confidence_0_1") or 0, "detail": a.get("detail") or ""})
    return out


def dedup(actions):
    kept = []
    for a in sorted(actions, key=lambda x: -x["confidence"]):
        twin = next((k for k in kept if k["action"] == a["action"] and k["team"] == a["team"]
                     and abs(k["t"] - a["t"]) <= DEDUP_S
                     and (k["number"] == a["number"] or None in (k["number"], a["number"]))), None)
        if twin is None:
            kept.append(dict(a))
        else:  # same action seen from the other window: keep any shirt it read
            twin["number"] = twin["number"] or a["number"]
            twin["shirt_name"] = twin["shirt_name"] or a["shirt_name"]
    kept.sort(key=lambda x: x["t"])
    for i, a in enumerate(kept):
        a["id"] = i
    return kept


def crop_number(crop, kit):
    """The number read on the zoomed crops: at least 2 readable reads whose shirt colour
    matches the team's kit, agreeing (one read among blanks is too easy to misread)."""
    words = set(re.findall(r"[a-z]+", (kit or "").lower()))
    nums = [num(r.get("shirt_number")) for r in (crop or {}).get("reads", [])
            if num(r.get("shirt_number")) and set(re.findall(r"[a-z]+", str(r.get("shirt_colour")).lower())) & words]
    c = Counter(nums).most_common()
    if not c or (len(c) > 1 and c[0][1] == c[1][1]):
        return None
    return c[0][0] if c[0][1] >= 2 else None


def identify(actions, roster, crops, kits):
    known = Counter(k.split("#")[0] for k in roster)
    for a in actions:
        cn = crop_number(crops.get(f"{a['t']}|{a['action']}|{a['team']}"), kits.get(a["team"]))
        n, src = None, "none"
        if cn:
            n, src = cn, "crop"
        elif a["number"]:
            n, src = a["number"], "shirt"
        if n and known[a["team"]] >= ROSTER_TRUSTED and f"{a['team']}#{n}" not in roster:
            n, src = None, "none"  # a number no shirt of that team ever showed: a misread
        player = roster.get(f"{a['team']}#{n}") if n else None
        if not player and a["shirt_name"]:
            player = next((v for v in roster.values() if fold(v) == fold(a["shirt_name"])), None)
            src = "shirt" if player else src
        a.update(number=n, player=player, identity_source=src if (n or player) else "none")
    carry(actions)


def carry(actions):
    """A player who wins or receives the ball and then passes, dribbles or shoots is the
    same player: link consecutive actions by one team when nothing happens in between."""
    for _ in range(3):  # chains of up to 3 links
        for a, b in zip(actions, actions[1:]):
            if a["team"] and a["team"] == b["team"] and b["t"] - a["t"] <= CARRY_S \
                    and a["action"] in CARRY_FROM and b["action"] in CARRY_NEXT:
                src, dst = (a, b) if a["number"] and not b["number"] else (b, a) if b["number"] and not a["number"] else (None, None)
                if src:
                    dst.update(number=src["number"], player=src["player"], identity_source="carried")


def side(z):
    return (z or {}).get("side") if isinstance(z, dict) else None


def derive(actions):
    episodes, patterns = [], []
    for i, a in enumerate(actions):
        if a["action"] in SET_PIECES:
            if episodes and episodes[-1]["type"] == a["action"] and a["t"] - episodes[-1]["start"] <= 8:
                continue  # set-up and delivery of the same set piece
            follow = [b for b in actions[i + 1:] if b["t"] - a["t"] <= OUTCOME_S]
            nxt = next((b for b in follow if b["action"] in SET_PIECES), None)
            if nxt:
                follow = [b for b in follow if b["t"] < nxt["t"]]
            rank = {"goal": 3, "save": 2, "shot": 1, "header": 1}
            best = max(follow, key=lambda b: rank.get(b["action"], 0) + (3 if b["outcome"] == "goal" else 0),
                       default=None)
            outcome = "cleared"
            if best and (best["action"] == "goal" or best["outcome"] == "goal"):
                outcome = "goal"
            elif best and best["action"] == "save":
                outcome = "save"
            elif best and best["action"] in ("shot", "header") and rank.get(best["action"]):
                outcome = "shot"
            episodes.append({"type": a["action"], "team": a["team"], "start": a["t"],
                             "end": best["t"] if best and outcome != "cleared" else a["t"] + 4,
                             "outcome": outcome, "taker": a.get("player"), "action_ids": [a["id"]] +
                             [b["id"] for b in follow]})
        if a["action"] in ("pass", "cross") and a["length"] == "long" and \
                {side(a["start"]), side(a["end"])} == {"left", "right"}:
            patterns.append({"type": "switch_of_play", "team": a["team"], "start": a["t"] - 2,
                             "end": a["t"] + 4, "action_ids": [a["id"]]})
        won = (a["action"] in ("interception", "tackle", "save") or
               (a["action"] == "clearance" and a["outcome"] == "won")) and \
            (a["start"] or {}).get("third") in ("def", "mid")
        if won and a["team"]:
            shot = next((b for b in actions[i + 1:] if b["t"] - a["t"] <= COUNTER_S and b["team"] == a["team"]
                         and b["action"] in ("shot", "header", "goal")), None)
            if shot and not any(p["type"] == "counter_attack" and p["end"] == shot["t"] + 3 for p in patterns):
                patterns.append({"type": "counter_attack", "team": a["team"], "start": a["t"] - 2,
                                 "end": shot["t"] + 3, "action_ids": [b["id"] for b in actions
                                                                      if a["t"] <= b["t"] <= shot["t"]]})
    return episodes, patterns


def score_changes(reads):
    """[(t, team_index)] for every score increment that holds for 2 consecutive reads."""
    valid = [r for r in reads if r["score"]]
    out, cur = [], None
    for r, nxt in zip(valid, valid[1:] + [None]):
        sc = r["score"]
        if cur is None:
            cur = sc
            continue
        if sc != cur and nxt and nxt["score"] == sc and sum(sc) == sum(cur) + 1:
            out.append((r["t"], 0 if sc[0] > cur[0] else 1))
            cur = sc
    return out


def drop_replays(actions, changes, teams):
    """One score change is one goal, however often the broadcast shows it. Of the goals
    the vision model saw, the live one is the scoring team's last sighting before the
    score changes; every other sighting (replays from any camera, disallowed goals)
    loses the REPLAY_S of actions leading up to it. Then renumber."""
    seen = [a for a in actions if a["action"] == "goal" or a["outcome"] == "goal"]
    live = set()
    for t, k in changes:
        before = [a["t"] for a in seen if a["team"] == teams[k]["name"] and t - REPLAY_WINDOW_S <= a["t"] <= t + 6]
        if before:  # a goal and its shot are ~1 s apart: keep both
            live |= {x for x in before if max(before) - x <= 3}
    fake = [a["t"] for a in seen if a["t"] not in live]
    kept = [a for a in actions if not any(f - REPLAY_S <= a["t"] <= f + 3 for f in fake)]
    for i, a in enumerate(kept):
        a["id"] = i
    return kept


def goals_seen(actions):
    """No scoreboard (e.g. training footage): goals the vision model saw go in during live play."""
    goals = []
    for i, a in enumerate(actions):
        if (a["action"] == "goal" or a["outcome"] == "goal") and (not goals or a["t"] - goals[-1]["t"] > 20):
            shot = a if a["action"] != "goal" else next(
                (b for b in reversed(actions[:i]) if a["t"] - b["t"] <= 3 and b["action"] in ("shot", "header")), a)
            goals.append({"t": a["t"], "scoreboard_t": None, "team": a["team"], "scorer": shot.get("player"),
                          "action_id": shot["id"]})
    return goals


def goals_from(reads, actions, teams):
    """Scoreboard says when and who; the action log's last shot by that team says how."""
    goals = []
    for t, k in score_changes(reads):
        team = teams[k]["name"]
        near = [a for a in actions if t - GOAL_LOOKBACK_S <= a["t"] <= t + 3 and a["team"] == team]
        # the scorer is whoever struck the ball: the last shot or header before the new
        # score; a "goal" action (ball seen crossing the line) only if no strike was seen
        shot = next((a for a in reversed(near) if a["action"] in ("shot", "header")), None) or \
            next((a for a in reversed(near) if a["action"] == "goal"), None)
        goals.append({"t": shot["t"] if shot else t, "scoreboard_t": t, "team": team,
                      "scorer": shot.get("player") if shot else None,
                      "action_id": shot["id"] if shot else None})
    return goals


def main():
    vision = json.loads((CLIP_DIR / "vision.json").read_text())
    ip = CLIP_DIR / "identity.json"
    crops = json.loads(ip.read_text()) if ip.exists() else {}
    teams = vision["teams"]
    roster = build_roster(vision["windows"], teams)
    actions = dedup(actions_from(vision, teams))
    identify(actions, roster, crops, {t["name"]: t["kit"] for t in teams})
    sp = CLIP_DIR / "scoreboard.json"
    reads = json.loads(sp.read_text()) if sp.exists() else []
    has_board = sum(1 for r in reads if r["score"]) >= SCOREBOARD_MIN_SHARE * max(len(reads), 1)
    if has_board:
        actions = drop_replays(actions, score_changes(reads), teams)
    episodes, patterns = derive(actions)
    goals = goals_from(reads, actions, teams) if has_board else goals_seen(actions)
    for e in episodes:  # a set piece "led to a goal" only if the scoreboard says a goal came
        scored = any(e["start"] <= g["t"] <= e["start"] + OUTCOME_S + 2 and g["team"] == e["team"] for g in goals)
        e["outcome"] = "goal" if scored else ("shot" if e["outcome"] == "goal" else e["outcome"])
    frames = sorted(p.name for p in FRAMES.glob("*.jpg"))
    for a in actions:
        a["timecode"] = tc(a["t"])
        a["image"] = f"frames/{frames[min(int(a['t'] // INTERVAL_S), len(frames) - 1)]}" if frames else None
    index = {"video": vision["video"], "duration": vision["duration"], "teams": teams, "roster": roster,
             "frames": [f"frames/{f}" for f in frames], "frame_interval_s": INTERVAL_S,
             "actions": actions, "episodes": episodes, "patterns": patterns, "goals": goals}
    (CLIP_DIR / "index.json").write_text(json.dumps(index, indent=1))
    ids = Counter(a["identity_source"] for a in actions)
    print(f"{len(roster)} roster entries, {len(actions)} actions (identity {dict(ids)}), "
          f"{len(episodes)} set pieces, {len(patterns)} patterns, {len(goals)} goals")


if __name__ == "__main__":
    main()
