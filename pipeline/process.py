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
                            | track (another read on the same player track, track.py)
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
from bisect import bisect_left
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import CLIP_DIR, FRAMES, INTERVAL_S

ROSTER_MIN_VOTES, ROSTER_MIN_SHARE = 2, 0.6
ROSTER_TRUSTED = 8    # with this many shirts read for a team, numbers off its roster are misreads
TRACK_MATCH_S = 0.1     # a tracked frame this close to a time counts as that time
TRACK_IOU = 0.3         # a drawn box and a tracked box this overlapping are the same player
NEAR_BALL_PX = 60       # the ball-nearest player must be this close (feet to ball, pixels)
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


def two_means(points, iters=20):
    """Deterministic 2-means: seeds are the two points farthest apart (via the first point)."""
    d2 = lambda p, q: sum((x - y) ** 2 for x, y in zip(p, q))
    a = max(points, key=lambda p: d2(p, points[0]))
    b = max(points, key=lambda p: d2(p, a))
    cents = [list(a), list(b)]
    for _ in range(iters):
        groups = [[], []]
        for p in points:
            groups[0 if d2(p, cents[0]) <= d2(p, cents[1]) else 1].append(p)
        cents = [[sum(c) / len(g) for c in zip(*g)] if g else cents[i] for i, g in enumerate(groups)]
    return cents, lambda p: 0 if d2(p, cents[0]) <= d2(p, cents[1]) else 1


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


class Tracks:
    """Player tracks (pipeline/track.py), cleaned: every outfield box gets a shirt-colour
    cluster, and a track is split wherever its smoothed cluster changes (the tracker
    jumped to a player of the other team)."""

    def __init__(self, data):
        self.w, self.h = data["width"], data["height"]
        self.frames = data["frames"]
        self.times = [f["t"] for f in self.frames]
        pts = [p["bgr"] for f in self.frames for p in f["people"] if p["cls"] == "player" and p["bgr"]]
        _, self.cluster = two_means(pts)
        seq = defaultdict(list)
        for f in self.frames:
            for p in f["people"]:
                if p["cls"] == "player" and p["bgr"]:
                    seq[p["id"]].append((f["t"], p))
        for tid, items in seq.items():
            labels = [self.cluster(p["bgr"]) for _, p in items]
            part = 0
            for i, (_, p) in enumerate(items):
                win = labels[max(0, i - 2):i + 3]
                p["team_k"] = max(set(win), key=win.count)
                if i and p["team_k"] != items[i - 1][1]["team_k"]:
                    part += 1
                p["track"] = f"{tid}.{part}"
        self.team_k, self.span = {}, {}
        for f in self.frames:
            for p in f["people"]:
                p.setdefault("track", str(p["id"]))
                p.setdefault("team_k", None)
                self.team_k[p["track"]] = p["team_k"]
                a, b = self.span.get(p["track"], (f["t"], f["t"]))
                self.span[p["track"]] = (min(a, f["t"]), max(b, f["t"]))

    def frame_at(self, t):
        i = bisect_left(self.times, t)
        near = [j for j in (i - 1, i) if 0 <= j < len(self.times)]
        j = min(near, key=lambda j: abs(self.times[j] - t), default=None)
        return self.frames[j] if j is not None and abs(self.times[j] - t) <= TRACK_MATCH_S else None

    def match(self, t, box_norm):
        """The track whose box overlaps a 0-1000 normalized [ymin, xmin, ymax, xmax] box at t."""
        f = self.frame_at(t)
        if not f or not box_norm:
            return None
        y0, x0, y1, x1 = box_norm
        b = [x0 * self.w / 1000, y0 * self.h / 1000, x1 * self.w / 1000, y1 * self.h / 1000]
        best = max(f["people"], key=lambda p: iou(p["box"], b), default=None)
        return best if best and iou(best["box"], b) >= TRACK_IOU else None

    def nearest_ball(self, t, team_k):
        """The outfield player of cluster team_k closest to the ball at t."""
        f = self.frame_at(t)
        if not f or not f["ball"]:
            return None
        bx, by = (f["ball"][0] + f["ball"][2]) / 2, (f["ball"][1] + f["ball"][3]) / 2
        cand = [p for p in f["people"] if p["cls"] == "player" and p["team_k"] == team_k]
        foot = lambda p: ((p["box"][0] + p["box"][2]) / 2 - bx) ** 2 + (p["box"][3] - by) ** 2
        best = min(cand, key=foot, default=None)
        return best if best and foot(best) <= NEAR_BALL_PX ** 2 else None


def track_numbers(tracks, actions, crops, kits, roster, track_reads):
    """Vote one shirt number per track from every crop read that overlaps it plus the
    reads taken along the track itself (track_read.py), and map shirt-colour clusters
    to team names by which team's actions their boxes belong to."""
    votes, team_votes = defaultdict(Counter), defaultdict(Counter)
    for a in actions:
        c = crops.get(f"{a['t']}|{a['action']}|{a['team']}")
        if not c:
            continue
        offs = c.get("offsets") or [-0.5, 0.0, 0.5]
        boxed = [(o, bx) for o, bx in zip(offs, c.get("boxes") or []) if bx]
        words = set(re.findall(r"[a-z]+", (kits.get(a["team"]) or "").lower()))
        for (o, bx), r in zip(boxed, c.get("reads") or []):
            p = tracks.match(a["t"] + o, bx)
            if not p:
                continue
            if p["team_k"] is not None and a["team"]:
                team_votes[p["team_k"]][a["team"]] += 1
            n = num(r.get("shirt_number"))
            if n and set(re.findall(r"[a-z]+", str(r.get("shirt_colour")).lower())) & words:
                votes[p["track"]][(a["team"], n)] += 1
    k2team = {k: c.most_common(1)[0][0] for k, c in team_votes.items()}
    for tr, reads in track_reads.items():
        team = k2team.get(tracks.team_k.get(tr))
        words = set(re.findall(r"[a-z]+", (kits.get(team) or "").lower()))
        for r in reads:
            n = num(r.get("shirt_number"))
            if team and n and set(re.findall(r"[a-z]+", str(r.get("shirt_colour")).lower())) & words:
                votes[tr][(team, n)] += 1
    known = Counter(k.split("#")[0] for k in roster)
    numbers = {}
    for tr, c in votes.items():
        (team, n), k = c.most_common(1)[0]
        if k >= 2 and k / sum(c.values()) >= 0.6 and (known[team] < ROSTER_TRUSTED or f"{team}#{n}" in roster):
            numbers[tr] = (team, n)
    return numbers, k2team


def actor_track(tracks, a, crop, k2team):
    """The track of the player doing action a: the box the identity step drew at the
    action time, else the player of the acting team nearest the ball."""
    if crop:
        offs = crop.get("offsets") or [-0.5, 0.0, 0.5]
        for o, bx in sorted(zip(offs, crop.get("boxes") or []), key=lambda x: abs(x[0])):
            p = tracks.match(a["t"] + o, bx) if bx else None
            if p:
                return p
    team_k = next((k for k, t in k2team.items() if t == a["team"]), None)
    return tracks.nearest_ball(a["t"], team_k) if team_k is not None else None


def identify(actions, roster, crops, kits, tracks=None, track_reads=None):
    known = Counter(k.split("#")[0] for k in roster)
    numbers, k2team = track_numbers(tracks, actions, crops, kits, roster, track_reads or {}) if tracks else ({}, {})
    for a in actions:
        crop = crops.get(f"{a['t']}|{a['action']}|{a['team']}")
        cn = crop_number(crop, kits.get(a["team"]))
        n, src = None, "none"
        p = actor_track(tracks, a, crop, k2team) if tracks else None
        a["track"] = p["track"] if p else None
        tn = numbers.get(a["track"]) if a["track"] else None
        if cn:
            n, src = cn, "crop"
        elif tn and tn[0] == a["team"]:
            n, src = tn[1], "track"
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
    tp = CLIP_DIR / "tracks.json"
    tracks = Tracks(json.loads(tp.read_text())) if tp.exists() else None
    rp = CLIP_DIR / "track_reads.json"
    identify(actions, roster, crops, {t["name"]: t["kit"] for t in teams}, tracks,
             json.loads(rp.read_text()) if rp.exists() else {})
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
             "actions": actions, "episodes": episodes, "patterns": patterns, "goals": goals,
             "track_parts": {a["track"]: tracks.span[a["track"]] for a in actions if a.get("track")} if tracks else {}}
    (CLIP_DIR / "index.json").write_text(json.dumps(index, indent=1))
    ids = Counter(a["identity_source"] for a in actions)
    print(f"{len(roster)} roster entries, {len(actions)} actions (identity {dict(ids)}), "
          f"{len(episodes)} set pieces, {len(patterns)} patterns, {len(goals)} goals")


if __name__ == "__main__":
    main()
