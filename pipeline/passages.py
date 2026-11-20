"""Construct searchable passages from chronological live observations and corroborated goals."""
from bisect import bisect_right

SET_PIECES = {"corner", "free_kick", "penalty"}
RESTARTS = SET_PIECES | {"throw_in", "goal_kick", "kick_off", "foul"}


def same_delivery(a, b):
    """Overlapping windows may label the same long delivery a cross or a pass."""
    return (a["action"] in ("cross", "pass") and b["action"] in ("cross", "pass")
            and a["length"] == b["length"] == "long" and a["team"] == b["team"]
            and a["track"] and a["track"] == b["track"] and b["t"] - a["t"] <= 1.5
            and a["start"] and a["start"] == b["start"] and a["end"] and a["end"] == b["end"])


def distinct_actions(actions):
    kept = []
    for a in actions:
        if kept and same_delivery(kept[-1], a):
            previous = kept[-1]
            if a["action"] == "cross" and previous["action"] != "cross":
                kept[-1] = a
            continue
        kept.append(a)
    return kept


def is_shot(a):
    return a["action"] == "shot" or (a["action"] == "header" and
            a["outcome"] in ("on_target", "off_target", "saved", "goal", "blocked"))


def won_ball(a):
    return a["action"] in ("tackle", "interception") and a["outcome"] == "won"


def progression(a, origin):
    if a["action"] not in ("pass", "dribble", "cross") or a["outcome"] in ("out", "lost", "intercepted"):
        return False
    zones = {"def": 0, "mid": 1, "att": 2}
    start, end = a["start"] or {}, a["end"] or {}
    s, e = zones.get(start.get("third")), zones.get(end.get("third"))
    # Ball carries often have no annotated end. A later same-possession
    # attacking touch in a more advanced third supplies independent evidence.
    origin_third = zones.get((origin["start"] or {}).get("third"))
    if origin_third is not None and s is not None and s > origin_third:
        return True
    if s is None or e is None:
        return False
    return e > s or (e == s == 2 and (a["action"] == "dribble" or
                                      (start.get("side") != "center" and end.get("side") == "center")))


def follow_play(actions, index, breaks):
    """Stay in an observed phase. A defensive deflection alone does not transfer possession."""
    anchor = actions[index]
    span = [anchor]
    next_cut = bisect_right(breaks, anchor["t"])
    for j in range(index + 1, len(actions)):
        b = actions[j]
        if (b["t"] - span[-1]["t"] > 6 or
                (next_cut < len(breaks) and breaks[next_cut] <= b["t"])):
            break
        if b["action"] in RESTARTS:
            break
        if b["team"] != anchor["team"] and (b["action"] in ("receive", "pass", "dribble", "cross", "shot")
                                              or won_ball(b)):
            break
        span.append(b)
        if b["action"] == "goal" or b["outcome"] in ("out", "off_target"):
            break
    return span


def evidence(a, stage):
    return {"t": a["t"], "action": a["action"], "team": a["team"], "player": a["player"],
            "detail": a["detail"], "source": a.get("source", "vision"), "stage": stage,
            "action_id": a["id"], "identity_source": a["identity_source"],
            "window_starts": a.get("window_starts", []), "confidence": a["confidence"]}


def build_passage(kind, span, goals, attempt_saves, duration):
    first = span[0]
    team = first["team"]
    finish = None
    goal = next((goals[a["id"]] for a in span if a["id"] in goals and goals[a["id"]]["team"] == team), None)
    shots = [a for a in span if is_shot(a) and a["team"] == team]
    restart = kind in SET_PIECES
    if goal:
        shot = next(a for a in span if a["id"] == goal["action_id"])
        finish, outcome = shot, "goal"
    elif shots:
        shot = shots[-1] if restart else shots[0]
        save = next((a for a in span if a["id"] == attempt_saves.get(shot["id"])), None)
        finish = save or shot
        outcome = "save" if save or shot["outcome"] == "saved" else "shot"
    else:
        shot = None
        clearances = [a for a in span if a["action"] == "clearance" and a["team"] != team]
        finish = span[-1]
        outcome = "cleared" if finish in clearances else "unknown"
    final = [a for a in span if a["t"] <= finish["t"]]
    moves = [a for a in final[1:] if a["team"] == team and progression(a, first)
             and (not shot or a["t"] < shot["t"])]
    if not restart and (not shot or not moves):
        return None
    steps = [evidence(first, "delivery" if restart else "regain")]
    move = next(iter(moves), None)
    if not move and restart:
        move = next((a for a in final[1:] if a["team"] == team and a["action"] == "pass"
                     and shot and a["t"] < shot["t"]), None)
    if move:
        steps.append(evidence(move, "progression"))
    if shot and shot is not first:
        steps.append(evidence(shot, "shot"))
    if finish is not shot and finish is not first:
        steps.append(evidence(finish, finish["action"]))
    if goal:
        steps.append({"t": goal["t"], "confirmed_at": goal["scoreboard_t"],
                      "action": "goal", "stage": "goal", "player": goal["scorer"], "team": team,
                      "source": "scoreboard" if goal["scoreboard_t"] is not None else "vision",
                      "detail": f"The score increases for {team}." if goal["scoreboard_t"] is not None
                                else f"{team} scores in live play."})
    return {"type": kind, "team": team, "start": first["t"], "end": min(duration, finish["t"] + 1),
            "taker": first["player"] if restart else None, "scorer": goal["scorer"] if goal else None,
            "shooter": shot["player"] if shot else None, "outcome": outcome,
            "action_ids": [a["id"] for a in final], "evidence": steps}


def recycled_clearance(actions, index, breaks):
    """A contested delivery can return to its team without controlled opposition possession."""
    delivery = actions[index]
    saw_clearance = False
    for i in range(index + 1, len(actions)):
        touch = actions[i]
        if (touch["t"] - delivery["t"] > 4 or
                bisect_right(breaks, touch["t"]) != bisect_right(breaks, delivery["t"])):
            break
        if touch["team"] == delivery["team"]:
            if saw_clearance and touch["action"] == "receive":
                return i
            break
        if (not touch["team"] or touch["action"] not in {"header", "clearance"}
                or is_shot(touch) or touch["outcome"] == "out"):
            break
        saw_clearance |= touch["action"] == "clearance"
    return index


def possession_index(actions, breaks):
    """Group observed control. Keep an opposing save with the attempt it stopped."""
    possessions, saves = [], {}
    phase = pending = previous = None
    recovered_at = -1
    for index, a in enumerate(actions):
        t, team, kind, outcome = a["t"], a["team"], a["action"], a["outcome"]
        if index < recovered_at:
            phase["action_ids"].append(a["id"])
            continue
        if previous is not None and (t - previous > 6 or
                bisect_right(breaks, t) != bisect_right(breaks, previous)):
            phase = pending = None
        previous = t
        if kind == "save":
            if (phase and pending and team and team != phase["team"]
                    and 0 < t - pending["t"] <= 3):
                saves[pending["id"]] = a["id"]
                phase["action_ids"].append(a["id"])
            phase = pending = None
            continue
        if not team or kind in RESTARTS:
            phase = pending = None
        if not team:
            continue
        if phase and team != phase["team"] and (
                kind == "block" or kind == "tackle" and outcome not in {"won", "completed"}):
            continue
        if phase is None or team != phase["team"]:
            phase = {"team": team, "action_ids": []}
            possessions.append(phase)
        phase["action_ids"].append(a["id"])
        pending = a if is_shot(a) and outcome in {"saved", "on_target"} else None
        if kind in {"cross", "pass"} and outcome == "intercepted":
            recovered_at = recycled_clearance(actions, index, breaks)
            if recovered_at > index:
                continue
        if kind == "goal" or outcome in {"intercepted", "lost", "out", "off_target", "goal", "blocked"}:
            phase = pending = None
    return possessions, saves


def derive(actions, goals, breaks, duration):
    episodes, patterns = [], []
    goal_by_action = {g["action_id"]: g for g in goals if g["action_id"] is not None}
    # Consecutive descriptions of setup and delivery are a single restart.
    plays = []
    for a in actions:
        if (plays and a["action"] in SET_PIECES and plays[-1]["action"] == a["action"]
                and a["team"] == plays[-1]["team"] and a["t"] - plays[-1]["t"] <= 3):
            continue
        plays.append(a)
    possessions, saves = possession_index(actions, breaks)
    by_id = {a["id"]: a for a in actions}
    continuation = {aid: (phase["action_ids"], offset) for phase in possessions
                    for offset, aid in enumerate(phase["action_ids"])}
    seen_finishes = set()
    for i, a in enumerate(plays):
        if not a["team"]:
            continue
        if a["action"] in SET_PIECES:
            episodes.append(build_passage(a["action"], follow_play(plays, i, breaks), goal_by_action, saves, duration))
        elif won_ball(a):
            kind = "counter_attack" if (a["start"] or {}).get("third") in ("def", "mid") else "regain_attack"
            p = build_passage(kind, follow_play(plays, i, breaks), goal_by_action, saves, duration)
            if p and (a["team"], p["end"]) not in seen_finishes:
                seen_finishes.add((a["team"], p["end"]))
                patterns.append(p)
        if (a["action"] in ("cross", "pass") and a["length"] == "long" and
                {(a[k] or {}).get("side") for k in ("start", "end")} == {"left", "right"}):
            ids, offset = continuation[a["id"]]
            span = [by_id[aid] for aid in ids[offset:]]
            shot = next((s for s in span[1:] if s["team"] == a["team"] and is_shot(s) and s["t"] > a["t"]), None)
            switch = {"type": "switch_of_play", "team": a["team"], "start": max(0, a["t"] - 2),
                      "end": min(duration, max(a["t"] + 4, span[-1]["t"] + 1)), "action_ids": ids[offset:],
                      "evidence": [evidence(a, "delivery")]}
            if shot:
                save = by_id.get(saves.get(shot["id"]))
                finish = save or shot
                goal = goal_by_action.get(shot["id"])
                switch.update(end=min(duration, finish["t"] + 1), shooter=shot["player"],
                              outcome="goal" if goal else "save" if save else "shot",
                              scorer=goal["scorer"] if goal else None,
                              action_ids=ids[offset:ids.index(finish["id"], offset) + 1])
                for j, touch in enumerate(span[1:], 1):
                    if touch["id"] == shot["id"]:
                        break
                    if touch["action"] == "clearance" or (touch["action"] == "receive" and
                            span[j-1]["team"] != touch["team"]):
                        switch["evidence"].append(evidence(touch, touch["action"]))
                switch["evidence"].append(evidence(shot, "shot"))
                if save:
                    switch["evidence"].append(evidence(save, "save"))
            patterns.append(switch)
    return episodes, patterns, possessions, saves
