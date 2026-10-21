# Evaluation — video-native event detection

Hand-verified by watching the footage (burst/contact-sheet frame review) with the
scorebug and commentary as ground-truth aids. Clip: a real Man Utd v Man City
broadcast (NOT AI-generated). Final score 2-0; extra goals were VAR-disallowed.

## Goals — the key correction

The first pass used the scorebug ("goals that count") as truth and scored
precision at 29%. That was the wrong measure. The specification is "detect the
ball that enters the net". **A VAR-disallowed goal is still a real ball-in-net
event.**

Re-reviewing the 7 detected goals frame-by-frame:

| detected | scorebug | what the frames show | verdict |
|----------|----------|----------------------|---------|
| 03:34 | 0-0 | attack then a player celebrating | real goal, VAR-disallowed |
| 04:18 | 0-0 | attack then a player celebrating | real goal, VAR-disallowed |
| 05:05 | →1-0 | shot + celebration | real goal (counted) |
| 05:29 | 1-0 | shot + keeper + celebration | real (replay of the 1-0) |
| 06:20 | →2-0 | goalmouth | real goal (counted) |
| 08:32 | 2-0 | #11 celebrating with a teammate | real goal, VAR-disallowed |
| 09:07 | 2-0 | keeper save, no celebration | FALSE POSITIVE (a save) |

- **5 of 7 detections are distinct real balls-in-the-net** (2 counted + 3
  disallowed). 1 is a duplicate (a replay of the 1-0, which the prompt is meant to
  exclude) and 1 is a false positive (a save).
- Recall of the 2 counted goals: **100%**.
- The model is NOT hallucinating goals — it correctly sees the ball cross the
  line, including on disallowed goals the scorebug never counts. "Goal scored"
  vs "goal awarded" are different things; the scorebug/VAR decides the latter.

## Extended eval — shots, saves, corners, cards, fouls

Ground-truth aid = the commentary (reliably names notable events). Recall = did
the pipeline catch each named event (±15 s). Precision = frame-checked sample.

| type | recall (vs commentary) | detected | precision (sampled) | notes |
|------|------------------------|----------|---------------------|-------|
| goal (ball-in-net) | 2/2 counted (+3 disallowed found) | 7 | 5/7 distinct real | 1 replay duplicate + 1 save; rest are real incl. disallowed |
| shot | 5/5 | 14 | ~60-70% | near-goal shots right; some buildup mislabeled "shot" |
| save | 2/2 | 9 | ~85%+ | mostly real keeper / goalmouth actions |
| corner | 1/2 | 1 | — | UNDER-detected (only 1 all clip) |
| foul | challenge caught, 0 as "foul" | 0 | — | sees the tackle/duel, misses the foul DECISION |
| yellow_card | 0/1 | 0 | — | MISS — never recognises a ref showing a card |

## What the model is good and bad at

Good:
- **High recall on near-goal action** — it misses almost no shot/save/goal.
- **Ball-in-net detection is good** (~71%, 5 of 7), including VAR-disallowed goals.

Bad:
- **Fine type precision is limited.** The model finds it hard to separate a shot,
  a cross, a save, and a scramble. This is the same cause as the earlier over-use
  of "cross". The model reliably reports a notable action near goal. It less
  reliably reports the exact action.
- **Referee events are missed** — cards (0/1) and fouls-as-fouls (0/1). A referee
  who shows a card or gives a foul is a short, specific event. The 8-frame
  sampling misses it.
- **Corners are under-detected** (1 of 2+).

## What would fix each (without a scoreboard)

- Fine type + goal precision: **ball-trajectory tracking** (objective shot/cross/
  line-crossed), or a per-candidate verification pass.
- Cards/fouls: a dedicated referee-cue detector (card colour, ref gesture,
  free-kick restart), or denser sampling around whistle/stoppage moments.
- Corners: detect the corner-flag setup / restart explicitly.

## Method note

Ground truth built from the scorebug (goals), commentary (named events), and
direct frame review. Precision samples are small; treat the percentages as
indicative, not exact. Repeatable per type via the same match-within-window scoring.

## Gemini 4 Argon watching 8-second chunks (2026-10)

The video-events stage now sends each 8 s chunk to Gemini 4 Argon as an mp4
`file` part, in place of the 8-frame burst. Goal events scored against the
scorebug goals on both clips (seed: 2, clip2: 5), counted as hits within ±10 s:

| method | goals found | false goals |
|--------|-------------|-------------|
| 8-frame bursts (previous index) | 6/7 | 13 |
| Argon on 8 s video chunks | 5/7 | 2 |

- **Precision is the gain.** Argon labels replays and celebrations as such, so
  their events are dropped. False goals fall from 13 to 2.
- **Both misses are chunk-boundary goals** (seed 306 s, clip2 198 s). The ball
  crosses the line in the first second of a chunk that is mostly celebration,
  so the chunk is classed `celebration` and its events are dropped.
- **Both false goals are the broadcast re-showing an earlier goal from the
  main camera** (seed 423 s, clip2 493 s). The scorebug does not change and the
  commentary is already talking about the goal.
- **A 3 s lead-in on each chunk made it worse** (4 more false goals): replays
  crept back in as `wide_live`. That change was reverted.

### Manual review of Argon's chunk summaries

66 of 155 chunks checked by eye against 2 fps contact sheets and the
commentary track: every chunk with a goal, shot, save, corner, free kick or
tackle, plus 3 random chunks per shot type.

| verdict | seed (35) | clip2 (31) | total |
|---------|-----------|------------|-------|
| correct | 22 | 20 | 42 (64%) |
| partly right (missed a save, wrong actor) | 4 | 5 | 9 |
| wrong outcome or invented play | 5 | 3 | 8 |
| replay labelled `wide_live` | 4 | 3 | 7 |

- **Seed has a third goal the scorebug truth missed:** Mount at ~512 s (3-0).
  Against 8 real goals, Argon finds 5/8 and the frame bursts 6/8. Argon
  misses it the same way as the other two: the ball goes in in the last
  half second of a chunk and the next chunk is `celebration`.
- **Strong on what happened and who:** shirt numbers and names match the
  commentary (Bruno #8, Mbeumo #19, Amad #16, Dorgu #13, Cunha #10, Lammens
  #31). It reads the scoreboard change and describes Dorgu's goal going in off
  the underside of the bar.
- **Weak on goalmouth outcomes.** Two disallowed goals where the ball is in the
  net (seed 144 s, 215 s) become "cleared off the line". Maguire's header onto
  the bar becomes a clearance and a shot over. A goal (seed 304 s) becomes a
  parry. It invents keeper catches (clip2 46 s) and a keeper touch on a shot
  he never reached (seed 487 s).
- **Main-camera replays read as live play.** 7 of the 13 replays sampled were
  labelled `wide_live`, mostly ones that reuse the main camera angle.
  Each mislabelled replay duplicates the original's events; two of them are
  the false goals.
- **A yellow card is lost.** The summary at 72 s gets the card shown to Dalot
  right, but the chunk is `stoppage` and only `wide_live`/`live` chunks keep
  events.

### Old setup on the same chunks, and overlapping windows (2026-10-21)

Old setup = Gemini 3.5 Flash on 8 still frames per 8s chunk (commit 801d725), graded by hand on the same 66 chunks (65 for the old run, which has no clip2 608 chunk).

| | Argon (mp4) | Old (8 frames) |
|---|---|---|
| Correct | 42 (64%) | 37 (57%) |
| Partly right | 9 | 7 |
| Wrong outcome or invented play | 8 | 11 |
| Replay labelled wide_live | 7 of 13 | 10 of 14 |

The old run gets team or outcome wrong more often (City attack that was Amad, invented saves, an invented goal in a summary) and rarely names players. Argon labels three replays correctly that the old run treated as live goals (clip2 80, 328, seed 128).

Overlap: a second Argon pass on 8s windows shifted by 4s (starts 4, 12, 20, ...), merged with the aligned chunks, goals deduped within 15s. Scorebug gate = keep a goal only if the OCR score changes between 10s before and 15s after it. Ground truth: 8 goals.

| Variant | Goals hit | False goals |
|---|---|---|
| Old setup | 6/8 | 13 |
| Old setup + scorebug gate | 6/8 | 1 |
| Argon aligned | 5/8 | 2 |
| Argon aligned + gate | 5/8 | 0 |
| Argon aligned + shifted | 8/8 | 7 |
| Argon aligned + shifted + gate | 7/8 | 0 |

The shifted windows catch all three boundary goals (seed 302.5, seed 512, clip2 199) because the shot and the ball in the net land in one window. They also see more replays, so ungated they add false goals. The gate removes all of them. The one gated miss (seed 512, Mount's 3-0) is an OCR gap: the scorebug reading never moves past 2-0. Cost: twice the Argon calls.
