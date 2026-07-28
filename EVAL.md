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

- **~6 of 7 detections are real balls-in-the-net** (2 counted + ~3 disallowed +
  1 replay); only 1 is a true error.
- Recall of the 2 counted goals: **100%**.
- The model is NOT hallucinating goals — it correctly sees the ball cross the
  line, including on disallowed goals the scorebug never counts. "Goal scored"
  vs "goal awarded" are different things; the scorebug/VAR decides the latter.

## Extended eval — shots, saves, corners, cards, fouls

Ground-truth aid = the commentary (reliably names notable events). Recall = did
the pipeline catch each named event (±15 s). Precision = frame-checked sample.

| type | recall (vs commentary) | detected | precision (sampled) | notes |
|------|------------------------|----------|---------------------|-------|
| goal (ball-in-net) | 2/2 counted (+3 disallowed found) | 7 | ~6/7 real | over-fires only ~1; rest are real incl. disallowed |
| shot | 5/5 | 14 | ~60-70% | near-goal shots right; some buildup mislabeled "shot" |
| save | 2/2 | 9 | ~85%+ | mostly real keeper / goalmouth actions |
| corner | 1/2 | 1 | — | UNDER-detected (only 1 all clip) |
| foul | challenge caught, 0 as "foul" | 0 | — | sees the tackle/duel, misses the foul DECISION |
| yellow_card | 0/1 | 0 | — | MISS — never recognises a ref showing a card |

## What the model is good and bad at

Good:
- **High recall on near-goal action** — it misses almost no shot/save/goal.
- **Ball-in-net detection is strong** (~85%), including VAR-disallowed goals.

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
