# Evaluation

All checks below were made against frames looked at by eye. The clips are the two
demo highlight reels at 1280x720.

## Local temporal grounding experiment

On 15 visually judged query/window pairs from the two reels, TimeLens2-2B
returned a span on eight of nine windows where the requested situation was
absent. A separate yes/no prompt answered yes to all 15. These cases deliberately
include hard negatives, overlapping footage, and paraphrases. They measure
specific failure modes and do not estimate accuracy across matches.

| Method | Positive cases with span IoU ≥ 0.5 | Absent cases correctly rejected |
|---|---:|---:|
| Current search API, offline structured routes | 3/6 | 6/9 |
| TimeLens2-2B, local video and original query | 2/6 | 1/9 |

The evaluator judged the full requested move, from delivery to outcome when
both were requested. For the API comparison, search ran on the whole reel
and returned episode spans were intersected with each test window. For
TimeLens2, each test window was decoded at 2 fps and 448 × 256 on Apple MPS.
The model saw the pixels and query only. The median time per case, including
decoding and generation but excluding model load, was 6.17 seconds.

Examples from Arsenal v United:

- The first live corner at about 346 s leads to a shot and save around 361 s.
  The index ends the episode at 351.1 s. At 366.5 s it indexes a replay of
  the shot as another corner.
- A separate live corner at about 383 s leads to the equalizer around 387 s.
  TimeLens2 found it, but also returned 462–469 s for the same query on a
  clip showing United scoring from open play.
- We also tried the same scoring-corner query on four overlapping 24-second
  windows between 336 and 408 s, then on three windows proposed by the
  index's corner detector. TimeLens2 returned a span in all seven windows,
  including both incorrect indexed proposals.

The model revision is `MCG-NJU/TimeLens2-2B@10a779b6ef6fc8ad41367db2d1eb6779ca84995a`.
The SHA-256 of `model.safetensors` is
`70facd614adbacf7d09bd6e6389ce11ece256e9bb0dcc52194c28882c71c8588`.
Weights are in `data/models/timelens2-2b/`. Raw local predictions are in
`data/evals/local-grounding/`. Both directories are git-ignored.
The reference windows and offline inference runner are in `evals/`.

On macOS with the dependencies already installed, run from the project root:

```bash
sandbox-exec -p '(version 1)(allow default)(deny network*)' \
  python evals/local_grounding.py data/models/timelens2-2b evals/corner_cases.jsonl
```

### Football-specific spotter

We also ran the 2025 SoccerNet Team Ball Action Spotting T-DEED baseline on
CPU, using three eight-second windows and the published two-head checkpoint.
This model assigns scores to 12 actions at each timestamp. We summed its two
team-side scores per action, then took each action's peak within the window.
The scores below are model outputs, not measured probabilities of correctness.

| Match and window | Observed play | Model output |
|---|---|---|
| Arsenal v United, 382–390 s | Arsenal delivers the scoring corner around 383–385 s | Cross at 383.84 s (0.977), shot at 387.52 s (0.202). Goal was outside the top five classes. |
| United v City, 103–111 s | United crosses from the right around 106.8–107.0 s | Cross at 106.92 s (0.116), ranked fifth by peak score. |
| United v City, 89–97 s | Midfield circulation | Pass at 91.56 s (0.773), the strongest event. |

This checks timing on selected windows. It does not measure full-match recall,
identify the taker or scorer, or establish that every detected action belongs
to the same move. Source: [SoccerNet/sn-teamspotting](https://github.com/SoccerNet/sn-teamspotting).

To rerun the scoring-corner window with the saved local assets:

```bash
sandbox-exec -p '(version 1)(allow default)(deny network*)' \
  python evals/local_spotter.py data/models/football-spotter-2025 data/clips/clip2/clip2.mp4 382
```

## Model choice for the action log

One 6 s window (Arsenal v Manchester United, 02:00), slowed to half speed. Ground
truth from the frames: an Arsenal player receives along the near touchline facing
forward; United #37 wins the loose ball at 02:07.

| Model | Scoreboard read | Receptions |
|---|---|---|
| Gemini 4 Argon | n/a | 3, plausible and specific |
| Gemini 3.8 Flash | n/a | 2, plausible |
| Gemini 3.1 Pro | n/a | 3, but contradicts the frames (wrong flank); 91 s |
| Cosmos 3 Super Reasoner | ARS 1-0 MU, 1st half (correct) | made up: #2 and #22 alternating every second |
| Cosmos 3.5 Ultra | ARS 1-0 MU, 1st half (correct) | made up: #2 on every second |

Cosmos models need the video as `video_url` on this endpoint, not as a `file` part.

## Identity

| Method | Actions with a shirt number (Arsenal clip) |
|---|---|
| Action log alone (video) | 15% of receptions |
| + crops of 3 frames over ±0.5 s | 109 of 210 |
| + crops of 5 frames over ±1 s, roster check, carry-over | 124 of 210 |

An audit of 8 crop-identified actions: every legible number on the crop matched
the number used. In one crop, two players (ARS #23, MU #37) overlapped.

## Roster from shirts (no squad sheet)

Arsenal clip: 27 number-to-name pairs, all consistent with the shirts seen in the
frames. One-off misreads (for example "PEPE", "ONANA", "HEATON") are removed by
the vote. Man United v Man City: 17 pairs, after dropping Dalot's duplicate #20.

## Goals

| Clip | Scoreboard changes | Goals | Scorers |
|---|---|---|---|
| Arsenal v Man United | 1-0, 1-1, 1-2, 2-2, 2-3 | 5 of 5 | Mbeumo, Dorgu, Merino, Cunha correct; the 1-0 own goal has no scorer |
| Man United v Man City | 1-0, 2-0 | 2 of 2 | Mbeumo, Casemiro (not checked against frames) |

Replays never add goals, because the score does not change during a replay.

## Free-text cutoff

| Clip | Weakest football query | Strongest off-topic query | Cutoff |
|---|---|---|---|
| Arsenal v Man United | 1.56 | -6.83 | -4.73 |
| Man United v Man City | 1.56 | -6.26 | -4.30 |

## Frame embeddings (tried, not used)

All 204 frames of the Arsenal clip embedded with `llama-nemotron-embed-vl-1b-v2`
(the only VL embedder on the endpoint that accepts images; Qwen3-VL-Embedding
takes the image as text). Top-5 hits within 4 s of frame-checked times:

| Query | Frame embeddings | Text search over action descriptions |
|---|---|---|
| corner kick being taken | 0/5 | 5/5 |
| goalkeeper diving save | 1/5 | 3/5 |
| defensive wall at a free kick | 0/5 | 1/5 |
| switch of play | 1/5 | 0/5 |
| counter attack | 0/5 | 1/5 |

Best similarity for "switch of play" was 0.15 and for "recipe for pasta" 0.04, so
frame scores are too flat to set a cutoff. A single frame cannot show motion, and
broadcast wide shots look alike.

## Fixes from review of the live reels

- A dribble at 03:17 was credited to Cunha from one "10" read among four blank
  crops; it was Mbeumo. Crop identity now needs 2 agreeing reads, and identity
  carries from an interception to the next touch.
- Mbeumo's goal appeared three times: at 02:49 and 03:41 as replays from the high
  tactical camera, which the vision model labels as live. Goal sightings are now
  reconciled with the scoreboard (one live sighting per score change).
- Clicking a result started 8 s early (a lead-in kept from the caption pipeline),
  so Cunha's goal started on the bench shot at 07:32. It is now 1 s.

## Player tracking (Arsenal v Man United)

Roboflow's football YOLOv8 detector plus ByteTrack, run locally on an M4 Pro (Apple
GPU): 3,942 frames of live play (315 s at 12.5 fps) in about 13 minutes, 1 GB of
memory, about 0.15 CPU cores (capped at 4 threads).

| | |
|---|---|
| Actions matched to the acting player's track | 152 of 162 |
| Shirt-colour clusters | 2, mapped to ARS and MU by the actions' teams |
| Tracks after splitting at team changes | 1,188, median 1.6 s, 90th percentile 7.6 s |

| Identity source | Before tracking | + track votes from action crops | + reads along tracks |
|---|---|---|---|
| Actions with a player number | 83 of 162 | 93 of 162 | 108 of 162 |

Spot check of 7 track-identified actions at the action frame: the box sat on a
plausible actor (right team, on the ball or in the duel) in 6; the number itself
was visible in 1 (#23 Merino), the others got it from elsewhere on the track. In
one, the ball was nearer a United player than the Arsenal receiver.

Known weakness: ByteTrack matches by position only and swaps IDs when players
cross during camera pans. Splitting at team changes fixes cross-team swaps, not
swaps between teammates.

## Passage search regression on the local reels

Run `uv run --offline evals/passage_search.py --details` from the repository root with both
rebuilt clip indexes present. The evaluator sends each unique query through
FastAPI's `/api/search` route on the complete reel. Only the manually reviewed
windows count toward the metrics. Returned clips elsewhere on the reel have
no judgment. A positive counts when one of the first five returned spans
has temporal intersection-over-union of at least 0.5 with the marked move.
A negative counts when the query is supported and no returned clip touches
that negative window. The two off-ball probes instead require an explicit
unverified status and an explanation.

The 15 `corner_cases.jsonl` judgments are the same ones used for the local
TimeLens2 study. The first six in `passage_cases.jsonl` cover winning possession,
progressing and shooting, wrong team, a stoppage/replay, and unsupported
off-ball causation. On United v City, United regains the ball around 119 s,
progresses down the right and shoots around 129 s. In Arsenal v United,
the 532–553 s window shows an Arsenal attack, stoppage and replay. It is a
negative for a United regain and shot.

Four more judgments cover the opening Arsenal header and save, two switches
followed by shots, and a United switch around 107 s that ends with City
winning the ball. The shot at 254.9 s on the United v City reel was missing
from the extracted actions. Local frame review supplied its timestamp and team.
That positive tests retrieval from a corrected index.

| Measured outcome | Result |
|---|---:|
| Positive queries with the reviewed passage among pre-verification candidates | 10/10 |
| Positive queries with the reviewed passage in the top five | 10/10 |
| Top-five hits also covering both ends of the move within two seconds | 10/10 |
| Extra top-five copies of those hits | 0 |
| Hard-negative windows correctly excluded | 13/13 |
| Off-ball probes explicitly identified as unverified | 2/2 |

A local in-process run of 14 distinct clip/query pairs took a median of about
4.5 ms per request after index loading. This excludes video loading and analyst
watch time. These are development-set regressions on two highlight reels,
with overlapping windows, repeated plays and paraphrases. The first Arsenal
corner and the replay include human-reviewed source corrections. The metrics
do not measure automatic recognition of an unseen replay, precision over
the entire reel, off-ball retrieval accuracy, or generalization to new matches.
