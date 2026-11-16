# Evaluation

All checks below were made against frames looked at by eye. The clips are the two
demo highlight reels at 1280x720.

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
