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
