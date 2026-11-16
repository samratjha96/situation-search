# Situation Search

**Live demo:** https://situsearch.bootloop.cc

A football analyst spends hours scrubbing match video to find one moment.
Situation Search removes the scrubbing. The analyst types what they want to see,
such as "corner that led to a goal", "receiving on the half turn" or a player's
name, and the app goes to those moments in the clip. If nothing matches, it says
so.

Vision models do all the perceiving. Everything after them is a fixed,
inspectable processing pipeline. The app uses no commentary, no captions, and no
hand-made squad sheet.

The demo runs on two public highlight reels:
[Manchester United v Manchester City](https://www.youtube.com/watch?v=mHcRPO21qlE)
and [Arsenal v Manchester United](https://www.youtube.com/watch?v=8I4Vov6p1jI).
Nothing in the pipeline is specific to these clips, to broadcasts, or to
footage with a scoreboard.

## What it does

- **Player reels.** Search a name ("Kobbie Mainoo", "Saka") to get every snippet
  where that player is on the ball. "Play reel" plays them back to back.
- **Set pieces with outcomes.** "corner that led to a goal" returns each corner as
  one episode (delivery to outcome) and filters on the outcome.
- **Actions with detail.** "receiving on the half turn", "Fernandes receiving the
  ball", "shot on target", "goalkeeper diving save".
- **Patterns.** "switch of play" and "counter attack", derived from where actions
  start and end.
- **Goals.** One per scoreboard change, credited to the last shot before it.
- **Free text.** Anything else ("a long ball over the top") is matched against
  the vision model's description of each action, and returns "no match" below a
  per-clip cutoff ("recipe for pasta").

## How it works

### Indexing (offline, once per clip): `pipeline/build.py`

| Step | Kind | What happens |
|---|---|---|
| 1. frames | ffmpeg | A thumbnail every 3 s. |
| 2. vision | vision model | **Team survey:** frames from across the clip; the model names both sides (from the scoreboard, or by kit or bib colour) and describes each kit. **Action log:** 6 s windows every 3 s, slowed to half speed so the model's ~1 fps sampling sees ~2 fps of play. For each window: shot type (live, replay, close-up, ...), every on-ball action (time, action, team, shirt number or name *if read*, body shape on receptions, start and end zones, pass length, outcome, pressure, a one-sentence description), and every shirt it can read, replays included. |
| 3. scoreboard | OCR model | The top of every frame. Footage without a scoreboard just yields no scores. |
| 4. process | fixed | Roster, dedup, goals, set pieces, patterns (below). |
| 5. identity | vision model | For each action: 5 full-resolution frames over ±1 s, the model boxes the player doing it, the box is cropped and enlarged 6x, and the model reads the number and shirt colour from the crops alone. |
| 6. process | fixed | Again, with the crop reads. |
| 7. embed | embedding model | The action descriptions, for free-text search. |
| 8. calibrate | fixed + reranker | The per-clip "no match" cutoff. |

**Fixed processing** (`pipeline/process.py`, no model calls):

- **Roster.** Majority vote over every shirt read: a number becomes a name after at
  least 2 sightings and 60% of that number's votes, and a name keeps only its most
  seen number. On the Arsenal clip this produced 27 correct pairs without a squad
  sheet.
- **Identity per action.** In order of preference:
  1. *crop*: the readable crop reads whose shirt colour matches the team agree.
  2. *shirt*: the action log read it.
  3. *carried*: a receive followed by a pass, dribble or shot by the same team
     within 2.5 s is the same player.

  A number missing from a well-populated roster is treated as a misread.
- **Dedup.** Overlapping windows see each action twice; one copy is kept.
- **Goals.** Each score increment that holds for 2 reads is a goal. The scorer is
  the scoring team's last shot or header in the 45 s before it. Without a
  scoreboard, goals are the ones the vision model saw go in.
- **Set pieces.** A corner, free kick or penalty, plus the actions in the next 10 s.
  The outcome is goal (only if the scoreboard confirms one), save, shot or
  cleared.
- **Switch of play.** A long pass or cross from one flank to the other.
- **Counter-attack.** The ball is won in the defensive or middle third and the same
  team shoots within 12 s.

### Search (live): `app/server.py`

1. **A name alone** returns the player's reel: each of his actions padded to ±3 s,
   with snippets less than 2 s apart merged.
2. **Football vocabulary** becomes filters on the index: set pieces, patterns,
   goals, actions, body shape, outcome, team, and player. The reranker orders
   the hits.
3. **Anything else** is a semantic search over the action descriptions, then
   reranking, then the clip's cutoff.

## Run it

```bash
# Build any clip; re-running skips finished work.
CLIP_VIDEO=data/clip2.mp4 python pipeline/build.py   # -> data/clips/clip2/
./run_server.sh          # serves every data/clips/<clip>/index.json; http://127.0.0.1:8008
```

The endpoint URL, key and model routes come from a git-ignored `.env` (see
`.env.example`). For Docker and hosting, see `DEPLOY.md`.

## Models

| Role | Model |
|------|-------|
| Team survey, action log, crop identity | Gemini 4 Argon |
| Scoreboard OCR | PaddleOCR-VL |
| Action description embeddings | NV-EmbedQA 1B |
| Reranker | NV-RerankQA 1B |

Model choices were tested against frames checked by eye (see `EVAL.md`). Cosmos 3
Super Reasoner and Cosmos 3.5 Ultra read the scoreboard correctly but produced
made-up action lists on the same window, so they are not used.

## Limitations

- **Identity depends on seeing the back of the shirt.** On the Arsenal clip, 124 of
  210 actions have a player number. Players facing the camera stay unnamed, so a
  player's reel shows only part of his involvement (Mainoo: 1 snippet).
- **The boxed player is not always the one acting.** Readings on the crops were
  right in an 8-crop audit, but in one crop two players overlapped.
- **Zones and pitch geometry are the vision model's estimate.** No pitch-keypoint
  model is used.
- **Highlight reels are mostly replays and close-ups.** Full-match footage gives
  far more live play.
- **The footage is a copyrighted highlight reel.** A public build needs licensed
  footage.

## Stack

FastAPI and vanilla JS. `ffmpeg` for cutting and cropping. NumPy for the cosine
search. Docker for deploy. All model inference runs on an OpenRouter-style
endpoint.
