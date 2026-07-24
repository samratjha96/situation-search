# Soccer Situation Search

A local, laptop-runnable tool for soccer video analysis. An analyst picks a
moment and gets ranked similar moments across a dataset. No manual footage
scrubbing.

The project is a portfolio piece aimed at an applied full-stack football
analytics role (build tools that sit in the analyst workflow), inspired by
Karun Singh's work (Expected Threat; self-supervised representations for
tracking data).

## Constraints

- Must run on a MacBook. No heavy GPU required.
- This machine: Apple M4 Pro, 48 GB RAM, macOS 26.5.
- Local models only. No paid cloud inference in the core path.

## Approach: hybrid situation search

Two complementary retrieval layers. Each moment gets a sidecar with both.

1. VLM captions (FrameDex-style). A local Gemma vision model reads sampled
   frames and writes structured English labels (phase, event, zone, pressure).
   Good for semantic and event search. Weak at fine tactical geometry.
2. Tracking-geometry embeddings. A small model encodes the positions of all
   players over a short window. Good for spatial shape and movement. Needs
   tracking data.

Build order: prove the concepts first, then the app.

## Spikes (prove theories cheaply, riskiest first)

- `spike0_vlm_captions/` — Go/no-go. Can local Gemma read a soccer frame
  tactically? Feed a few frames, judge the structured output by eye.
- `spike2_geometry/` — Does geometry similarity retrieve tactically similar
  moments? Uses Metrica open tracking data. No video, no model download.
- Spike 1 (later) — real pipeline slice: match clip -> frames -> Gemma
  sidecars -> query.

## Stack (all Mac-local)

- Data: Polars / pandas
- VLM: Gemma 3 vision (12B) via Ollama
- Embeddings: PyTorch (MPS backend)
- Retrieval: FAISS (CPU) or hnswlib
- Video: ffmpeg
- App (later): FastAPI + React/TypeScript

## Data sources (open)

- StatsBomb open event data — for reproducing xT (domain warm-up).
- Metrica Sports open tracking data — full 22 players + ball.
- SoccerNet — broadcast clips for the VLM layer (needs quick registration).

## Status: working webapp (v0)

An end-to-end Situation Search webapp runs on the laptop.

Pipeline: `data/seed.mp4` (a match clip) -> ffmpeg samples a frame every 10 s
-> `pipeline/caption.py` sends each frame to a hosted VLM for a structured
soccer caption and embeds the prose -> `data/index.json`. The FastAPI app
(`app/server.py`) serves an analyst UI (`app/static/index.html`) with hybrid
semantic + keyword search. An analyst types plain English and gets ranked
moments; clicking one jumps the video.

### What v0 proves (de-risked)

- A local 4B VLM (Gemma 3 4B via MLX) is NOT reliable for tactical captions:
  it hallucinates events and player names and mislabels pitch zones.
  See `spike0_vlm_captions/FINDINGS.md`.
- A hosted frontier VLM (Gemini 3.1 Pro on the inference endpoint) IS reliable:
  correct phase/zone/event, reads the on-screen scoreline, no name guessing.
- Semantic English search returns the right moments. Examples that land on
  the correct clip: "goalkeeper makes a save" -> the save at 04:30;
  "VAR check" -> the VAR moment at 09:20; "goal celebration" -> the
  celebrations. Ranking uses cosine similarity over caption embeddings.
- The design is tiered: heavy captioning is a one-time offline indexing pass
  (hosted VLM); query-time search runs locally on the laptop.

### v0.2 — multimodal player attribution (who scored / who shot)

Each moment is now captioned from BOTH the video frame AND the commentary
window around it, so moments carry player names.

- Commentary: local Whisper (large-v3-turbo, MLX) transcribes the audio track
  with timestamps. `pipeline/caption.py` passes the ~20 s window around each
  frame to the VLM.
- Fusion: the VLM uses the video for the visual event/zone and the commentary
  for player identity, normalizing phonetic ASR spellings (e.g. "Bumo" ->
  "Bryan Mbeumo") with football knowledge.
- Commentary is OPTIONAL. With no transcript, or a silent moment, the model
  captions from video alone and sets players=[] / scorer=null. It never
  invents names without audio or a clear on-screen graphic.
- The UI shows a scorer / assist / player chip row; names are in the search
  corpus, so "who scored" and player-name queries work.

What works: the real goals attribute correctly (Amad ~02:20, Mbeumo ~05:00,
Mount ~08:30). What is still noisy: on celebration / replay frames the model
sometimes assigns the wrong scorer, and it occasionally names a player not in
the match (misheard commentary). Fix path below.

### v0.3 — per-player highlights

An analyst can pull one player's big moments out of a clip.

- The backend groups phonetic / partial name variants into canonical players
  (fuzzy match on the caption names), then ranks each player's moments by an
  event-importance weight (goal > assist > shot > card > ... > buildup) with a
  bonus when the player is the protagonist (scorer/assist/shot_by/card_player).
- Endpoints: `/api/players` (roster with goals/assists/shots/cards) and
  `/api/player?name=` (one player's moments, ranked). Deep-link: `?player=Name`.
- UI: a Players panel; click a player to see their ranked highlights with a
  stat line. Each attribution shows its provenance (via commentary / jersey).
- Honest limit: player stats inherit the attribution noise (e.g. a wrong scorer
  on a replay frame inflates a goal count). The fix is the same as above —
  event-level attribution, not per-frame.

### v0.4 — event-level attribution (the accuracy fix)

Per-frame attribution double-counts: one goal spans a goal frame plus several
celebration/replay frames, and each frame independently guesses a scorer from a
narrow commentary window. Result: inflated, wrong stats (Bruno "3 goals",
Dorgu "5"). Frame-level is the wrong unit.

`pipeline/events.py` attributes at the EVENT level from two grounded signals:
- **OCR scorebug score timeline -> WHEN/WHETHER a goal happened.** The score
  increments exactly once per goal, so this is an objective, replay-proof goal
  detector.
- **Full commentary transcript -> WHO.** One holistic pass over the whole
  transcript names each scorer once (de-duplicated). Uses a reasoning model and
  line-delimited output (JSON was unreliable at length across models).

Goals = score increments, each attributed to the scorer named nearby. Other
events (cards, shots, saves, fouls, subs) come from the de-duplicated
commentary list. Each event aligns to the nearest frame for playback. Player
stats are then counted from this event list, not from frames.

Result on the seed clip: 5-8 phantom goals -> 2 confirmed; Bruno 3 goals -> 0
goals / 1 assist (correct); Mbeumo goal correct.

Degrades gracefully: no scorebug -> commentary goals; no commentary ->
score-change goals with no name; silent + no scorebug -> events from video only.

Known refinements: (1) cross-spelling name matching — the reasoning model's
"Bryan Mbeumo" does not yet merge with the frame model's "Mbumo"; (2) sparse
1-frame/10s sampling mislocates the exact score-change second and can miss a
goal — denser sampling near score changes fixes both WHEN and WHO alignment.

### v0.5 — video-native event detection (works on training footage)

Goal: attribute events on ANY clip an analyst uploads — training-ground footage
with no scoreboard, no commentary, no name graphics. `pipeline/video_events.py`
chunks the clip and, per chunk, sends a burst of sampled frames to a VLM that
reads the motion and returns events with best-effort identity (jersey number or
shirt-colour + position, never a guessed name). Identity degrades gracefully:
name (broadcast) -> number -> colour+position -> (future) a stable track ID.

Two hard-won findings from verifying this (analyst-grade rigor):
- The gateway (OpenAI-compatible proxy) SILENTLY DROPS video sent via video_url.
  Gemini then confabulates a generic buildup->shot->goal from priors — it
  invented scorelines ("FRA 0 SCO 0") and player names ("De Bruyne", "Haaland")
  for a clip that was 0-0 possession. Impressive-looking, entirely fake.
  Verified by asking a pure-perception question: video_url got shirt colours and
  the scoreboard WRONG; a single image_url got both exactly right.
- Fix: send a BURST of real frames as images (the gateway does process images),
  with a calibrated prompt that may answer "nothing notable" and must not assume
  a shot/goal. Result: confabulation gone; chunks return empty when quiet; a
  0-0 possession chunk correctly reports "duel" instead of a phantom goal.

Lesson baked into the repo: every layer looks great on a cherry-picked demo;
only adversarial verification against ground truth tells you if it is real.

### Tracking backbone (de-risked, not yet built)

YOLO + ByteTrack runs locally (MPS) and detects players + ball, but out-of-the-
box IDs fragment (42 IDs for ~15 players in 6s; ball barely seen in wide shots).
Consistent "Player 1..N throughout" needs appearance re-ID + a stronger detector,
and is far easier on a static training camera than on broadcast (cuts/pan/zoom).
Treated as a flagship CV investment, not a quick win.

### Models used

- VLM captions (batch): `google/gemini-2.5-flash` (fast, parallelised)
- VLM captions (highest quality, single checks): `google/gemini-2.5-pro`
- Commentary ASR: `mlx-community/whisper-large-v3-turbo` (local, MLX)
- Embeddings (search): `openai/text-embedding-3-small`
- Local VLM fallback (tested, not good enough alone): `mlx-community/gemma-3-4b-it-4bit`

### Run it

```
# 1. Extract frames + commentary audio
ffmpeg -i data/seed.mp4 -vf "fps=1/10,scale=960:-1" data/frames/f_%04d.jpg
ffmpeg -i data/seed.mp4 -vn -ac 1 -ar 16000 data/seed_audio.wav

# 2. Transcribe commentary (local, optional — skip for silent footage)
uv run --with mlx-whisper mlx_whisper data/seed_audio.wav \
  --model mlx-community/whisper-large-v3-turbo --language en \
  --output-format json --output-dir data/transcript

# 3. Build the per-frame index (needs OPENROUTER_API_KEY; resumable, parallel)
uv run --with openai python pipeline/caption.py

# 4. Attribute events (accurate player stats): scorebug goals + commentary WHO
uv run --with openai python pipeline/events.py

# 5. Start the app
./run_server.sh        # or: .venv/bin/python app/server.py
# open http://127.0.0.1:8008
```

The server reads `OPENROUTER_API_KEY` from the environment, or falls back to the
export in `~/.zshrc`, so query-time semantic search works.

### Known limits / next

- Seed video is a copyrighted YouTube highlight reel — fine for a spike, but a
  shareable portfolio build must use SoccerNet or other licensed/open footage.
- Highlights are close-ups: good for events, poor for full tactical geometry.
  The tracking-geometry layer (Spike 2, Metrica data) is still the plan for
  spatial/positional search.
- Moments are single frames sampled every 10 s. Next: possession-window clips
  and playback of the surrounding seconds, not just a seek point.
- Attribution accuracy: per-frame attribution on a narrow commentary window is
  noisy (wrong scorer on replay/celebration frames; occasional out-of-match
  names). Fix path: attribute at the EVENT level (detect the goal/shot, then
  reason over the whole surrounding commentary once), corroborate identity with
  jersey-number OCR from the frame, and constrain names to the match lineup.
- ~4 of 62 frames still fail JSON parse from the fast model even with retries;
  use the pro model or a JSON-repair step for those.
- Frontend is vanilla JS for speed. Migrating to Vite + React + TypeScript is
  the productionization step (matches the target role's stack).
