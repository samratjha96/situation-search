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

# 3. Build the index (needs OPENROUTER_API_KEY; resumable, parallel)
uv run --with openai python pipeline/caption.py

# 4. Start the app
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
