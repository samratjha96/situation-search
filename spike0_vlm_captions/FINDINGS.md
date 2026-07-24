# Spike 0 findings — can a local VLM caption soccer frames?

Seed: `data/seed.mp4` — Man United vs Man City highlights, 720p, ~10 min.
Frames: 21 stills, one every 30 s (`frames/frame_###.jpg`).
Note: highlights are edited close-ups. Good for event tests, poor for full
tactical geometry (rarely all 22 players in shot).

## Local model: Gemma 3 4B (4-bit) via MLX

- Runs fully local on the M4 Pro. Peak 5 GB RAM, ~32 tokens/s. Valid JSON.
- Verdict: NOT reliable enough for a search index.

Evidence (model output vs. what is actually on screen):

| Frame | Reality | Model said | Verdict |
|-------|---------|-----------|---------|
| 009 | Attack at edge of City box | phase="mid_third" (broke enum), "midfield" | Wrong third + bad schema |
| 010 | Stationary set-piece, Maguire | event="duel", "Rashford in a duel" | Hallucinated event + player |
| 012 | Goal celebration, 1-0 | phase="progression", "Rashford + staff member" | Coarse hit, wrong details |
| 015 | Shot at near post, 2-0 | event="shot", att_third | Correct |

Failure modes:
1. Mislabels the tactically important fields (phase, pitch zone).
2. Hallucinates specifics — player names, events that are not happening.
3. Breaks the enum schema.
4. Ignores on-screen cues (scoreline that disambiguates goal vs. play).

Coarse gist correct roughly half the time. Not good enough — a search index
built on this retrieves the wrong clips.

## Options up in quality

- A. Bigger local model (Gemma 3 12B / 27B). Slower, heavier, stays offline.
- B. Offload to a strong hosted VLM on the inference endpoint
     (`openrouter.ai/api`, OpenAI-compatible). Candidates:
     `google/gemini-2.5-pro`, `anthropic/claude-3.5-sonnet`,
     `openai/gpt-4o`. Reachable; needs `OPENROUTER_API_KEY`.

## Recommended architecture: tiered inference

Captioning is a one-time, offline INDEXING pass per video — not real-time.
So it is fine to use a strong hosted VLM to build the index, then run the
sidecars + search fully local on the laptop.

- Index time (heavy, occasional): hosted VLM for high-quality captions.
- Query time (light, always): local search over the sidecar index.
- Keep local Gemma as the offline / privacy fallback.

This mirrors FrameDex's own cli/api/local backends and fits the
"index on power, serve on a laptop" constraint.

## Next step

Set OPENROUTER_API_KEY, then re-run frames 009/010/012 through a hosted VLM and
compare to the table above. If hosted nails them, adopt the tiered design.
