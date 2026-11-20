# Situation Search

Search football video for a passage of play and play it from the initiating
action through the outcome. Each structured result includes the observations
and timestamps that satisfied the query. Search uses the local clip index and
runs without an inference service or API key.

## Run the app

From the project root:

```bash
./run_server.sh
```

Open http://127.0.0.1:8008. The launcher uses uv to resolve Python dependencies
from `requirements.txt`. Once the wheels are cached,
`UV_OFFLINE=1 ./run_server.sh` runs with dependency downloads disabled.
Videos and indexes live under `data/clips/<clip>/` and are git-ignored.
The two local evaluation clips are Manchester United v Manchester City
(`seed`) and Arsenal v Manchester United (`clip2`).

## Search

| Query | Evidence used |
|---|---|
| Live Arsenal corner followed by a saved shot | A live delivery, continuation of play, the attacking shot, and its outcome |
| Corner that ended in a goal by Merino | A corner, a linked goal and the scorer's identity |
| Corner that did not lead to a goal | The terminal outcome of the corner passage |
| Manchester United regain possession, advance and shoot | A regain, a later progressive action, and a same-team shot |
| Goalkeeper saves a header | A headed attempt linked to the opposing keeper’s save |
| Switch of play followed by a shot | A long delivery across the pitch and a later attempt in the same attack |
| Long passes under tight pressure | Length and pressure on the same indexed pass |
| Saka | A reel of identified touches and goals |

The query parser distinguishes taker, shooter, scorer, and keeper. It matches
team aliases and checks action order. A request containing an unsupported
condition, such as an off-ball run drawing a defender away, is labeled
unverified. Local description hits or optional visual candidates can help an
analyst inspect the footage. They do not establish a verified tactical claim.

`GET /api/search?q=...&clip=...` searches a match.
`diagnostics=true` includes candidate spans before the final conditions.
`GET /api/visual-search?q=...&clip=...` retrieves review candidates using a
separate local frame index.

## Build from saved observations

The default build runs deterministic processing on the clip's existing
`vision.json`, using any available track, shirt-read, and scoreboard artifacts.
It rebuilds the action index and passages without calling a model endpoint:

```bash
CLIP_VIDEO=data/clips/clip2/clip2.mp4 uv run --no-project python pipeline/build.py
CLIP_VIDEO=data/clips/seed/seed.mp4 uv run --no-project python pipeline/build.py
```

The action index retains timestamps, observation sources, and available player
identities. Set pieces can continue through a deflection or recycled possession.
The possession index links attempts to observed opposing saves within three seconds.
A switch can continue through a defensive clearance when the delivering team
visibly receives the loose ball within four seconds and the opponent has no
controlled action in between. A goalkeeper gathering a loose ball supplies
no shot evidence on its own.
A restart, observed opposition possession, gap, or non-live segment ends the
passage. A counterattack requires a defensive or midfield regain followed by
progression and a shot. Attacking-third regains are separately indexed.
Content-addressed reviews in `pipeline/reviews/` can correct a replay or
supply a verified observation for the matching video. Coverage on other
videos depends on the quality of their extracted observations.

Optional visual retrieval needs the locally provisioned CLIP model in
`data/models/clip-search/`, including `tokenizer.json`,
`preprocessor_config.json`, and both quantized ONNX encoders in `onnx/`.
Build the frame indexes locally with:

```bash
uv run --no-project --with-requirements requirements.txt python pipeline/visual_index.py
```

## Extract observations for new footage

`pipeline/build.py --extract-with-endpoint` runs the original video ingestion
workflow. Its vision and OCR stages send video windows, images, and player
crops to the inference endpoint configured by the operator. Use the saved-
observation workflow above for local-only rebuilding. For footage authorized
for the configured endpoint, ingestion requires ffmpeg, the football detector
at `data/models/football-player-detection.pt`, and the endpoint settings shown
in `.env.example` exported in the process environment.

The extraction stages survey team kits, log on-ball actions and field zones,
read the scoreboard, run a local detector and ByteTrack, and identify actors
from shirt crops and track votes. A missing shirt number or a camera cut can
leave a participant unknown. Broadcast highlights also contain replays, which
can be mislabeled as live play.

[Evaluation and local model experiments](EVAL.md) · [Container instructions](DEPLOY.md)
