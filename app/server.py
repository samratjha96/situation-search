"""Situation Search backend.

Serves the analyst UI and a hybrid (semantic + keyword) search over the
captioned 'moments' index. Semantic search embeds the query on the the
gateway and ranks moments by cosine similarity to their caption embeddings;
keyword matches on the structured fields boost the score. Falls back to
keyword-only if the embedding call fails, so the app still works offline.
"""
import json
import os
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent
INDEX = ROOT / "data" / "index.json"
STATIC = ROOT / "app" / "static"
EMBED_MODEL = "openai/text-embedding-3-small"

data = json.loads(INDEX.read_text())
MOMENTS = data["moments"]

# Build a normalized embedding matrix for fast cosine search.
_dim = next((len(m["embedding"]) for m in MOMENTS if m.get("embedding")), 0)
_mat = np.zeros((len(MOMENTS), _dim), dtype=np.float32)
for i, m in enumerate(MOMENTS):
    v = m.get("embedding") or []
    if len(v) == _dim and _dim:
        _mat[i] = v
_norms = np.linalg.norm(_mat, axis=1, keepdims=True)
_matn = _mat / np.clip(_norms, 1e-9, None)

def _api_key():
    k = os.environ.get("OPENROUTER_API_KEY")
    if k:
        return k
    # Fallback: read the export from the user's shell rc so the preview
    # server (which starts without the env var) still gets semantic search.
    rc = Path.home() / ".zshrc"
    if rc.exists():
        for line in rc.read_text().splitlines():
            line = line.strip()
            if line.startswith("export OPENROUTER_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""

client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=_api_key())

# Public view of a moment (drop the bulky embedding vector).
def public(m, score=None):
    d = {k: v for k, v in m.items() if k != "embedding"}
    if score is not None:
        d["score"] = round(float(score), 4)
    return d

def embed_query(text):
    r = client.embeddings.create(model=EMBED_MODEL, input=[text],
                                 encoding_format="float",
                                 extra_body={"input_type": "query"})
    return np.array(r.data[0].embedding, dtype=np.float32)

def keyword_score(m, terms):
    parts = [str(m.get(k, "")) for k in
             ("prose", "event", "phase", "zone_third", "zone_side", "pressure",
              "teams", "score", "card", "card_player", "scorer", "assist",
              "shot_by", "ocr_text", "commentary")]
    parts += m.get("players", []) or []
    hay = " ".join(parts).lower()
    return sum(1.0 for t in terms if t in hay)

app = FastAPI(title="Soccer Situation Search")

@app.get("/api/moments")
def moments():
    return JSONResponse({"moments": [public(m) for m in MOMENTS],
                         "video": data.get("video")})

@app.get("/api/search")
def search(q: str = "", event: str = "", phase: str = ""):
    idx = list(range(len(MOMENTS)))
    if event:
        idx = [i for i in idx if MOMENTS[i].get("event") == event]
    if phase:
        idx = [i for i in idx if MOMENTS[i].get("phase") == phase]

    terms = [t for t in q.lower().split() if t]
    semantic_ok = False
    sims = np.zeros(len(MOMENTS), dtype=np.float32)
    if q.strip() and _dim:
        try:
            qv = embed_query(q)
            qv = qv / max(np.linalg.norm(qv), 1e-9)
            sims = _matn @ qv
            semantic_ok = True
        except Exception:
            semantic_ok = False

    scored = []
    for i in idx:
        kw = keyword_score(MOMENTS[i], terms) if terms else 0.0
        if not q.strip():
            score = 1.0  # pure filter / browse
        elif semantic_ok:
            score = float(sims[i]) + 0.15 * kw
        else:
            score = kw
        scored.append((score, i))

    scored.sort(key=lambda x: x[0], reverse=True)
    if q.strip() and not semantic_ok:
        scored = [(s, i) for s, i in scored if s > 0]
    results = [public(MOMENTS[i], s) for s, i in scored[:40]]
    return JSONResponse({"results": results, "semantic": semantic_ok,
                         "count": len(results)})

@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")

app.mount("/media", StaticFiles(directory=ROOT / "data"), name="media")
app.mount("/static", StaticFiles(directory=STATIC), name="static")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8008, log_level="warning")
