# Serves the pre-built index. All heavy AI is at index time; serving only needs a
# small web server + query-time calls to the inference endpoint (key passed at runtime).
FROM python:3.12-slim

WORKDIR /app
RUN pip install --no-cache-dir fastapi uvicorn "numpy<2.3" openai httpx

# app code + the built index (index.json, the clip, and frame thumbnails)
COPY app/ ./app/
COPY data/index.json ./data/index.json
COPY data/frames/ ./data/frames/
# the video the index points to (build arg lets you pick which clip to ship)
ARG CLIP=clip2.mp4
COPY data/${CLIP} ./data/${CLIP}

EXPOSE 8008
# OPENROUTER_API_KEY is supplied at runtime: docker run -e OPENROUTER_API_KEY=sk-...
CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8008"]
