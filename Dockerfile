# Serves the pre-built index. All heavy AI is at index time; serving only needs a
# small web server + query-time calls to the inference endpoint (key passed at runtime).
FROM python:3.12-slim

WORKDIR /app
RUN pip install --no-cache-dir fastapi uvicorn "numpy<2.3" openai httpx

# app code + the built per-clip indexes (each clip has index.json, frames, video)
COPY app/ ./app/
COPY data/clips/ ./data/clips/

EXPOSE 8008
# Endpoint + key are supplied at runtime: docker run --env-file .env
# (.env sets OPENROUTER_URL and OPENROUTER_API_KEY)
CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8008"]
