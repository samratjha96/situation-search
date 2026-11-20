FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim
WORKDIR /app
COPY requirements.txt .
RUN uv venv /opt/serve && uv pip install --python /opt/serve/bin/python -r requirements.txt
COPY app/ app/
EXPOSE 8008
CMD ["/opt/serve/bin/uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8008"]
