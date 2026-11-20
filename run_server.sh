#!/bin/sh
set -eu
cd "$(dirname "$0")"
exec uv run --no-project --with-requirements requirements.txt \
  uvicorn app.server:app --host 127.0.0.1 --port 8008 "$@"
