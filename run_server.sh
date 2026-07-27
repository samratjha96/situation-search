#!/bin/zsh
# Launch the Situation Search server locally.
# Reads the inference endpoint + key from a local .env (git-ignored):
#   OPENROUTER_API_KEY=...
#   OPENROUTER_URL=https://.../v1
cd /path/to/situation-search
[ -f .env ] && set -a && source ./.env && set +a
export OPENROUTER_URL="${OPENROUTER_URL:-https://openrouter.ai/api/v1}"
exec uv run --with fastapi --with uvicorn --with numpy --with openai python app/server.py
