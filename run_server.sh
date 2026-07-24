#!/bin/zsh
# Launch the Situation Search server with the inference endpoint key loaded,
# so query-time semantic search works.
cd /path/to/situation-search
eval "$(grep -E '^[[:space:]]*export[[:space:]]+OPENROUTER_API_KEY=' ~/.zshrc | head -1)"
export OPENROUTER_API_KEY
exec uv run --with fastapi --with uvicorn --with numpy --with openai python app/server.py
