"""Embed each action's vision-written description for free-text search.

Writes data/clips/<clip>/embeddings.json: {"model": ..., "vectors": {action_id: [...]}}.

    CLIP_VIDEO=data/clips/clip2/clip2.mp4 python pipeline/embed.py
"""
import json
import sys
from pathlib import Path

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import CLIP_DIR, EMBED_MODEL, OPENROUTER_API_KEY, OPENROUTER_URL

client = OpenAI(base_url=OPENROUTER_URL, api_key=OPENROUTER_API_KEY)


def text(a):
    who = a["player"] or (f"{a['team']} #{a['number']}" if a["number"] else f"{a['team']} player")
    return f"{a['action'].replace('_', ' ')} by {who}. {a['detail']}"


def main():
    actions = json.loads((CLIP_DIR / "index.json").read_text())["actions"]
    vecs = {}
    for i in range(0, len(actions), 32):
        batch = actions[i:i + 32]
        r = client.embeddings.create(model=EMBED_MODEL, input=[text(a) for a in batch],
                                     encoding_format="float", extra_body={"input_type": "passage"})
        vecs.update({a["id"]: d.embedding for a, d in zip(batch, r.data)})
    (CLIP_DIR / "embeddings.json").write_text(json.dumps({"model": EMBED_MODEL, "vectors": vecs}))
    print(f"{len(vecs)} action embeddings")


if __name__ == "__main__":
    main()
