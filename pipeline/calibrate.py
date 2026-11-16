"""Set each clip's "no match" cutoff from queries that need no hand labels.

Every football highlight reel contains goals, shots and saves, and none contains
pasta. The cutoff is the midpoint between the best reranker score of the weakest
football query and the strongest off-topic query, a quarter of the way up from junk:
tactical queries ("switch of play") legitimately score well below "goal". Writes data/clips/<clip>/search.json.

    python pipeline/calibrate.py            # all clips
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))
import server  # noqa: E402

POSITIVE = ["goal", "shot on goal", "goalkeeper save", "players celebrate a goal",
            "attack down the wing"]
JUNK = ["recipe for pasta", "basketball slam dunk", "weather forecast", "cat asleep on a sofa",
        "tennis serve", "stock market chart", "a car driving on a motorway"]

server.SHOW_GAP = 0.0
for name in server.CLIP_ORDER:
    c = server.CLIPS[name]
    c.no_match = -1e9  # let every query return its best candidate

    def best(q):
        res = json.loads(server.search(q=q, clip=name).body)["results"]
        return res[0]["relevance"] if res else -1e9

    pos = {q: best(q) for q in POSITIVE}
    junk = {q: best(q) for q in JUNK}
    lo, hi = min(pos.values()), max(junk.values())
    cut = round(hi + (lo - hi) / 4, 3)
    print(f"{name}: weakest football {lo:.2f} ({min(pos, key=pos.get)}), "
          f"strongest junk {hi:.2f} ({max(junk, key=junk.get)}) -> cutoff {cut}")
    if lo <= hi:
        print(f"  WARNING: junk outscores football on {name}; the cutoff cannot separate them")
    (server.DATA / "clips" / name / "search.json").write_text(json.dumps(
        {"no_match_threshold": cut, "positive": pos, "junk": junk}, indent=2))
