"""Set each clip's "no match" cutoff for free-text search, with no hand labels.

Any soccer footage (match or training) has players passing, receiving and shooting,
and none has pasta. Free-text search scores both kinds of query against the
vision model's action descriptions; the cutoff goes a quarter of the way up from
the strongest off-topic score to the weakest football score. Writes
data/clips/<clip>/search.json.

    python pipeline/calibrate.py            # all clips
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))
import server  # noqa: E402

POSITIVE = ["a player passes the ball to a teammate", "a player controls the ball",
            "a player shoots towards goal", "a defender wins the ball back", "a player runs with the ball"]
JUNK = ["recipe for pasta", "basketball slam dunk", "weather forecast", "cat asleep on a sofa",
        "tennis serve", "stock market chart", "a car driving on a motorway"]

for name in server.CLIP_ORDER:
    c = server.CLIPS[name]

    def best(q):
        ranked = server.semantic(c, q)
        return ranked[0][0] if ranked else -1e9

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
