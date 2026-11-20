# /// script
# requires-python = ">=3.10"
# dependencies = ["fastapi", "httpx", "uvicorn"]
# ///
"""Evaluate the local search endpoint against reviewed passages. Run from the repository root."""
import argparse
from collections import Counter
import json
from pathlib import Path
from statistics import median
import sys
from time import perf_counter

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.server import app


def interval_iou(left, right):
    start = max(left[0], right[0])
    end = min(left[1], right[1])
    union = (left[1] - left[0]) + (right[1] - right[0]) - max(end - start, 0)
    return max(end - start, 0) / union if union > 0 else 0


def overlaps_window(span, case):
    if span[0] == span[1]:
        return case["start"] <= span[0] < case["end"]
    return max(span[0], case["start"]) < min(span[1], case["end"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--details", action="store_true", help="Print each query and its judgment")
    args = parser.parse_args()
    inputs = [ROOT / "evals" / name for name in ("corner_cases.jsonl", "passage_cases.jsonl")]
    counts = Counter()
    responses = {}
    latencies = []
    with TestClient(app) as client:
        loaded = {c["name"] for c in client.get("/api/clips").json()["clips"]}
        for path in inputs:
            for line in path.read_text().splitlines():
                case = json.loads(line)
                clip = Path(case["video"]).stem
                if clip not in loaded:
                    raise ValueError(f"Missing index for {clip}")
                key = (clip, case["query"])
                if key not in responses:
                    start = perf_counter()
                    response = client.get("/api/search", params={"clip": clip, "q": case["query"], "diagnostics": True})
                    response.raise_for_status()
                    responses[key] = response.json()
                    latencies.append(1000 * (perf_counter() - start))
                body = responses[key]
                status = body["status"]
                rows = body["results"]
                if "expected_status" in case:
                    passed = status == case["expected_status"] and bool(body.get("message"))
                    counts["uncertainty_cases"] += 1
                    counts["uncertainty_explained"] += passed
                else:
                    spans = [(r["t_seconds"], r.get("episode_end", r["t_seconds"])) for r in rows]
                    expected = case["expected"]
                    if expected:
                        # Each reviewed positive contains exactly one distinct play.
                        truth, = expected
                        top_indices = [rank for rank, span in enumerate(spans[:5], start=1)
                                       if interval_iou(span, truth) >= .5]
                        passed = status == "verified" and bool(top_indices)
                        counts["positives"] += 1
                        counts["localized_in_top5"] += passed
                        counts["top5_covers_both_ends"] += passed and any(
                            spans[rank - 1][0] <= truth[0] + 2 and spans[rank - 1][1] >= truth[1] - 2
                            for rank in top_indices
                        )
                        counts["extra_top5_copies"] += max(len(top_indices) - 1, 0)
                        candidates = body.get("candidate_spans", [])
                        counts["candidate_found"] += any(interval_iou(s, truth) >= .5 for s in candidates)
                    else:
                        # Search ran over the entire reel. Only this window has a negative judgment.
                        passed = status == "verified" and not any(overlaps_window(s, case) for s in spans)
                        counts["hard_negatives"] += 1
                        counts["hard_negatives_rejected"] += passed
                if args.details:
                    print(json.dumps({"clip": clip, "query": case["query"], "window_start": case.get("start"),
                                      "passed": passed, "status": status, "result_count": len(rows)}))
    print(json.dumps({"unique_queries": len(responses), "median_api_ms": round(median(latencies), 1),
                      **counts}, indent=2))
    if (any(counts[key] != counts["positives"] for key in
            ("localized_in_top5", "top5_covers_both_ends", "candidate_found"))
            or counts["extra_top5_copies"]
            or counts["hard_negatives_rejected"] != counts["hard_negatives"]
            or counts["uncertainty_explained"] != counts["uncertainty_cases"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
