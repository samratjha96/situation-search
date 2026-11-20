# /// script
# requires-python = ">=3.10"
# dependencies = ["fastapi", "httpx", "uvicorn"]
# ///
"""Run contrasting natural-language questions through the local search API."""
from pathlib import Path
import sys

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.server import app


def main():
    with TestClient(app) as client:
        def ask(query, status="verified"):
            r = client.get("/api/search", params={"q": query, "clip": "clip2"})
            r.raise_for_status()
            body = r.json()
            assert body["status"] == status, (query, body)
            if status == "unverified":
                assert body["message"] and not body["no_match"], query
            return body["results"]

        # The first corner ends in a save. The second includes goalkeeper contact
        # but ends in a goal. Both saves belong to United.
        rows = ask("Arsenal corners that ended in a save")
        assert [r["outcome"] for r in rows] == ["save"]
        assert not ask("corner followed by an Arsenal save")
        assert {r["outcome"] for r in ask("corner followed by a Manchester United save")} == {"save", "goal"}

        header = ask("goalkeeper saves a header")
        assert len(header) == 1
        assert [e["action"] for e in header[0]["evidence"]] == ["header", "save"]
        assert ask("saved headers") == ask("header saved by Lammens") == header
        assert ask("Manchester United saves an Arsenal header") == header
        assert not ask("Arsenal saves a Manchester United header")
        assert not ask("corner leading to a saved header")
        switch = ask("shot after a switch of play")
        assert len(switch) == 1 and switch == ask("switch of play followed by a shot")
        assert [e["action"] for e in switch[0]["evidence"]] == ["pass", "clearance", "receive", "shot"]

        # The index attributes the finish to Merino; taking the corner is a different role.
        assert [r["scorer"] for r in ask("Merino scores after an Arsenal corner")] == ["Merino"]
        assert not ask("goal after a corner taken by Merino")
        ask("shot by Merino before a corner", "unverified")
        ask("pass before a regain then a shot", "unverified")

        # Different players can perform consecutive actions. Reversing the
        # wording with "after" keeps the same chronology and names.
        forward = ask("pass by Mbeumo then Fernandes shot")
        reverse = ask("shot by Fernandes after a Mbeumo pass")
        assert forward == reverse and any(135 < r["t_seconds"] < 138 for r in forward)
        assert ask("pass by Fernandes then Mbeumo shot") != forward
        shots = ask("shot by Saka")
        assert shots and all(r["players"] == ["Saka"] for r in shots)

        # A passage's coarse outcome can be "shot" while the shot itself went wide.
        assert any(135 < r["t_seconds"] < 137 for r in ask("regain ending in a shot off target"))
        assert not any(135 < r["t_seconds"] < 137 for r in ask("regain ending in a shot on target"))

        ask("corner that did not lead to no goal", "unverified")
        ask("corner that ended in a goal without a save", "unverified")
        ask("Liverpool corners", "unverified")
        assert not ask("Manchester City corners")
    print("Passed query regressions for roles, direction, terminal outcomes, exclusions and teams.")


if __name__ == "__main__":
    main()
