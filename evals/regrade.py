"""
Re-score saved investigations without calling any model.

    python -m evals.regrade runs/eval_20261004_224902_churn [more run dirs...]

Grading is deterministic and every run saves its full ledger, so when a grader
improves, past live runs can be re-scored for free.
"""

import json
import os
import sys
from types import SimpleNamespace

from ada.ledger import EvidenceLedger
from evals.graders import grade, passed
from evals.run import table
from evals.scenarios import SCENARIOS, load


def load_run(run_dir: str) -> SimpleNamespace:
    with open(os.path.join(run_dir, "investigation.json"), encoding="utf-8") as f:
        data = json.load(f)
    summary = data["trace"]["summary"]
    return SimpleNamespace(ledger=EvidenceLedger.from_dict(data), lint=data["lint"], memo=data["memo"],
                           task_log=data["task_log"], tracer=SimpleNamespace(summary=lambda: summary))


def regrade(run_dir: str) -> dict:
    name = next(s for s in SCENARIOS if run_dir.rstrip("/\\").endswith(s))
    score = grade(load(name), load_run(run_dir)) | {"mode": "live (re-graded)", "run_dir": run_dir}
    score["passed"] = passed(score)
    return score


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    scores = [regrade(d) for d in (argv or sys.argv[1:])]
    print(table(scores))
    return scores


if __name__ == "__main__":
    main()
