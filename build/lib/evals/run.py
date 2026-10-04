"""
Run the eval suite.

    python -m evals.run                     # every scenario, real Claude (costs money)
    python -m evals.run churn null_world    # selected scenarios
    python -m evals.run --offline           # scripted agents: checks the harness, not the AI

Writes evals/results/<timestamp>.json and prints a markdown table.
"""

import argparse
import json
import os
import sys
from datetime import datetime

from ada import llm
from ada.config import Budget, Settings
from ada.offline import ScriptedClaude
from ada.run import investigate
from evals.graders import grade, passed
from evals.scenarios import SCENARIOS, load

RESULTS_DIR = os.path.join("evals", "results")


def run(names: list[str], offline: bool = False, max_cost: float = 3.0) -> list[dict]:
    if offline:
        llm.set_client(ScriptedClaude())
    settings = Settings(budget=Budget(max_cost_usd=max_cost))
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    scores = []
    for name in names:
        scenario = load(name)
        print(f"\n=== {name}: {scenario.question}")
        result = investigate(scenario.df, scenario.question, target=scenario.target,
                             settings=settings,
                             run_dir=os.path.join("runs", f"eval_{stamp}_{name}"),
                             on_event=lambda e: print(f"  [{e['agent']}] {e['kind']}: {e['message'][:110]}"))
        score = grade(scenario, result) | {"mode": "offline" if offline else "live"}
        score["passed"] = passed(score)
        scores.append(score)
        print(json.dumps(score, indent=1))

    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"{stamp}{'_offline' if offline else ''}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(scores, f, indent=2)
    print("\n" + table(scores) + f"\n\nSaved to {path}")
    return scores


def table(scores: list[dict]) -> str:
    rows = ["| Scenario | Driver recall | False discoveries | Traps caught | Grounding | "
            "Rejected (verify / critic) | Cost | Pass |",
            "|---|---|---|---|---|---|---|---|"]
    for s in scores:
        recall = "n/a" if s["driver_recall"] is None else f"{s['driver_recall']:.0%}"
        traps = ", ".join(f"{k} {'✅' if v else '❌'}" for k, v in s["traps_caught"].items()) or "—"
        rows.append(f"| {s['scenario']} | {recall} | {', '.join(s['false_discoveries']) or 'none'} | "
                    f"{traps} | {s['grounding_rate']:.0%} | {s['rejected_by_verification']} / "
                    f"{s['rejected_by_critic']} | ${s['cost_usd']:.2f} | {'✅' if s['passed'] else '❌'} |")
    return "\n".join(rows)


def main(argv=None):
    # Windows consoles default to a legacy codepage that cannot print the table.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Run ADA's eval scenarios")
    parser.add_argument("scenarios", nargs="*", default=list(SCENARIOS))
    parser.add_argument("--offline", action="store_true", help="scripted agents, no API calls")
    parser.add_argument("--max-cost", type=float, default=3.0, help="USD budget per scenario")
    args = parser.parse_args(argv)
    run(args.scenarios, offline=args.offline, max_cost=args.max_cost)


if __name__ == "__main__":
    main()
