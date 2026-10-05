"""
Command-line investigations.

    python -m ada data/sample.csv "What determined who survived?" --target Survived
    python -m ada data.csv "What drives churn?" --offline        # scripted agents, no API key
"""

import argparse
import sys

from ada import llm
from ada.config import Budget, Settings
from ada.offline import ScriptedClaude
from ada.run import investigate

ICONS = {"plan": "🧭", "start": "▶", "tool_call": "🔧", "evidence": "📎", "submitted": "📤",
         "accepted": "✅", "rejected": "❌", "demoted": "⬇", "error": "⚠", "done": "📝"}


def main(argv=None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(prog="python -m ada", description="ADA — an AI data-science team")
    parser.add_argument("csv")
    parser.add_argument("question")
    parser.add_argument("--target", help="outcome column (the lead picks one if omitted)")
    parser.add_argument("--max-cost", type=float, default=3.0, help="USD budget (default 3.0)")
    parser.add_argument("--offline", action="store_true", help="scripted agents, no API calls")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    if args.offline:
        llm.set_client(ScriptedClaude())

    def show(e):
        if not args.quiet and e["kind"] != "thought":
            print(f"{e['t']:7.1f}s  {ICONS.get(e['kind'], '·')} [{e['agent']}] {e['message'][:140]}")

    result = investigate(args.csv, args.question, target=args.target,
                         settings=Settings(budget=Budget(max_cost_usd=args.max_cost)), on_event=show)
    summary = result.tracer.summary()
    print("\n" + result.memo)
    print(f"\n— {len(result.ledger.findings('accepted'))}/{len(result.ledger.findings())} findings accepted · "
          f"grounding {result.lint['grounding_rate']:.0%} · {summary['llm_calls']} LLM calls · "
          f"${summary['total_cost_usd']:.2f}{' (estimated, offline)' if args.offline else ''} · "
          f"{summary['wall_seconds']}s · artifacts in {result.run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
