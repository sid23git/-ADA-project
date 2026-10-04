"""
Run the ADA pipeline headless.

    python main.py                                   # Titanic sample, target Survived
    python main.py path/to/data.csv --target Price
"""

import argparse
import sys

from ada_graph import run_ada


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="ADA — Autonomous Data Analysis Agent")
    parser.add_argument("csv", nargs="?", default="data/sample.csv",
                        help="CSV file to analyse (default: data/sample.csv)")
    parser.add_argument("--target", default=None,
                        help="target column (default: Survived for the sample, "
                             "otherwise the last column)")
    args = parser.parse_args(argv)

    target = args.target
    if target is None and args.csv == "data/sample.csv":
        target = "Survived"

    results = run_ada(filepath=args.csv, target_col=target)

    if results.get("error"):
        print(f"\nPipeline stopped: {results['error']}")
        return 1

    print("\nFinal report preview:")
    print((results.get("final_report") or "")[:500])
    print(f"\nAudit trail entries: {len(results['audit_trail'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
