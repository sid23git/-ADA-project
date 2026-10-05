"""
Deterministic grading of an investigation against a scenario's ground truth.

Graded on the structured findings (each lists the columns it is about), not on
memo prose, so a score never depends on wording.
"""

from evals.scenarios import Scenario

LEAK_WORDS = ("leak", "after the outcome", "recorded after", "post-outcome")
NULL_WORDS = ("no meaningful", "no association", "not associated", "no evidence", "not supported",
              "no significant", "does not", "did not", "not predictive", "no better than", "barely")


def _claims_effect(f) -> bool:
    """
    A finding asserting that its columns matter for the target. Agents declare
    this in the structured `effect` field; wording is only a fallback for
    findings that leave it unset.
    """
    if f.effect != "not_applicable":
        return f.effect == "present" and f.kind != "data_quality"
    if f.kind not in ("statistical", "predictive"):
        return False
    return not any(w in f.claim.lower() for w in NULL_WORDS)


def _is_leak_finding(f) -> bool:
    return any(w in f.claim.lower() for w in LEAK_WORDS)


def grade(scenario: Scenario, result) -> dict:
    accepted = result.ledger.findings("accepted")
    # A finding that calls a column leakage is a warning about it, not a claim
    # that it drives the outcome.
    effect_findings = [f for f in accepted if _claims_effect(f) and not _is_leak_finding(f)]
    effect_cols = {c for f in effect_findings for c in f.columns}
    # Asymmetric on purpose. Recall: a driver is found if any accepted finding
    # says it matters. False discovery: a noise column only counts if it is
    # claimed to matter and never explicitly declared irrelevant — a full SHAP
    # ranking lists every feature, including ones another finding tested and
    # dismissed. ("absent" is not subtracted from recall, because agents also
    # use it for other null results, e.g. "contract is not confounded with tenure".)
    declared_absent = {c for f in accepted if f.effect == "absent" for c in f.columns}

    found = scenario.drivers & effect_cols
    false_hits = scenario.nulls & (effect_cols - declared_absent)
    if scenario.traps.get("target_leakage"):
        # Using the leaked column as a driver is a false discovery even if
        # another finding also flagged it.
        false_hits |= {scenario.traps["target_leakage"]} & {c for f in effect_findings for c in f.columns}

    traps = {}
    for trap, column in scenario.traps.items():
        if trap == "target_leakage":
            traps[trap] = any(column in f.columns and any(w in f.claim.lower() for w in LEAK_WORDS)
                              for f in accepted)
        elif trap == "sentinel_value":
            traps[trap] = any(column in f.columns and ("-999" in f.claim or "placeholder" in f.claim.lower()
                                                       or "sentinel" in f.claim.lower())
                              for f in accepted)
        elif trap == "duplicate_rows":
            traps[trap] = any("duplicate" in f.claim.lower() for f in accepted)

    all_findings = result.ledger.findings()
    summary = result.tracer.summary()
    failed_tasks = [t for t in result.task_log if t.get("error")]
    return {
        "scenario": scenario.name,
        # Guards against a vacuous pass: no work means no false discoveries.
        "completed": bool(result.task_log) and len(failed_tasks) < len(result.task_log)
                     and bool(result.memo),
        "failed_tasks": len(failed_tasks),
        "driver_recall": round(len(found) / len(scenario.drivers), 3) if scenario.drivers else None,
        "drivers_found": sorted(found),
        "drivers_missed": sorted(scenario.drivers - found),
        "false_discoveries": sorted(false_hits),
        "traps_caught": traps,
        "grounding_rate": result.lint.get("grounding_rate"),
        "invalid_citations": len(result.lint.get("invalid_citations", [])),
        "findings_total": len(all_findings),
        "findings_accepted": len(accepted),
        "rejected_by_verification": sum(1 for f in all_findings if f.status == "rejected"
                                        and f.critic_reason.startswith(("Failed verification",
                                                                        "Lost support"))),
        "rejected_by_critic": sum(1 for f in all_findings if f.status == "rejected"
                                  and not f.critic_reason.startswith(("Failed verification",
                                                                      "Lost support"))),
        "hypothesis_tests": len(result.ledger.hypothesis_tests()),
        "cost_usd": summary["total_cost_usd"],
        "llm_calls": summary["llm_calls"],
        "wall_seconds": summary["wall_seconds"],
    }


def passed(score: dict) -> bool:
    recall_ok = score["driver_recall"] is None or score["driver_recall"] >= 0.5
    return (score["completed"] and score["failed_tasks"] == 0
            and recall_ok and not score["false_discoveries"] and all(score["traps_caught"].values())
            and score["grounding_rate"] is not None and score["grounding_rate"] >= 0.9)
