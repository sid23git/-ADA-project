"""
Deterministic verification: the checks that do not trust any model.

`check_finding` runs before the critic agent sees a finding, and a finding that
fails a blocking check cannot be accepted whatever the critic says.
`lint_memo` runs on the final report and measures how much of it is grounded
in computed evidence.
"""

import re
from collections.abc import Iterable
from typing import Optional

from ada.ledger import VERDICT_CONFIRMED, EvidenceLedger, Finding, numbers_in

# Numbers like 12, -0.31, 1,204, 3.5e-4, 42% — but not the digits inside an
# identifier such as H1, E12, F3 or a column name like var2.
_NUMBER = re.compile(r"(?<![A-Za-z_\d.])-?\d[\d,]*(?:\.\d+)?(?:[eE]-?\d+)?%?(?![A-Za-z_\d])")
_CITATION = re.compile(r"\[((?:[EF]\d+)(?:\s*,\s*[EF]\d+)*)\]")
_CAUSAL = re.compile(r"\b(causes?|caused by|causal(?:ly)?|leads? to|results? in|because of)\b",
                     re.IGNORECASE)

# Small whole numbers ("3 segments", "top 5 features") and years are counts or
# labels, not measurements — requiring evidence for them is noise.
TRIVIAL_INT_MAX = 10


def extract_numbers(text: str) -> list[float]:
    """Numeric values in text. A percentage yields both 42 and 0.42."""
    out = []
    for match in _NUMBER.finditer(text):
        token = match.group(0).replace(",", "")
        is_pct = token.endswith("%")
        try:
            value = float(token.rstrip("%"))
        except ValueError:
            continue
        out.append(value)
        if is_pct:
            out.append(value / 100)
    return out


# Numbers that label rather than measure. An analyst's bin edges ("13-36
# months", "tenure <=12"), a confidence level ("95% CI") or a cut-off ("top
# 10%") are choices, not results, so they need no evidence. Only whole-number
# bounds count as bin edges; "0.8-0.9" is still checked.
_LABELS = [
    re.compile(r"\b\d{2}(?:\.\d+)?\s*%\s*(?:CI\b|C\.I\.|confidence)", re.IGNORECASE),
    re.compile(r"\b(?:top|bottom|first|last)\s+\d+(?:\.\d+)?\s*%?", re.IGNORECASE),
    re.compile(r"(?<![\d.])\d+\s*(?:[-–—]|to)\s*\d+(?![\d.%])"),
    re.compile(r"(?:[≤≥<>]=?|\bover|\bunder|\babove|\bbelow)\s*\d+(?![\d.%])", re.IGNORECASE),
]


def _strip_labels(text: str) -> str:
    for pattern in _LABELS:
        text = pattern.sub(" ", text)
    return text


def _claim_numbers(text: str) -> list[tuple[float, int, bool]]:
    """(value as written, decimals shown, is_percent) for each number a claim asserts."""
    found = []
    for match in _NUMBER.finditer(_strip_labels(text)):
        token = match.group(0).replace(",", "")
        is_pct = token.endswith("%")
        core = token.rstrip("%")
        try:
            value = float(core)
        except ValueError:
            continue
        decimals = len(core.split(".")[1]) if "." in core and "e" not in core.lower() else 0
        if decimals == 0 and not is_pct and (abs(value) <= TRIVIAL_INT_MAX or 1900 <= value <= 2100):
            continue
        found.append((value, decimals, is_pct))
    return found


def _matches(claimed: float, decimals: int, observed: float) -> bool:
    """Does `claimed` faithfully report `observed`, allowing for rounding?"""
    if round(observed, decimals) == round(claimed, decimals):
        return True
    tolerance = max(0.5 * 10 ** -decimals, 0.01 * abs(observed))
    return abs(claimed - observed) <= tolerance


def ungrounded_numbers(text: str, pool: Iterable[float]) -> list[float]:
    """
    Numbers in `text` that no value in `pool` supports. A percentage may be
    backed either by the same figure (19.87% <- 19.87) or by a proportion
    (19.87% <- 0.1987), since tools report both forms.
    """
    pool = list(pool)
    missing = []
    for value, decimals, is_pct in _claim_numbers(text):
        readings = [(value, decimals)]
        if is_pct:
            readings.append((value / 100, decimals + 2))
        if not any(_matches(v, d, obs) for v, d in readings for obs in pool):
            missing.append(f"{value:g}%" if is_pct else value)
    return missing


def check_finding(finding: Finding, ledger: EvidenceLedger,
                  columns: Optional[set] = None) -> list[dict]:
    """
    Blocking checks (passed=False rejects the finding):
      citations_resolve, numbers_grounded, statistical_support,
      predictive_support, columns_exist
    Warning checks (never block on their own): causal_language
    """
    checks = []

    def add(name, passed, detail, blocking=True):
        checks.append({"check": name, "passed": bool(passed), "blocking": blocking, "detail": detail})

    cited = [ledger.evidence(eid) for eid in finding.evidence_ids]
    missing = [eid for eid, ev in zip(finding.evidence_ids, cited) if ev is None]
    add("citations_resolve", not missing,
        f"unknown evidence ids: {missing}" if missing else f"{len(cited)} citation(s) resolve")
    cited = [ev for ev in cited if ev is not None]

    pool = [n for ev in cited for n in numbers_in(ev.result)]
    pool += [n for ev in cited for n in numbers_in(ev.args)]
    bad = ungrounded_numbers(finding.claim, pool)
    add("numbers_grounded", not bad,
        f"numbers not found in cited evidence: {bad}" if bad else "every number appears in cited evidence")

    if finding.kind == "statistical":
        tests = [ev for ev in cited if ev.tool == "run_hypothesis_test"]
        verdicts = {ev.id: ev.result.get("verdict") for ev in tests}
        ok = any(v == VERDICT_CONFIRMED for v in verdicts.values())
        if not tests:
            detail = "statistical claim cites no hypothesis test"
        elif ok:
            detail = f"supported by confirmed test(s) {[k for k, v in verdicts.items() if v == VERDICT_CONFIRMED]}"
        else:
            family = tests[0].result.get("family_size")
            detail = (f"cited test verdicts {verdicts} after correction across {family} tests — "
                      f"none CONFIRMED")
        add("statistical_support", ok, detail)

    if finding.kind == "predictive":
        ok = any(ev.tool in ("train_models", "explain_model") for ev in cited)
        add("predictive_support", ok,
            "cites model evidence" if ok else "predictive claim cites no train_models/explain_model result")

    if columns is not None and finding.columns:
        unknown = [c for c in finding.columns if c not in columns]
        add("columns_exist", not unknown,
            f"unknown columns: {unknown}" if unknown else "all referenced columns exist")

    causal = _CAUSAL.findall(finding.claim)
    add("causal_language", not causal,
        f"causal wording {sorted(set(w.lower() for w in causal))} on observational evidence"
        if causal else "no causal wording", blocking=False)

    return checks


def blocking_failures(checks: list[dict]) -> list[dict]:
    return [c for c in checks if c["blocking"] and not c["passed"]]


def lint_memo(memo: str, ledger: EvidenceLedger) -> dict:
    """
    Grounding report for the final memo. Every sentence that states a number
    must cite an accepted finding or evidence item that contains that number.
    """
    accepted = {f.id: f for f in ledger.findings("accepted")}
    # Split into sentences, but never between a sentence and the citation that
    # follows its full stop: "... 0.543. [F5]" is one cited sentence.
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+(?!\[)|\n+", memo) if s.strip()]

    invalid, ungrounded, numeric = [], [], 0
    for sentence in sentences:
        if sentence.startswith("#") or sentence.startswith("|"):
            continue
        ids = [i.strip() for group in _CITATION.findall(sentence) for i in group.split(",")]
        for cid in ids:
            if cid.startswith("F") and cid not in accepted:
                invalid.append(cid)
            if cid.startswith("E") and ledger.evidence(cid) is None:
                invalid.append(cid)
        if not _claim_numbers(_CITATION.sub("", sentence)):
            continue
        numeric += 1
        pool: list[float] = []
        for cid in ids:
            if cid in accepted:
                f = accepted[cid]
                pool += extract_numbers(f.claim)
                for eid in f.evidence_ids:
                    ev = ledger.evidence(eid)
                    if ev:
                        pool += numbers_in(ev.result)
            elif (ev := ledger.evidence(cid)) is not None:
                pool += numbers_in(ev.result)
        if not ids or ungrounded_numbers(_CITATION.sub("", sentence), pool):
            ungrounded.append(sentence)

    grounded = numeric - len(ungrounded)
    return {
        "numeric_sentences": numeric,
        "grounded_sentences": grounded,
        "grounding_rate": round(grounded / numeric, 3) if numeric else 1.0,
        "ungrounded": ungrounded,
        "invalid_citations": sorted(set(invalid)),
    }
