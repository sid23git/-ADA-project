"""
The evidence ledger: the shared memory every agent reads and writes.

Every tool result is recorded as an evidence item (E1, E2, ...) before the
agent sees it, and every claim an agent makes is a Finding that must cite
evidence IDs. Nothing an agent says reaches the final memo unless it can be
traced back to something that was actually computed.

The ledger is also where multiple-comparison correction happens. Agents run
statistical tests independently and in parallel, so a test that looked
significant to the agent that ran it may not survive correction across every
test the whole team ran. `correct_family()` recomputes that after each round.
"""

import math
import threading
from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from ada.stats.hypothesis_schema import TestSpec, Verdict
from ada.stats.stat_tests import CORRECTION_METHODS, decide_verdict, render_evidence

FindingKind = Literal["statistical", "predictive", "descriptive", "data_quality"]
# Does the finding say its columns matter for the target, that they do not, or neither?
Effect = Literal["present", "absent", "not_applicable"]


class Evidence(BaseModel):
    id: str
    agent: str
    tool: str
    args: dict
    result: dict
    created: str


class Finding(BaseModel):
    """A claim proposed by a specialist. It reaches the memo only if accepted."""

    id: str = ""
    agent: str = ""
    claim: str = Field(description="One specific, quantified sentence")
    kind: FindingKind
    columns: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(min_length=1)
    confidence: Literal["high", "medium", "low"] = "medium"
    implication: str = ""
    effect: Effect = "not_applicable"

    # Filled in by review.
    status: Literal["pending", "accepted", "rejected"] = "pending"
    checks: list[dict] = Field(default_factory=list)
    critic_reason: str = ""
    round: int = 0


class EvidenceLedger:
    def __init__(self, alpha: float = 0.05, correction: str = "benjamini-hochberg"):
        self.alpha = alpha
        self.correction = correction
        self._evidence: dict[str, Evidence] = {}
        self._findings: dict[str, Finding] = {}
        self._lock = threading.RLock()

    # ── evidence ────────────────────────────────────────────────────────────

    def add_evidence(self, agent: str, tool: str, args: dict, result: dict) -> str:
        with self._lock:
            eid = f"E{len(self._evidence) + 1}"
            self._evidence[eid] = Evidence(
                id=eid, agent=agent, tool=tool, args=args, result=result,
                created=datetime.now().isoformat(timespec="seconds"),
            )
            return eid

    def evidence(self, eid: str) -> Optional[Evidence]:
        with self._lock:
            return self._evidence.get(eid)

    def all_evidence(self) -> list[Evidence]:
        with self._lock:
            return list(self._evidence.values())

    # ── findings ────────────────────────────────────────────────────────────

    def add_finding(self, finding: Finding) -> Finding:
        with self._lock:
            finding.id = f"F{len(self._findings) + 1}"
            self._findings[finding.id] = finding
            return finding

    def findings(self, status: Optional[str] = None) -> list[Finding]:
        with self._lock:
            items = list(self._findings.values())
        return [f for f in items if status is None or f.status == status]

    def finding(self, fid: str) -> Optional[Finding]:
        with self._lock:
            return self._findings.get(fid)

    # ── multiple-comparison correction across the whole team ────────────────

    def hypothesis_tests(self) -> list[Evidence]:
        return [e for e in self.all_evidence()
                if e.tool == "run_hypothesis_test" and e.result.get("p_value") is not None
                and e.result.get("status") == "tested"]

    def correct_family(self) -> int:
        """
        Re-adjust every p-value against every test run so far, then recompute
        each verdict. Returns the family size.
        """
        with self._lock:
            tests = self.hypothesis_tests()
            if not tests:
                return 0
            adjust = CORRECTION_METHODS[self.correction]
            adjusted = adjust([e.result["p_value"] for e in tests])
            for ev, q in zip(tests, adjusted):
                record = dict(ev.result["record"])
                record["p_adjusted"] = q
                record["correction_method"] = self.correction
                record["family_size"] = len(tests)
                spec = TestSpec.model_validate(ev.args)
                outcome = decide_verdict(spec, record, alpha=self.alpha)
                ev.result.update({
                    "p_adjusted": q,
                    "family_size": len(tests),
                    "verdict": outcome["verdict"].value,
                    "evidence": render_evidence(record | outcome, outcome["verdict"]),
                })
            return len(tests)

    def test_verdict(self, eid: str) -> Optional[str]:
        ev = self.evidence(eid)
        return ev.result.get("verdict") if ev and ev.tool == "run_hypothesis_test" else None

    # ── serialisation ───────────────────────────────────────────────────────

    @classmethod
    def from_dict(cls, data: dict) -> "EvidenceLedger":
        ledger = cls()
        for ev in data.get("evidence", []):
            ledger._evidence[ev["id"]] = Evidence.model_validate(ev)
        for f in data.get("findings", []):
            ledger._findings[f["id"]] = Finding.model_validate(f)
        return ledger

    def to_dict(self) -> dict:
        return {
            "evidence": [e.model_dump() for e in self.all_evidence()],
            "findings": [f.model_dump() for f in self.findings()],
        }


def numbers_in(value: Any) -> list[float]:
    """Every finite number anywhere in a nested result, for grounding checks."""
    out: list[float] = []
    if isinstance(value, bool):
        return out
    if isinstance(value, (int, float)):
        if math.isfinite(value):
            out.append(float(value))
    elif isinstance(value, dict):
        for v in value.values():
            out.extend(numbers_in(v))
    elif isinstance(value, (list, tuple)):
        for v in value:
            out.extend(numbers_in(v))
    elif isinstance(value, str):
        from ada.verify import extract_numbers
        out.extend(extract_numbers(value))
    return out


VERDICT_CONFIRMED = Verdict.CONFIRMED.value
