"""
Typed hypothesis schema shared by the hypothesis and validator agents.

This module deliberately imports nothing from ada.llm and nothing from scipy —
it is the vocabulary both agents speak, not the machinery either one runs.
"""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ClaimType(str, Enum):
    """The kinds of claim ADA can actually put a statistical test behind."""

    GROUP_DIFFERENCE_NUMERIC = "group_difference_numeric"
    PROPORTION_DIFFERENCE = "proportion_difference"
    CORRELATION_NUMERIC = "correlation_numeric"
    ASSOCIATION_CATEGORICAL = "association_categorical"


class Direction(str, Enum):
    """
    What the hypothesis claims about direction.

    A significant test does not confirm a *directional* claim, so every spec
    carries one of these and the verdict logic checks the observed direction
    against it.
    """

    HIGHER_IN_FOCUS = "higher_in_focus"   # focus_level has the higher outcome
    LOWER_IN_FOCUS = "lower_in_focus"     # focus_level has the lower outcome
    POSITIVE = "positive"                 # var_a and var_b move together
    NEGATIVE = "negative"                 # var_a and var_b move oppositely
    ANY = "any"                           # non-directional: "these differ / are associated"


DIRECTIONAL_GROUP = (Direction.HIGHER_IN_FOCUS, Direction.LOWER_IN_FOCUS)
DIRECTIONAL_CORR = (Direction.POSITIVE, Direction.NEGATIVE)


class Verdict(str, Enum):
    """
    Every verdict is computed in code from a test result. SIGNIFICANT_BUT_TRIVIAL
    exists because "p < 0.05 on 50,000 rows" is the single most common way a tool
    like this overstates a finding.
    """

    CONFIRMED = "CONFIRMED"
    REJECTED = "REJECTED"
    SIGNIFICANT_BUT_TRIVIAL = "SIGNIFICANT_BUT_TRIVIAL"
    NOT_SUPPORTED = "NOT_SUPPORTED"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
    NOT_TESTABLE = "NOT_TESTABLE"


class NotTestableReason(str, Enum):
    """Machine-readable reasons, so a NOT_TESTABLE verdict is diagnosable."""

    UNKNOWN_COLUMN = "unknown_column"
    WRONG_DTYPE = "wrong_dtype"
    NOT_BINARY = "not_binary"
    TOO_MANY_LEVELS = "too_many_levels"
    UNKNOWN_FOCUS_LEVEL = "unknown_focus_level"
    MISSING_VARIABLE = "missing_variable"


class TestSpec(BaseModel):
    __test__ = False  # not a pytest test class, despite the name
    """The machine-readable half of a hypothesis: what to test, on what, which way."""

    model_config = ConfigDict(extra="forbid", use_enum_values=False)

    claim_type: ClaimType
    direction: Direction

    # group_difference_numeric / proportion_difference
    group_var: Optional[str] = None
    outcome_var: Optional[str] = None
    focus_level: Optional[str] = None

    # correlation_numeric / association_categorical
    var_a: Optional[str] = None
    var_b: Optional[str] = None

    @model_validator(mode="after")
    def _check_shape(self):
        group_like = (ClaimType.GROUP_DIFFERENCE_NUMERIC, ClaimType.PROPORTION_DIFFERENCE)
        pair_like = (ClaimType.CORRELATION_NUMERIC, ClaimType.ASSOCIATION_CATEGORICAL)

        if self.claim_type in group_like:
            if not self.group_var or not self.outcome_var:
                raise ValueError(
                    f"{self.claim_type.value} requires both group_var and outcome_var"
                )
            if self.direction not in DIRECTIONAL_GROUP + (Direction.ANY,):
                raise ValueError(
                    f"{self.claim_type.value} direction must be one of "
                    f"higher_in_focus / lower_in_focus / any"
                )
            # Without a focus level there is no way to check which group is higher,
            # so a directional claim that omits it is not a directional claim.
            if self.direction in DIRECTIONAL_GROUP and not self.focus_level:
                raise ValueError(
                    "focus_level is required when direction is higher_in_focus or lower_in_focus"
                )

        elif self.claim_type in pair_like:
            if not self.var_a or not self.var_b:
                raise ValueError(f"{self.claim_type.value} requires both var_a and var_b")
            if self.var_a == self.var_b:
                raise ValueError("var_a and var_b must be different columns")
            if self.claim_type == ClaimType.CORRELATION_NUMERIC:
                if self.direction not in DIRECTIONAL_CORR + (Direction.ANY,):
                    raise ValueError(
                        "correlation_numeric direction must be positive / negative / any"
                    )
            else:
                # Chi-square is inherently non-directional — there is no "higher".
                if self.direction != Direction.ANY:
                    raise ValueError("association_categorical direction must be 'any'")

        return self

    def columns(self) -> list[str]:
        """The columns this spec touches, for per-test complete-case filtering."""
        named = [self.group_var, self.outcome_var, self.var_a, self.var_b]
        return [c for c in named if c]


class Hypothesis(BaseModel):
    """
    A hypothesis. The first five fields are preserved verbatim from the pre-typed
    schema because app.py and the report node index them directly; test_spec is
    additive.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    hypothesis: str
    reasoning: str
    expected_evidence: str
    confidence: str = Field(pattern="^(high|medium|low)$")
    test_spec: TestSpec


class HypothesisSet(BaseModel):
    """Top-level response from the hypothesis agent."""

    model_config = ConfigDict(extra="forbid")

    dataset_type: str
    hypotheses: list[Hypothesis] = Field(min_length=1, max_length=8)
    most_important_hypothesis: str
    analysis_strategy: str


class Interpretation(BaseModel):
    """
    The LLM's only contribution to validation.

    There is deliberately no verdict field. extra="forbid" means a model that
    tries to emit one fails validation and gets retried rather than having the
    stray key silently dropped.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    insight: str = Field(max_length=600)
    caveat: Optional[str] = Field(default=None, max_length=300)


class InterpretationBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    interpretations: list[Interpretation]
