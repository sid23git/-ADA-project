"""
A stand-in for the Anthropic client, so the whole pipeline runs offline.

It routes on a phrase unique to each agent's prompt and returns a canned
response shaped like the real one. Installed via utils.llm.set_client(), which
is the same single choke point every agent already calls through.
"""

import json
from types import SimpleNamespace

# Hypotheses written against data/sample.csv (Titanic). H5 names a column that
# does not exist, to exercise the NOT_TESTABLE path end to end.
HYPOTHESES = {
    "dataset_type": "Titanic passenger manifest with survival outcome",
    "hypotheses": [
        {
            "id": "H1",
            "hypothesis": "Female passengers survived at a higher rate than male passengers",
            "reasoning": "Women and children first",
            "expected_evidence": "Higher survival proportion for Sex=female",
            "confidence": "high",
            "test_spec": {"claim_type": "proportion_difference", "direction": "higher_in_focus",
                          "group_var": "Sex", "outcome_var": "Survived", "focus_level": "female"},
        },
        {
            "id": "H2",
            "hypothesis": "Survivors paid higher fares than non-survivors",
            "reasoning": "Wealthier passengers had cabins nearer the lifeboats",
            "expected_evidence": "Higher mean Fare where Survived=1",
            "confidence": "medium",
            "test_spec": {"claim_type": "group_difference_numeric", "direction": "higher_in_focus",
                          "group_var": "Survived", "outcome_var": "Fare", "focus_level": "1"},
        },
        {
            "id": "H3",
            "hypothesis": "Age is positively correlated with fare",
            "reasoning": "Older passengers could afford better tickets",
            "expected_evidence": "Positive correlation between Age and Fare",
            "confidence": "low",
            "test_spec": {"claim_type": "correlation_numeric", "direction": "positive",
                          "var_a": "Age", "var_b": "Fare"},
        },
        {
            "id": "H4",
            "hypothesis": "Passenger class is associated with port of embarkation",
            "reasoning": "Ports served different demographics",
            "expected_evidence": "Significant chi-square between Pclass and Embarked",
            "confidence": "medium",
            "test_spec": {"claim_type": "association_categorical", "direction": "any",
                          "var_a": "Pclass", "var_b": "Embarked"},
        },
        {
            "id": "H5",
            "hypothesis": "Higher income passengers survived more often",
            "reasoning": "Income is a proxy for class",
            "expected_evidence": "Higher Income among survivors",
            "confidence": "high",
            "test_spec": {"claim_type": "group_difference_numeric", "direction": "higher_in_focus",
                          "group_var": "Survived", "outcome_var": "Income", "focus_level": "1"},
        },
    ],
    "most_important_hypothesis": "H1",
    "analysis_strategy": "Focus on demographic and socioeconomic drivers of survival",
}

# Deliberately lists the target column under columns_to_drop: the cleaning
# agent must refuse to apply that part of the plan.
CLEANING = {
    "missing_value_strategies": {"Age": "median", "Embarked": "mode", "Cabin": "drop_column"},
    "columns_to_drop": ["PassengerId", "Name", "Ticket", "Survived"],
    "reasoning": "Impute Age with the median, Embarked with the mode; drop identifiers.",
}

# Includes a best_model the code did not choose: the ML agent must ignore it.
ML_INTERPRETATION = {
    "best_model": "Logistic Regression",
    "reasoning": "Highest cross-validated F1",
    "performance_summary": "Solid performance for a tabular baseline",
    "concerns": "Moderate dataset size",
    "recommendation": "Engineer family-size and title features",
}

SHAP_INTERPRETATION = {
    "plain_english_summary": "Sex and passenger class dominate the predictions.",
    "top_3_features": [
        {"feature": "Sex", "impact": "positive", "explanation": "Being female raises survival odds"},
        {"feature": "Pclass", "impact": "negative", "explanation": "Lower class lowers survival odds"},
        {"feature": "Fare", "impact": "positive", "explanation": "Higher fare tracks higher class"},
    ],
    "surprising_findings": "Age matters less than expected",
    "business_insight": "Socioeconomic status was a major survival factor",
}

INTERPRETATIONS = {
    "interpretations": [
        {"id": f"H{i}", "insight": f"Interpretation of H{i}.", "caveat": None}
        for i in range(1, 6)
    ]
}

# (phrase that appears only in that agent's prompt, response)
ROUTES = [
    ("form 5 testable hypotheses", json.dumps(HYPOTHESES)),
    ("performing Exploratory Data Analysis", "## EDA\nThe dataset has 891 rows."),
    ("deciding how to clean a dataset", json.dumps(CLEANING)),
    ("reviewing ML model results", json.dumps(ML_INTERPRETATION)),
    ("explaining model predictions", json.dumps(SHAP_INTERPRETATION)),
    ("interpreting the results of statistical tests", json.dumps(INTERPRETATIONS)),
    ("professional data analysis report", "# Final Report\nAll stages completed."),
]


class FakeAnthropic:
    """
    Duck-types the slice of anthropic.Anthropic that utils.llm uses:
    client.messages.create(...).content[0].text

    fail_times maps a route phrase to how many calls should raise before the
    route starts answering — used to exercise the graph's retry path.
    """

    def __init__(self, fail_times: dict | None = None):
        self.calls: list[str] = []
        self.fail_times = dict(fail_times or {})
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, model, max_tokens, system, messages):
        prompt = messages[0]["content"]
        for phrase, response in ROUTES:
            if phrase in prompt:
                self.calls.append(phrase)
                if self.fail_times.get(phrase, 0) > 0:
                    self.fail_times[phrase] -= 1
                    raise RuntimeError(f"simulated API failure for '{phrase}'")
                return SimpleNamespace(content=[SimpleNamespace(text=response)])
        raise AssertionError(f"FakeAnthropic has no route for prompt: {prompt[:200]!r}")
