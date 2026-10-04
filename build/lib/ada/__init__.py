"""
ADA — an auditable multi-agent AI data-science team.

    from ada import investigate
    result = investigate("data.csv", "What drives churn?", target="churned")
    print(result.memo)
"""

from ada.config import Budget, Settings
from ada.run import InvestigationResult, investigate

__all__ = ["investigate", "InvestigationResult", "Settings", "Budget"]
__version__ = "4.0.0"
