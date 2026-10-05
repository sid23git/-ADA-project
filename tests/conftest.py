import os

import pandas as pd
import pytest

from ada import llm
from ada.offline import ScriptedClaude

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLE_CSV = os.path.join(ROOT, "data", "sample.csv")


@pytest.fixture
def scripted():
    """Route every model call to the offline scripted client."""
    client = ScriptedClaude()
    llm.set_client(client)
    yield client
    llm.set_client(None)


@pytest.fixture
def titanic() -> pd.DataFrame:
    return pd.read_csv(SAMPLE_CSV)
