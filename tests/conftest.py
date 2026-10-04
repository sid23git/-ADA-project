import os

import pandas as pd
import pytest

from tests.fakes import FakeAnthropic
from utils import llm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLE_CSV = os.path.join(ROOT, "data", "sample.csv")


@pytest.fixture
def fake_llm():
    """Route every LLM call in the process to the offline fake."""
    fake = FakeAnthropic()
    llm.set_client(fake)
    yield fake
    llm.set_client(None)


@pytest.fixture
def titanic() -> pd.DataFrame:
    return pd.read_csv(SAMPLE_CSV)


@pytest.fixture
def in_tmp_dir(tmp_path, monkeypatch):
    """Run with tmp_path as the working directory, so outputs/ lands there."""
    monkeypatch.chdir(tmp_path)
    return tmp_path
