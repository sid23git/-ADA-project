"""Smoke test: the Streamlit app renders its landing page without errors."""

from streamlit.testing.v1 import AppTest


def test_landing_page_renders(fake_llm):
    app = AppTest.from_file("app.py", default_timeout=30).run()

    assert not app.exception
    assert "Autonomous Data Analysis Agent" in app.title[0].value
    # The run button stays disabled until a CSV is uploaded.
    assert app.sidebar.button[0].disabled
