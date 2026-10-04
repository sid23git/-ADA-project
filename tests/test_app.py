"""The Streamlit app: landing page, and a full offline investigation driven through the UI."""

from streamlit.testing.v1 import AppTest


def test_landing_page_renders():
    app = AppTest.from_file("app.py", default_timeout=60).run()
    assert not app.exception
    assert "AI data-science team" in app.title[0].value


def test_offline_investigation_through_the_ui(scripted):
    app = AppTest.from_file("app.py", default_timeout=180).run()
    app.sidebar.radio[1].set_value("Offline — scripted").run()
    app.sidebar.button[0].click().run()

    assert not app.exception
    labels = {m.label: m.value for m in app.metric}
    assert labels["Memo grounding"] == "100%"
    accepted, total = labels["Findings accepted"].split(" / ")
    assert int(accepted) > 0 and int(total) >= int(accepted)
    assert any("Bottom line" in m.value for m in app.markdown)
