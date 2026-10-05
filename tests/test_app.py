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


def test_public_demo_never_uses_the_servers_api_key(monkeypatch):
    import ada.config
    monkeypatch.setattr(ada.config, "PUBLIC_DEMO", True)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-server-key")

    app = AppTest.from_file("app.py", default_timeout=60).run()
    agents = app.sidebar.radio[1]
    assert agents.value == "Offline — scripted"          # free mode by default
    agents.set_value("Live — Claude").run()
    assert app.sidebar.button[0].disabled                 # server key is not an option
    app.sidebar.text_input[0].set_value("sk-ant-visitor-key").run()
    assert not app.sidebar.button[0].disabled
    assert app.sidebar.slider[0].max == ada.config.PUBLIC_MAX_COST_USD
