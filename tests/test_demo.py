import pathlib

import pytest

from c2c import demo

ROOT = pathlib.Path(__file__).resolve().parents[1]
HAVE_MODELS = (ROOT / "models/distilbert-c2c/config.json").exists() and (ROOT / "artifacts/aml.joblib").exists()


@pytest.mark.skipif(not HAVE_MODELS, reason="needs the local DistilBERT model and exported artifacts")
def test_check_reports_oracle_verdict_and_class_size():
    premise, hypothesis = demo.EXAMPLES[0]  # A -> C <- B: the collider is identified, one DAG in the class
    table, summary, img = demo.check(premise, hypothesis, demo.load_models())
    oracle = table.set_index("system").loc["Symbolic oracle"]
    assert (oracle["P(valid)"], oracle["verdict"]) == (1.0, "valid")
    assert list(table.system) == ["AML (pattern LR)", "DistilBERT", "Symbolic oracle"]
    assert "(collider, A, B)" in summary and "1 DAG " in summary
    assert img is not None


def test_unparseable_input_raises_readable_error():
    with pytest.raises(ValueError, match="Corr2Cause template"):
        demo.check("Not a premise.", "A directly causes B.", models=None)


def test_free_text_intervention():
    _, img, nodes, (cause, effect) = demo.extract(demo.DEFAULT_TEXT)
    assert img is not None and (cause, effect) == ("heavy storm", "flood") and "protest" in nodes
    effect_size = float(demo.intervene(demo.DEFAULT_TEXT, "heavy storm", "protest").split("**")[3])
    assert 0 < effect_size <= 1
    assert float(demo.intervene(demo.DEFAULT_TEXT, "protest", "heavy storm").split("**")[3]) == 0


def test_streamlit_app_runs_without_loading_models(monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setattr(demo, "load_models", lambda: pytest.fail("models loaded before Check was pressed"))
    at = AppTest.from_file(str(ROOT / "demo/streamlit_app.py")).run(timeout=120)
    assert not at.exception
    assert any("**ATE**" in m.value for m in at.markdown)
    assert [s.label for s in at.selectbox] == ["Test-set examples", "Cause X", "Effect Y"]
