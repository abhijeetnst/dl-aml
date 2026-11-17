import pathlib

import pytest

import app

HAVE_MODELS = pathlib.Path("models/distilbert-c2c/config.json").exists() and pathlib.Path("artifacts/aml.joblib").exists()


@pytest.mark.skipif(not HAVE_MODELS, reason="needs the local DistilBERT model and exported artifacts")
def test_check_reports_oracle_verdict_and_class_size():
    premise, hypothesis = app.EXAMPLES[0]  # A -> C <- B: the collider is identified, one DAG in the class
    table, summary, img = app.check(premise, hypothesis)
    oracle = table.set_index("system").loc["Symbolic oracle"]
    assert (oracle["P(valid)"], oracle["verdict"]) == (1.0, "valid")
    assert list(table.system) == ["AML (pattern LR)", "DistilBERT", "Symbolic oracle"]
    assert "(collider, A, B)" in summary and "1 DAG " in summary
    assert img is not None


def test_unparseable_input_gives_message_not_exception():
    table, summary, img = app.check("Not a premise.", "A directly causes B.")
    assert table.empty and img is None and "Could not parse" in summary


def test_free_text_intervention():
    _, img, cause, effect = app.extract(app.DEFAULT_TEXT)
    assert img is not None and (cause.value, effect.value) == ("heavy storm", "flood")
    effect_size = float(app.intervene(app.DEFAULT_TEXT, "heavy storm", "protest").split("**")[3])
    assert 0 < effect_size <= 1
    assert float(app.intervene(app.DEFAULT_TEXT, "protest", "heavy storm").split("**")[3]) == 0
