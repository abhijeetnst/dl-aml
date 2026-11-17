import pathlib

import pandas as pd
import pytest

from c2c import demo
from c2c.aml import train_aml
from c2c.oracle import entailed

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


def stub_models():
    """Real AML fitted on the six examples, fixed thresholds, no DistilBERT (tests pass a stub bert_fn)."""
    labels = [entailed(p, h) for p, h in demo.EXAMPLES]
    df = pd.DataFrame(demo.EXAMPLES, columns=["premise", "hypothesis"]).assign(label=labels)
    return train_aml(df), {"aml": 0.5, "bert": 0.5}, None


def stub_bert(premises, hypotheses, progress=None):
    if progress:
        progress(len(premises), len(premises))
    return [0.9] * len(premises)


BAD_ROW = b'"Not a premise.","A directly causes B.",1\n'


def test_batch_flags_unparseable_rows_and_scores_the_rest():
    df = demo.read_csv(demo.sample_csv() + BAD_ROW)
    calls = []
    out, metrics = demo.score_batch(df, stub_models(), bert_fn=stub_bert, progress=lambda *a: calls.append(a))
    assert calls == [(3, 3)]
    assert list(out.error != "") == [False, False, False, True] and "Corr2Cause template" in out.error[3]
    assert out.oracle_pred[:3].tolist() == demo.SAMPLE_LABELS and out.loc[3, ["aml_pred", "bert_pred"]].isna().all()
    m = metrics.set_index("system")
    assert (m.loc["Symbolic oracle", "f1"], m.loc["DistilBERT", "precision"], m.loc["DistilBERT", "recall"]) == (1, 0.667, 1)
    assert (m.rows == 3).all()


def test_batch_without_labels_or_valid_rows():
    out, metrics = demo.score_batch(demo.read_csv(b"premise,hypothesis\nx,y\n"), stub_models(), bert_fn=stub_bert)
    assert metrics is None and out.error[0] and out.bert_pred.isna().all()


@pytest.mark.parametrize("data, message", [
    (b"premise,label\nx,1\n", "Missing column"),
    (b"premise,hypothesis\n" + b"x,y\n" * (demo.MAX_ROWS + 1), "1 to 500 rows"),
    (b"premise,hypothesis,label\nx,y,maybe\n", "only 0 or 1"),
    (b"x" * (demo.MAX_BYTES + 1), "larger than 2 MB"),
])
def test_read_csv_rejects_bad_files(data, message):
    with pytest.raises(ValueError, match=message):
        demo.read_csv(data)


def test_input_limits(monkeypatch):
    with pytest.raises(ValueError, match="longer than 6,000"):
        demo.parse("x" * (demo.MAX_PREMISE + 1), "A directly causes B.")
    with pytest.raises(ValueError, match="longer than 5,000"):
        demo.extract("Rain causes floods. " * 300)
    many = pd.DataFrame({"sentence_id": range(51)})
    monkeypatch.setattr(demo, "extract_triples", lambda text: many)
    with pytest.raises(ValueError, match="at most 50"):
        demo.intervene("short text", "a", "b")


@pytest.mark.skipif(not HAVE_MODELS, reason="needs the local DistilBERT model and exported artifacts")
def test_batched_bert_matches_single_pair():
    models = demo.load_models()
    out, _ = demo.score_batch(demo.read_csv(demo.sample_csv()), models)
    single = [demo.bert_probs([p], [h], models[2])[0] for p, h in demo.EXAMPLES[:3]]
    assert out.bert_p.tolist() == pytest.approx(single, abs=1e-3)


def test_streamlit_app_runs_without_loading_models(monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setattr(demo, "load_models", lambda: pytest.fail("models loaded before Check was pressed"))
    at = AppTest.from_file(str(ROOT / "demo/streamlit_app.py")).run(timeout=120)
    assert not at.exception
    assert any("**ATE**" in m.value for m in at.markdown)
    assert [s.label for s in at.selectbox] == ["Test-set examples", "Cause X", "Effect Y"]
    assert [t.label for t in at.tabs] == ["Corr2Cause", "Batch", "Free text"]
    assert any("Version " in c.value for c in at.sidebar.caption)


def test_streamlit_unexpected_error_is_logged_without_user_input(monkeypatch, caplog):
    import logging

    from streamlit.testing.v1 import AppTest

    def boom(text):
        raise RuntimeError(text)  # the message is the user's text

    monkeypatch.setattr(demo, "extract", boom)
    app_log = logging.getLogger("corr2cause.app")
    app_log.addHandler(caplog.handler)
    try:
        at = AppTest.from_file(str(ROOT / "demo/streamlit_app.py")).run(timeout=120)
    finally:
        app_log.removeHandler(caplog.handler)
    assert not at.exception and at.error[0].value.startswith("Something went wrong")
    assert "extract failed: RuntimeError" in caplog.text and "in boom" in caplog.text  # type and traceback frames
    assert "Heavy storms" not in caplog.text


def test_streamlit_batch_tab_scores_upload(monkeypatch):
    import streamlit as st
    from streamlit.testing.v1 import AppTest

    st.cache_resource.clear()
    monkeypatch.setattr(demo, "load_models", stub_models)
    monkeypatch.setattr(demo, "bert_probs", lambda p, h, bert, progress=None: stub_bert(p, h, progress))
    at = AppTest.from_file(str(ROOT / "demo/streamlit_app.py")).run(timeout=120)
    at.file_uploader[0].upload("rows.csv", demo.sample_csv() + BAD_ROW, "text/csv").run(timeout=120)
    st.cache_resource.clear()
    assert not at.exception and not at.error
    metrics, results = (d.value for d in at.tabs[1].dataframe)
    assert len(results) == 4 and results.error.astype(bool).sum() == 1
    assert metrics.system.tolist() == list(demo.SYSTEMS.values())
