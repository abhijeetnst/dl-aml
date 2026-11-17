"""Streamlit app: Corr2Cause checker (AML, DistilBERT, symbolic oracle), batch scoring, free-text causal graphs."""
import contextlib
import logging
import pathlib
import sys
import traceback

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import streamlit as st  # noqa: E402

from c2c import demo  # noqa: E402

log = logging.getLogger("corr2cause.app")
if not log.handlers:  # the script reruns on every interaction; attach the stderr handler once per process
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    log.addHandler(handler)
    log.setLevel(logging.INFO)
    log.propagate = False
REPO = "https://github.com/abhijeetnst/dl-aml"
REPORT = REPO + "/blob/main/docs/T043_Project1_Report.pdf"

HEADER = """# Corr2Cause: neuro-symbolic causal inference
Does a set of correlation and independence statements entail a causal claim? A pattern-feature logistic regression
(AML) and a fine-tuned DistilBERT read the text; a PC-algorithm oracle reasons over the Markov equivalence class.

| Test set, positive-class F1 | AML | DistilBERT | Symbolic oracle |
|---|---|---|---|
| original letters | 0.617 | 0.605 | **0.871** |
| variables renamed | 0.279 | 0.264 | **0.871** |
| DAG edge F1 | 0.703 | 0.610 | **0.942** |

Text models learn the dataset's alphabetical-order shortcut; the oracle does not depend on letter names."""

SIDEBAR = f"""### About
Corr2Cause asks whether correlation and independence statements entail a causal claim. This app scores a claim with a
classical baseline, a fine-tuned language model and a symbolic oracle side by side, and turns plain English causal
sentences into a graph you can intervene on.

### How it works
- **AML**: pattern features of the premise → logistic regression.
- **DistilBERT**: fine-tuned on (premise, hypothesis) pairs → P(valid).
- **PC oracle**: PC skeleton and v-structures → Markov equivalence class → valid only if every DAG agrees.
- **Bayesian network**: noisy-OR network on the extracted graph → ATE of do(X).

### Limits
- Corr2Cause tab: premises must follow the Corr2Cause template language.
- At most {demo.MAX_VARS} variables per premise.
- Free text: a pattern-based English extractor with a fixed list of causal verbs.
- ATEs assume noisy-OR: every edge raises its effect.

### Privacy
Inputs are processed in memory and not stored or logged.

[Code on GitHub]({REPO}) · [Project report (PDF)]({REPORT})

Team T043 · Section D"""


@contextlib.contextmanager
def guard(action: str):
    """Readable warning for bad input; anything else is logged (type and traceback only, never the input)."""
    try:
        yield
    except ValueError as e:
        st.warning(str(e))
    except Exception as e:
        log.error("%s failed: %s\n%s", action, type(e).__name__, "".join(traceback.format_tb(e.__traceback__)))
        st.error("Something went wrong… Please try again, or try a different input.")


@st.cache_resource(show_spinner=False)
def _load_models():
    return demo.load_models()


def models():
    with st.spinner("Loading models…"):
        return _load_models()


def fill_example():
    i = st.session_state.example
    if i is not None:
        st.session_state.premise, st.session_state.hypothesis = demo.EXAMPLES[i]


def rename():
    try:
        st.session_state.premise, st.session_state.hypothesis = demo.rename(st.session_state.premise,
                                                                             st.session_state.hypothesis)
    except ValueError as e:
        st.session_state.rename_error = str(e)
    except Exception as e:
        log.error("rename failed: %s\n%s", type(e).__name__, "".join(traceback.format_tb(e.__traceback__)))
        st.session_state.rename_error = "Something went wrong while renaming. Please try again."


def batch_tab():
    st.markdown(f"Upload a CSV with columns `premise` and `hypothesis`, plus an optional 0/1 `label` for precision, "
                f"recall and F1. Up to {demo.MAX_ROWS} rows and {demo.MAX_BYTES // 2**20} MB.")
    st.download_button("Download sample CSV", demo.sample_csv(), "corr2cause_sample.csv", "text/csv",
                       on_click="ignore")
    upload = st.file_uploader("CSV file", type="csv")
    if upload is None:
        return
    if st.session_state.get("batch_id") != upload.file_id:
        st.session_state.pop("batch", None)
        df = demo.read_csv(upload.getvalue())
        loaded = models()
        bar = st.progress(0.0, text="Scoring with DistilBERT…")
        st.session_state.batch = demo.score_batch(
            df, loaded, progress=lambda done, total: bar.progress(done / total, text=f"DistilBERT {done}/{total}"))
        st.session_state.batch_id = upload.file_id
        bar.empty()
    results, metrics = st.session_state.batch
    n_err = int((results.error != "").sum())
    st.caption(f"{len(results)} rows scored" + (f", {n_err} could not be parsed (see the error column)" if n_err else ""))
    if metrics is not None:
        st.dataframe(metrics, hide_index=True)
    st.dataframe(results, hide_index=True, column_config={"premise": st.column_config.TextColumn(width="small"),
                                                          "hypothesis": st.column_config.TextColumn(width="medium")})
    st.download_button("Download results CSV", results.to_csv(index=False).encode(), "corr2cause_results.csv",
                       "text/csv", type="primary", on_click="ignore")


st.set_page_config(page_title="Corr2Cause causal inference", page_icon="🔗", layout="wide")
with st.sidebar:
    st.markdown(SIDEBAR)
    st.caption(f"Version {demo.version()}")
st.markdown(HEADER)
tab_c2c, tab_batch, tab_text = st.tabs(["Corr2Cause", "Batch", "Free text"])

with tab_text:  # rendered first so a model load or batch run in the other tabs never holds it up
    text = st.text_area("Text", value=demo.DEFAULT_TEXT, height=120, max_chars=demo.MAX_TEXT)
    with guard("extract"):
        triples, dag, nodes, (x0, y0) = demo.extract(text)
        st.dataframe(triples, hide_index=True)
        st.image(dag, caption="Causal graph")
        col1, col2 = st.columns(2)
        x = col1.selectbox("Cause X", nodes, index=nodes.index(x0))
        y = col2.selectbox("Effect Y", nodes, index=nodes.index(y0))
        with guard("intervene"):
            st.markdown(demo.intervene(text, x, y))

with tab_c2c:
    st.selectbox("Test-set examples", range(len(demo.EXAMPLES)), index=None, key="example", on_change=fill_example,
                 format_func=lambda i: demo.EXAMPLES[i][1], placeholder="Pick an example to fill the inputs")
    st.text_area("Premise", key="premise", height=120, max_chars=demo.MAX_PREMISE)
    st.text_input("Hypothesis", key="hypothesis", max_chars=300)
    left, right = st.columns([1, 6])
    clicked = left.button("Check", type="primary")
    right.button("Rename variables", on_click=rename)
    if msg := st.session_state.pop("rename_error", None):
        st.warning(msg)
    if clicked:
        with guard("check"):
            demo.parse(st.session_state.premise, st.session_state.hypothesis)  # bad input never waits on the models
            table, summary, img = demo.check(st.session_state.premise, st.session_state.hypothesis, models())
            st.dataframe(table, hide_index=True)
            col1, col2 = st.columns(2)
            col1.markdown(summary)
            col2.image(img, caption="PC pattern")

with tab_batch:
    with guard("batch"):
        batch_tab()
