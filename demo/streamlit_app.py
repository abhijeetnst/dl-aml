"""Streamlit demo: Corr2Cause checker (AML, DistilBERT, symbolic oracle) and free-text causal extraction + do-queries."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import streamlit as st  # noqa: E402

from c2c import demo  # noqa: E402

HEADER = """# Corr2Cause: neuro-symbolic causal inference
Does a set of correlation and independence statements entail a causal claim? A pattern-feature logistic regression
(AML) and a fine-tuned DistilBERT read the text; a PC-algorithm oracle reasons over the Markov equivalence class.

| Test set, positive-class F1 | AML | DistilBERT | Symbolic oracle |
|---|---|---|---|
| original letters | 0.617 | 0.605 | **0.871** |
| variables renamed | 0.279 | 0.264 | **0.871** |
| DAG edge F1 | 0.703 | 0.610 | **0.942** |

Text models learn the dataset's alphabetical-order shortcut; the oracle does not depend on letter names.
Code and report: [github.com/abhijeetnst/dl-aml](https://github.com/abhijeetnst/dl-aml)"""

load_models = st.cache_resource(show_spinner="Loading models...")(demo.load_models)


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


st.set_page_config(page_title="Corr2Cause causal inference", layout="wide")
st.markdown(HEADER)
tab_c2c, tab_text = st.tabs(["Corr2Cause", "Free text"])

with tab_c2c:
    st.selectbox("Test-set examples", range(len(demo.EXAMPLES)), index=None, key="example", on_change=fill_example,
                 format_func=lambda i: demo.EXAMPLES[i][1], placeholder="Pick an example to fill the inputs")
    st.text_area("Premise", key="premise", height=120)
    st.text_input("Hypothesis", key="hypothesis")
    left, right = st.columns([1, 6])
    clicked = left.button("Check", type="primary")
    right.button("Rename variables", on_click=rename)
    if msg := st.session_state.pop("rename_error", None):
        st.warning(msg)
    if clicked:
        try:
            table, summary, img = demo.check(st.session_state.premise, st.session_state.hypothesis, load_models())
        except ValueError as e:
            st.warning(f"**Could not parse the input.** {e}")
        except Exception as e:
            st.error(f"Scoring failed: {e}")
        else:
            st.dataframe(table, hide_index=True)
            col1, col2 = st.columns(2)
            col1.markdown(summary)
            col2.image(img, caption="PC pattern")

with tab_text:
    text = st.text_area("Text", value=demo.DEFAULT_TEXT, height=120)
    try:
        triples, dag, nodes, (x0, y0) = demo.extract(text)
    except ValueError as e:
        st.warning(str(e))
    else:
        st.dataframe(triples, hide_index=True)
        st.image(dag, caption="Causal graph")
        col1, col2 = st.columns(2)
        x = col1.selectbox("Cause X", nodes, index=nodes.index(x0))
        y = col2.selectbox("Effect Y", nodes, index=nodes.index(y0))
        try:
            st.markdown(demo.intervene(text, x, y))
        except ValueError as e:
            st.warning(str(e))
