"""Gradio demo: Corr2Cause checker (AML, DistilBERT, symbolic oracle) and free-text causal extraction + do-queries."""
import functools
import io
import json
import os
import pathlib
import random

import gradio as gr
import joblib
import networkx as nx
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from matplotlib.figure import Figure
from PIL import Image
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from c2c import aml
from c2c.bert import encode
from c2c.bn import ate, noisy_or_bn
from c2c.data import parse_hypothesis, parse_premise, rename_variables
from c2c.oracle import entailed, equivalence_class, pc_pattern
from c2c.text import extract_triples, triples_to_dag

MODEL_REPO = os.environ.get("MODEL_REPO", "models/distilbert-c2c")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_VARS = 6  # the dataset's largest premise; the equivalence class is brute-forced over 2^edges orientations


def artifact(name: str) -> str:
    local = pathlib.Path("artifacts") / name
    return str(local) if local.exists() else hf_hub_download(MODEL_REPO, name)


@functools.cache
def aml_model():
    return joblib.load(artifact("aml.joblib"))


@functools.cache
def thresholds() -> dict:
    return json.loads(pathlib.Path(artifact("thresholds.json")).read_text())


@functools.cache
def bert_model():
    tok = AutoTokenizer.from_pretrained(MODEL_REPO)
    return tok, AutoModelForSequenceClassification.from_pretrained(MODEL_REPO).to(DEVICE).eval()


def bert_prob(premise: str, hypothesis: str) -> float:
    tok, model = bert_model()
    enc = encode(tok, [premise], [hypothesis], return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        return float(model(**enc).logits.softmax(-1)[0, 1])


def draw(nodes, arrows, lines=(), title="", pos=None) -> Image.Image:
    """`arrows` drawn directed, `lines` undirected; circular layout unless `pos` is given. Returned as a PNG image."""
    fig = Figure(figsize=(9, 4) if pos else (6, 4))
    ax = fig.add_subplot()
    ax.margins(0.2)
    nodes = sorted(nodes)
    pos = pos or nx.circular_layout(nodes)
    g = nx.Graph()
    g.add_nodes_from(nodes)
    nx.draw_networkx_nodes(g, pos, ax=ax, node_color="#dde6f0", node_size=900)
    nx.draw_networkx_labels(g, pos, ax=ax, font_size=9 if max(map(len, nodes), default=1) > 1 else 12)
    nx.draw_networkx_edges(g, pos, ax=ax, edgelist=list(lines), width=1.5)
    nx.draw_networkx_edges(nx.DiGraph(list(arrows)), pos, ax=ax, edge_color="tab:red", width=1.5, arrowsize=18,
                           node_size=900)
    ax.set_title(title, fontsize=10)
    ax.axis("off")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    return Image.open(buf)


def parse(premise: str, hypothesis: str):
    """-> (variables, indep, triple). Raises ValueError with a readable message."""
    try:
        variables, indep = parse_premise(premise.strip())
    except (ValueError, TypeError) as e:
        raise ValueError(str(e) if "unrecognised" in str(e) else
                         "The premise must follow the Corr2Cause template: 'Suppose there is a closed system of "
                         "N variables, A, B and C. All the statistical relations among these N variables are as "
                         "follows: ...'") from None
    triple = parse_hypothesis(hypothesis.strip())
    if not set(triple[1:]) <= set(variables):
        raise ValueError(f"The hypothesis mentions {triple[1]} and {triple[2]}; the premise has {', '.join(variables)}.")
    if len(variables) > MAX_VARS:
        raise ValueError(f"At most {MAX_VARS} variables are supported.")
    return variables, indep, triple


def check(premise: str, hypothesis: str):
    """-> (scores table, markdown summary, PC pattern image)."""
    try:
        variables, indep, (relation, x, y) = parse(premise, hypothesis)
    except ValueError as e:
        gr.Warning(str(e))
        return pd.DataFrame(), f"**Could not parse the input.** {e}", None
    premise, hypothesis = premise.strip(), hypothesis.strip()
    th = thresholds()
    row = pd.DataFrame({"premise": [premise], "hypothesis": [hypothesis]})
    probs = {"AML (pattern LR)": (aml.predict_proba(aml_model(), row)[0], th["aml"]),
             "DistilBERT": (bert_prob(premise, hypothesis), th["bert"]),
             "Symbolic oracle": (float(entailed(premise, hypothesis)), 1.0)}
    table = pd.DataFrame([{"system": k, "P(valid)": round(p, 3), "threshold": round(t, 3),
                           "verdict": "valid" if p >= t else "not valid"} for k, (p, t) in probs.items()])
    n_dags = len(equivalence_class(premise))
    summary = (f"**Parsed triple:** ({relation}, {x}, {y})  \n"
               f"**Markov equivalence class:** {n_dags} DAG{'s share' if n_dags != 1 else ' shares'} this premise's independences; "
               f"the oracle calls the hypothesis valid only if it holds in all of them.")
    skel, vs = pc_pattern(variables, indep)
    arrows = {(a, z) for u, z, v in vs for a in (u, v)}
    lines = [e for e in skel.edges if e not in arrows and e[::-1] not in arrows]
    return table, summary, draw(variables, arrows, lines, "PC skeleton, v-structures in red")


def rename(premise: str, hypothesis: str):
    try:
        parse(premise, hypothesis)
    except ValueError as e:
        gr.Warning(str(e))
        return premise, hypothesis
    return rename_variables(premise.strip(), hypothesis.strip(), seed=random.randrange(10**9))


def extract(text: str):
    triples = extract_triples(text)
    if triples.empty:
        gr.Warning("No causal sentences found. Try 'X causes Y', 'X leads to Y' or 'Y is caused by X'.")
    g = triples_to_dag(triples)
    nodes = sorted(g.nodes)
    table = triples[["sentence", "cause", "verb", "effect"]]
    for depth, layer in enumerate(nx.topological_generations(g)):  # causes on the left, effects to the right
        g.add_nodes_from(layer, depth=depth)
    img = draw(nodes, g.edges, title="Causal graph from the text",
               pos=nx.multipartite_layout(g, subset_key="depth")) if nodes else None
    cause, effect = (list(g.edges)[0] if g.edges else (None, None))
    return table, img, gr.Dropdown(choices=nodes, value=cause), gr.Dropdown(choices=nodes, value=effect)


def intervene(text: str, x: str, y: str) -> str:
    g = triples_to_dag(extract_triples(text))
    if x not in g or y not in g or x == y:
        return "Pick two different variables from the extracted graph (press Extract after editing the text)."
    effect = ate(noisy_or_bn(g), x, y)
    return (f"**ATE** = P({y}=1 | do({x}=1)) − P({y}=1 | do({x}=0)) = **{effect:+.3f}**\n\n"
            "Every variable is binary and each cause independently switches its effect on with probability 0.8 "
            "(noisy-OR, 0.1 leak, root causes are fair coins), so the number reflects the graph structure, "
            "not real-world effect sizes.")


# real test-set rows (causalnlp/corr2cause): test-39, test-11, test-19, test-20, test-514, test-375
P3 = "Suppose there is a closed system of 3 variables, A, B and C. All the statistical relations among these 3 variables are as follows: "
P5 = "Suppose there is a closed system of 5 variables, A, B, C, D and E. All the statistical relations among these 5 variables are as follows: "
COLLIDER = P3 + "A correlates with C. B correlates with C. However, A is independent of B."
FULL = P3 + "A correlates with B. A correlates with C. B correlates with C."
EXAMPLES = [
    [COLLIDER, "There exists at least one collider (i.e., common effect) of A and B."],
    [COLLIDER, "A directly causes C."],
    [FULL, "A directly causes B."],
    [FULL, "A causes something else which causes B."],
    [P5 + "A correlates with B. A correlates with D. A correlates with E. B correlates with D. B correlates with E. "
          "C correlates with D. C correlates with E. D correlates with E. However, A is independent of C. "
          "A and C are independent given B. B is independent of C. B and C are independent given A. "
          "D and E are independent given A, B and C.",
     "There exists at least one confounder (i.e., common cause) of D and E."],
    [P5 + "A correlates with B. A correlates with C. A correlates with D. A correlates with E. B correlates with C. "
          "B correlates with D. B correlates with E. C correlates with D. C correlates with E. D correlates with E. "
          "However, A and C are independent given B. B and D are independent given A and C. "
          "B and D are independent given A, C and E. B and E are independent given A and C. "
          "B and E are independent given A, C and D.",
     "B is a cause for E, but not a direct one."],
]
DEFAULT_TEXT = ("Heavy storms cause floods. Floods lead to crop failures. Crop failures raise food prices. "
                "Food prices trigger protests. Heavy storms also cause power outages.")
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

with gr.Blocks(title="Corr2Cause: neuro-symbolic causal inference") as demo:
    gr.Markdown(HEADER)
    with gr.Tab("Corr2Cause"):
        premise = gr.Textbox(label="Premise", lines=4)
        hypothesis = gr.Textbox(label="Hypothesis")
        gr.Examples(EXAMPLES, inputs=[premise, hypothesis], label="Test-set examples")
        with gr.Row():
            check_btn = gr.Button("Check", variant="primary")
            rename_btn = gr.Button("Rename variables")
        scores = gr.Dataframe(label="Scores")
        with gr.Row():
            summary = gr.Markdown()
            pattern = gr.Image(label="PC pattern", type="pil")
        check_btn.click(check, [premise, hypothesis], [scores, summary, pattern])
        rename_btn.click(rename, [premise, hypothesis], [premise, hypothesis])
    with gr.Tab("Free text"):
        text = gr.Textbox(label="Text", value=DEFAULT_TEXT, lines=5)
        extract_btn = gr.Button("Extract", variant="primary")
        triples = gr.Dataframe(label="Causal triples")
        dag = gr.Image(label="Causal graph", type="pil")
        with gr.Row():
            cause = gr.Dropdown(label="Cause X")
            effect = gr.Dropdown(label="Effect Y")
        intervene_btn = gr.Button("Intervene")
        ate_out = gr.Markdown()
        extract_btn.click(extract, text, [triples, dag, cause, effect])
        intervene_btn.click(intervene, [text, cause, effect], ate_out)

if __name__ == "__main__":
    aml_model(), thresholds(), bert_model()  # load once at startup, not on the first click
    demo.launch()
