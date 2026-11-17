"""UI-independent demo logic: Corr2Cause checks (AML, DistilBERT, symbolic oracle) and free-text interventions."""
import io
import json
import os
import pathlib
import random

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

ROOT = pathlib.Path(__file__).resolve().parents[1]
HUB_REPO = "abhijeetnst/distilbert-c2c"
LOCAL_MODEL = ROOT / "models" / "distilbert-c2c"
MODEL_REPO = os.environ.get("MODEL_REPO", str(LOCAL_MODEL) if LOCAL_MODEL.is_dir() else HUB_REPO)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_VARS = 6  # the dataset's largest premise; the equivalence class is brute-forced over 2^edges orientations


def artifact(name: str) -> str:
    local = ROOT / "artifacts" / name
    if local.exists():
        return str(local)
    return hf_hub_download(HUB_REPO if pathlib.Path(MODEL_REPO).is_dir() else MODEL_REPO, name)


def load_models():
    """-> (AML pipeline, thresholds dict, (tokenizer, DistilBERT))."""
    tok = AutoTokenizer.from_pretrained(MODEL_REPO)
    bert = AutoModelForSequenceClassification.from_pretrained(MODEL_REPO).to(DEVICE).eval()
    return (joblib.load(artifact("aml.joblib")), json.loads(pathlib.Path(artifact("thresholds.json")).read_text()),
            (tok, bert))


def bert_prob(premise: str, hypothesis: str, bert) -> float:
    tok, model = bert
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


def check(premise: str, hypothesis: str, models):
    """-> (scores table, markdown summary, PC pattern image). `models` is load_models()'s result."""
    variables, indep, (relation, x, y) = parse(premise, hypothesis)
    premise, hypothesis = premise.strip(), hypothesis.strip()
    aml_model, th, bert = models
    row = pd.DataFrame({"premise": [premise], "hypothesis": [hypothesis]})
    probs = {"AML (pattern LR)": (aml.predict_proba(aml_model, row)[0], th["aml"]),
             "DistilBERT": (bert_prob(premise, hypothesis, bert), th["bert"]),
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
    parse(premise, hypothesis)
    return rename_variables(premise.strip(), hypothesis.strip(), seed=random.randrange(10**9))


def extract(text: str):
    """-> (triples table, DAG image, node names, default (cause, effect))."""
    triples = extract_triples(text)
    if triples.empty:
        raise ValueError("No causal sentences found. Try 'X causes Y', 'X leads to Y' or 'Y is caused by X'.")
    g = triples_to_dag(triples)
    nodes = sorted(g.nodes)
    for depth, layer in enumerate(nx.topological_generations(g)):  # causes on the left, effects to the right
        g.add_nodes_from(layer, depth=depth)
    img = draw(nodes, g.edges, title="Causal graph from the text", pos=nx.multipartite_layout(g, subset_key="depth"))
    return triples[["sentence", "cause", "verb", "effect"]], img, nodes, next(iter(g.edges))


def intervene(text: str, x: str, y: str) -> str:
    g = triples_to_dag(extract_triples(text))
    if x not in g or y not in g or x == y:
        raise ValueError("Pick two different variables from the extracted graph.")
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
