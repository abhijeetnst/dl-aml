"""UI-independent app logic: Corr2Cause checks (AML, DistilBERT, symbolic oracle), batch scoring, free-text interventions."""
import functools
import io
import json
import logging
import os
import pathlib
import random
import subprocess

import joblib
import networkx as nx
import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from matplotlib.figure import Figure
from PIL import Image
from sklearn.metrics import precision_recall_fscore_support
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from c2c import aml
from c2c.bert import encode
from c2c.bn import ate, noisy_or_bn
from c2c.data import parse_hypothesis, parse_premise, rename_variables
from c2c.oracle import entailed, equivalence_class, pc_pattern
from c2c.text import extract_triples, triples_to_dag

log = logging.getLogger(__name__)
ROOT = pathlib.Path(__file__).resolve().parents[1]
HUB_REPO = "abhijeetnst/distilbert-c2c"
LOCAL_MODEL = ROOT / "models" / "distilbert-c2c"
MODEL_REPO = os.environ.get("MODEL_REPO", str(LOCAL_MODEL) if LOCAL_MODEL.is_dir() else HUB_REPO)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_VARS = 6  # the dataset's largest premise; the equivalence class is brute-forced over 2^edges orientations
MAX_PREMISE, MAX_TEXT, MAX_TRIPLES = 6_000, 5_000, 50  # characters, characters, extracted triples
MAX_ROWS, MAX_BYTES, BATCH = 500, 2 * 1024 * 1024, 32  # batch CSV limits; DistilBERT pairs per forward pass
SYSTEMS = {"aml": "AML (pattern LR)", "bert": "DistilBERT", "oracle": "Symbolic oracle"}


@functools.cache
def version() -> str:
    """Short git SHA of the running code, or 'dev' outside a git checkout."""
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                             timeout=5, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "dev"
    return sha or "dev"


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


def bert_probs(premises, hypotheses, bert, progress=None) -> list[float]:
    """DistilBERT P(valid) per pair, BATCH pairs per forward pass; progress(done, total) runs after each batch."""
    tok, model = bert
    out = []
    for i in range(0, len(premises), BATCH):
        enc = encode(tok, premises[i:i + BATCH], hypotheses[i:i + BATCH], padding=True, return_tensors="pt")
        with torch.no_grad():
            out += model(**enc.to(DEVICE)).logits.softmax(-1)[:, 1].tolist()
        if progress:
            progress(len(out), len(premises))
    return out


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
    if len(premise) > MAX_PREMISE:
        raise ValueError(f"The premise is longer than {MAX_PREMISE:,} characters.")
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
             "DistilBERT": (bert_probs([premise], [hypothesis], bert)[0], th["bert"]),
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


def read_csv(data: bytes) -> pd.DataFrame:
    """Uploaded batch file -> premise, hypothesis and, if present, a 0/1 label column. Raises ValueError."""
    if len(data) > MAX_BYTES:
        raise ValueError(f"The file is larger than {MAX_BYTES // 2**20} MB.")
    try:
        df = pd.read_csv(io.BytesIO(data), dtype=str, keep_default_na=False)
    except (ValueError, UnicodeDecodeError, pd.errors.ParserError):
        raise ValueError("The file is not a readable UTF-8 CSV.") from None
    df.columns = df.columns.str.strip().str.lower()
    if missing := {"premise", "hypothesis"} - set(df.columns):
        raise ValueError(f"Missing column(s): {', '.join(sorted(missing))}. "
                         "Expected premise, hypothesis and an optional label.")
    if not 1 <= len(df) <= MAX_ROWS:
        raise ValueError(f"The file must have 1 to {MAX_ROWS} rows; it has {len(df)}.")
    df = df[[c for c in ("premise", "hypothesis", "label") if c in df]]
    if "label" in df:
        labels = df.label.str.strip().str.lower().map({"1": 1, "0": 0, "true": 1, "false": 0})
        if labels.isna().any():
            raise ValueError("The label column must contain only 0 or 1.")
        df = df.assign(label=labels.astype(int))
    return df


def _row_error(premise: str, hypothesis: str) -> str:
    try:
        parse(premise, hypothesis)
        entailed(premise.strip(), hypothesis.strip())  # cheap; surfaces premises the oracle cannot handle
    except ValueError as e:
        return str(e)
    except Exception as e:
        log.warning("batch row rejected: %s", type(e).__name__)
        return "This row could not be scored."
    return ""


def score_batch(df: pd.DataFrame, models, bert_fn=None, progress=None):
    """read_csv's frame through the three systems -> (results, per-system metrics or None).
    A row that fails to parse gets a message in `error` and no scores; metrics need a label column.
    `bert_fn(premises, hypotheses, progress=...)` defaults to bert_probs on models[2]."""
    aml_model, th, bert = models
    out = df.reset_index(drop=True)
    errors = [_row_error(p, h) for p, h in zip(out.premise, out.hypothesis)]
    ok = np.array([not e for e in errors])
    for key in SYSTEMS:
        out[f"{key}_p"], out[f"{key}_pred"] = np.nan, pd.NA
    if ok.any():
        p, h = out.premise[ok].str.strip().tolist(), out.hypothesis[ok].str.strip().tolist()
        scores = {"aml": (aml.predict_proba(aml_model, pd.DataFrame({"premise": p, "hypothesis": h})), th["aml"]),
                  "bert": ((bert_fn or functools.partial(bert_probs, bert=bert))(p, h, progress=progress), th["bert"]),
                  "oracle": ([entailed(a, b) for a, b in zip(p, h)], 1.0)}
        for key, (probs, t) in scores.items():
            probs = np.asarray(probs, dtype=float)
            out.loc[ok, f"{key}_p"] = probs.round(4)
            out.loc[ok, f"{key}_pred"] = (probs >= t).astype(int)
    for key in SYSTEMS:
        out[f"{key}_pred"] = out[f"{key}_pred"].astype("Int64")
    out["error"] = errors
    if "label" not in out or not ok.any():
        return out, None
    rows = []
    for key, name in SYSTEMS.items():
        prec, rec, f1, _ = precision_recall_fscore_support(out.label[ok], out[f"{key}_pred"][ok].astype(int),
                                                           average="binary", zero_division=0)
        rows.append({"system": name, "precision": round(prec, 3), "recall": round(rec, 3), "f1": round(f1, 3),
                     "rows": int(ok.sum())})
    return out, pd.DataFrame(rows)


def triples(text: str) -> pd.DataFrame:
    """extract_triples within the app's limits. Raises ValueError with a readable message."""
    if len(text) > MAX_TEXT:
        raise ValueError(f"The text is longer than {MAX_TEXT:,} characters.")
    df = extract_triples(text)
    if df.empty:
        raise ValueError("No causal sentences found. Try 'X causes Y', 'X leads to Y' or 'Y is caused by X'.")
    if len(df) > MAX_TRIPLES:
        raise ValueError(f"The text has {len(df)} causal statements; at most {MAX_TRIPLES} are supported.")
    return df


def extract(text: str):
    """-> (triples table, DAG image, node names, default (cause, effect))."""
    found = triples(text)
    g = triples_to_dag(found)
    nodes = sorted(g.nodes)
    for depth, layer in enumerate(nx.topological_generations(g)):  # causes on the left, effects to the right
        g.add_nodes_from(layer, depth=depth)
    img = draw(nodes, g.edges, title="Causal graph from the text", pos=nx.multipartite_layout(g, subset_key="depth"))
    return found[["sentence", "cause", "verb", "effect"]], img, nodes, next(iter(g.edges))


def intervene(text: str, x: str, y: str) -> str:
    g = triples_to_dag(triples(text))
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
SAMPLE_LABELS = [1, 1, 0]  # Corr2Cause labels of EXAMPLES[:3]
DEFAULT_TEXT = ("Heavy storms cause floods. Floods lead to crop failures. Crop failures raise food prices. "
                "Food prices trigger protests. Heavy storms also cause power outages.")


def sample_csv() -> bytes:
    """Batch-tab format example: three test rows with their labels."""
    df = pd.DataFrame(EXAMPLES[:3], columns=["premise", "hypothesis"]).assign(label=SAMPLE_LABELS)
    return df.to_csv(index=False).encode()
