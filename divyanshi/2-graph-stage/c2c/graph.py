"""Graph stage: reviewed 'direct' predictions -> one acyclic causal graph per document, with provenance."""
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

from c2c.data import parse_hypothesis, parse_premise


def best_threshold(y_true, prob) -> float:
    """Confidence gate, tuned on validation only: the threshold with the best F1."""
    p, r, t = precision_recall_curve(y_true, prob)
    f1 = 2 * p * r / np.maximum(p + r, 1e-12)
    return float(t[f1[:-1].argmax()])


def build_dag(doc: pd.DataFrame, threshold: float) -> nx.DiGraph:
    """doc: all rows of ONE premise, with columns premise, hypothesis, prob, doc_id, row_id.
    Review rules: confidence gate, no self loops, one direction per pair, break cycles at the weakest edge."""
    g = nx.DiGraph()
    g.add_nodes_from(parse_premise(doc.premise.iloc[0])[0])
    for r in doc.itertuples():
        relation, x, y = parse_hypothesis(r.hypothesis)
        if relation != "direct" or r.prob < threshold or x == y:
            continue
        rival = g.edges.get((y, x)) or g.edges.get((x, y))
        if rival and rival["prob"] >= r.prob:
            continue
        g.remove_edges_from([(y, x), (x, y)])
        g.add_edge(x, y, prob=float(r.prob), doc_id=r.doc_id, row_id=r.row_id)
    while not nx.is_directed_acyclic_graph(g):
        u, v = min(nx.find_cycle(g), key=lambda e: g.edges[e]["prob"])
        g.remove_edge(u, v)
    return g


def gold_edges(doc: pd.DataFrame) -> set[tuple[str, str]]:
    edges = set()
    for h, label in zip(doc.hypothesis, doc.label):
        relation, x, y = parse_hypothesis(h)
        if relation == "direct" and label == 1:
            edges.add((x, y))
    return edges


def edge_scores(pred: dict[str, set], gold: dict[str, set]) -> dict[str, float]:
    """Micro-averaged over documents."""
    tp = sum(len(pred[d] & gold[d]) for d in gold)
    fp = sum(len(pred[d] - gold[d]) for d in gold)
    fn = sum(len(gold[d] - pred[d]) for d in gold)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"edge_precision": p, "edge_recall": r, "edge_f1": 2 * p * r / (p + r) if p + r else 0.0,
            "tp": tp, "fp": fp, "fn": fn}
