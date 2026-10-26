import networkx as nx
import pandas as pd

from c2c.graph import best_threshold, build_dag, edge_scores, gold_edges

PREMISE = ("Suppose there is a closed system of 3 variables, A, B and C. All the statistical relations among "
           "these 3 variables are as follows: A correlates with B. A correlates with C. B correlates with C.")


def doc(rows):
    return pd.DataFrame([{"premise": PREMISE, "hypothesis": h, "prob": p, "label": 0, "doc_id": "d1",
                          "row_id": f"r{i}"} for i, (h, p) in enumerate(rows)])


def test_gate_self_loop_and_provenance():
    g = build_dag(doc([("A directly causes B.", 0.9), ("B directly causes C.", 0.2),
                       ("A directly causes A.", 0.99), ("There exists at least one collider (i.e., common effect) of A and B.", 0.9)]), 0.5)
    assert set(g.edges) == {("A", "B")}
    assert set(g.nodes) == {"A", "B", "C"}
    assert g.edges["A", "B"] == {"prob": 0.9, "doc_id": "d1", "row_id": "r0"}


def test_opposite_directions_keep_the_more_confident():
    g = build_dag(doc([("A directly causes B.", 0.6), ("B directly causes A.", 0.8)]), 0.5)
    assert set(g.edges) == {("B", "A")}


def test_cycle_is_broken_at_weakest_edge():
    g = build_dag(doc([("A directly causes B.", 0.9), ("B directly causes C.", 0.8), ("C directly causes A.", 0.7)]), 0.5)
    assert nx.is_directed_acyclic_graph(g)
    assert set(g.edges) == {("A", "B"), ("B", "C")}


def test_gold_edges_and_scores():
    d = doc([("A directly causes B.", 0), ("B directly causes C.", 0)])
    d["label"] = [1, 0]
    assert gold_edges(d) == {("A", "B")}
    s = edge_scores({"d1": {("A", "B"), ("C", "B")}}, {"d1": {("A", "B"), ("A", "C")}})
    assert (s["tp"], s["fp"], s["fn"], s["edge_precision"]) == (1, 1, 1, 0.5)


def test_best_threshold_separates_clean_scores():
    assert 0.3 < best_threshold([0, 0, 1, 1], [0.1, 0.3, 0.7, 0.9]) <= 0.7
