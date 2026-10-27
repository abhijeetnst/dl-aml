import networkx as nx
import pytest

from c2c.bn import ate, fit_bn, noisy_or_bn, query_table


def dag(*edges, nodes="ABC"):
    g = nx.DiGraph(list(edges))
    g.add_nodes_from(nodes)
    return g


def test_noisy_or_effect_is_known_exactly():
    bn = noisy_or_bn(dag(("A", "B")))
    assert ate(bn, "A", "B") == pytest.approx(0.72)  # (1 - 0.9*0.2) - 0.1
    assert ate(bn, "B", "A") == pytest.approx(0.0)


def test_fitted_bn_recovers_effect_and_survives_many_parents():
    wide = dag(*[(p, "F") for p in "ABCDE"], nodes="ABCDEF")
    data = noisy_or_bn(wide).simulate(n_samples=20, seed=0, show_progress=False)
    assert len(data.groupby(list("ABCDE"), observed=True)) < 32  # some parent configurations never occur
    assert data["F"].nunique() == 1  # and F never takes one of its two values
    fitted = fit_bn(wide, data)
    assert abs(ate(fitted, "A", "F")) <= 1.0  # finite, no NaN or crash
    chain = dag(("A", "B"), ("B", "C"))
    fitted = fit_bn(chain, noisy_or_bn(chain).simulate(n_samples=5000, seed=0, show_progress=False))
    assert ate(fitted, "A", "B") == pytest.approx(0.72, abs=0.05)


def test_query_table_penalises_missing_and_reversed_edges():
    gold = dag(("A", "B"), ("B", "C"))
    t = query_table(gold, {"perfect": gold, "empty": dag(), "reversed": dag(("B", "A"), ("C", "B"))}).set_index("query")
    assert set(t.index) == {"ATE A->B", "ATE B->A", "ATE A->C", "ATE C->A", "ATE B->C", "ATE C->B"}
    err = (t.drop(columns="gold").sub(t.gold, axis=0)).abs().mean()
    assert err["perfect"] < 0.05 and err["empty"] > 0.2 and err["reversed"] > 0.2


def test_simulation_does_not_depend_on_edge_insertion_order():
    # gold DAGs are built from a set of edges, whose order changes between Python processes
    a = dag(("A", "C"), ("B", "C"), ("C", "D"), nodes="ABCD")
    b = dag(("C", "D"), ("B", "C"), ("A", "C"), nodes="DCBA")
    sim = lambda g: noisy_or_bn(g).simulate(n_samples=200, seed=0, show_progress=False).sort_index(axis=1)
    assert sim(a).equals(sim(b))
