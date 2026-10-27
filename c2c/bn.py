"""Bayesian-network stage: simulate from the gold graph, fit each system's graph with pgmpy, compare do-queries."""
import itertools

import networkx as nx
import pandas as pd
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import CausalInference
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.parameter_estimator import DiscreteMLE


def noisy_or_bn(dag: nx.DiGraph, leak: float = 0.1, strength: float = 0.8) -> DiscreteBayesianNetwork:
    """Gold BN with strong, known effects. pgmpy's random CPDs give near-zero effects, which makes queries uninformative."""
    # sorted, so a seeded simulate() gives the same rows however the DAG was assembled
    bn = DiscreteBayesianNetwork(sorted(dag.edges))
    bn.add_nodes_from(sorted(dag.nodes))
    for v in sorted(dag.nodes):
        parents = sorted(dag.predecessors(v))
        if not parents:
            bn.add_cpds(TabularCPD(v, 2, [[0.5], [0.5]]))
            continue
        # column order matches pgmpy: last parent varies fastest, same as itertools.product
        p1 = [1 - (1 - leak) * (1 - strength) ** sum(cfg) for cfg in itertools.product((0, 1), repeat=len(parents))]
        bn.add_cpds(TabularCPD(v, 2, [[1 - p for p in p1], p1], evidence=parents, evidence_card=[2] * len(parents)))
    assert bn.check_model()
    return bn


def fit_bn(dag: nx.DiGraph, data: pd.DataFrame) -> DiscreteBayesianNetwork:
    bn = DiscreteBayesianNetwork(list(dag.edges))
    bn.add_nodes_from(dag.nodes)
    # maximum likelihood with both states declared, so a value missing from the data keeps cardinality 2
    bn.fit(data[list(dag.nodes)], estimator=DiscreteMLE(state_names={v: [0, 1] for v in dag.nodes}))
    return bn


def ate(bn: DiscreteBayesianNetwork, x: str, y: str) -> float:
    """Average treatment effect P(y=1 | do(x=1)) - P(y=1 | do(x=0))."""
    if y not in nx.descendants(bn, x):  # do(x) cannot move a non-descendant (pgmpy also refuses such queries)
        return 0.0
    ci = CausalInference(bn)
    p1 = lambda v: float(ci.query([y], do={x: v}, show_progress=False).values[1])
    return p1(1) - p1(0)


def query_table(gold_dag: nx.DiGraph, systems: dict[str, nx.DiGraph], n: int = 5000, seed: int = 0) -> pd.DataFrame:
    """Known queries: every ordered pair linked by a directed path in the gold graph, in both directions
    (forward has a real effect, backward must be ~0 and catches reversed edges)."""
    gold = noisy_or_bn(gold_dag)
    data = gold.simulate(n_samples=n, seed=seed, show_progress=False)
    fitted = {"gold_structure_fitted": fit_bn(gold_dag, data)} | {k: fit_bn(d, data) for k, d in systems.items()}
    rows = []
    for x, y in itertools.permutations(sorted(gold_dag.nodes), 2):
        if nx.has_path(gold_dag, x, y) or nx.has_path(gold_dag, y, x):
            rows.append({"query": f"ATE {x}->{y}", "gold": ate(gold, x, y)} | {k: ate(m, x, y) for k, m in fitted.items()})
    return pd.DataFrame(rows)
