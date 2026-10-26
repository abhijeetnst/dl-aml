"""Symbolic reasoner: independence facts -> PC skeleton + v-structures -> Markov equivalence class -> entailment."""
import itertools
from functools import lru_cache

import networkx as nx

from c2c.data import parse_hypothesis, parse_premise

RELATIONS = {
    "direct": lambda g, x, y: g.has_edge(x, y),
    "indirect": lambda g, x, y: x in nx.ancestors(g, y) and not g.has_edge(x, y),
    "collider": lambda g, x, y: any(g.has_edge(x, z) and g.has_edge(y, z) for z in g),
    "confounder": lambda g, x, y: any(g.has_edge(z, x) and g.has_edge(z, y) for z in g),
}


def pc_pattern(variables, indep) -> tuple[nx.Graph, set]:
    """PC algorithm with a perfect independence oracle (the premise lists every independence)."""
    sepset = {}
    for x, y, s in indep:
        sepset.setdefault(frozenset((x, y)), s)
    skel = nx.Graph()
    skel.add_nodes_from(variables)
    skel.add_edges_from(p for p in itertools.combinations(variables, 2) if frozenset(p) not in sepset)
    vs = set()
    for x, y in itertools.combinations(sorted(variables), 2):
        if not skel.has_edge(x, y):
            for z in set(skel[x]) & set(skel[y]):
                if z not in sepset[frozenset((x, y))]:
                    vs.add((x, z, y))
    return skel, vs


def v_structures(g: nx.DiGraph) -> set:
    return {(x, z, y) for z in g for x, y in itertools.combinations(sorted(g.predecessors(z)), 2)
            if not g.has_edge(x, y) and not g.has_edge(y, x)}


@lru_cache(maxsize=None)
def equivalence_class(premise: str) -> tuple[nx.DiGraph, ...]:
    """Every DAG with the premise's skeleton and v-structures.
    Brute force over 2^|edges| orientations; fine up to 6 variables (15 edges = 32k checks)."""
    variables, indep = parse_premise(premise)
    skel, vs = pc_pattern(variables, indep)
    edges = sorted(tuple(sorted(e)) for e in skel.edges)
    dags = []
    for flips in itertools.product((False, True), repeat=len(edges)):
        g = nx.DiGraph()
        g.add_nodes_from(variables)
        g.add_edges_from((b, a) if f else (a, b) for (a, b), f in zip(edges, flips))
        if nx.is_directed_acyclic_graph(g) and v_structures(g) == vs:
            dags.append(g)
    return tuple(dags)


def entailed(premise: str, hypothesis: str) -> int:
    """Corr2Cause semantics: a hypothesis is valid iff it holds in every DAG of the equivalence class."""
    relation, x, y = parse_hypothesis(hypothesis)
    return int(all(RELATIONS[relation](g, x, y) for g in equivalence_class(premise)))
