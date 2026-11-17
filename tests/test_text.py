import networkx as nx

from c2c.bn import ate, noisy_or_bn
from c2c.text import extract_triples, triples_to_dag


def test_active_prepositional_and_passive_sentences():
    df = extract_triples("Heavy storms cause floods. Smoking leads to lung cancer. Inflation was caused by the storms.")
    assert list(zip(df.cause, df.effect)) == [("heavy storm", "flood"), ("smoking", "lung cancer"),
                                              ("storm", "inflation")]
    assert list(df.row_id) == ["s0-0", "s1-0", "s2-0"]
    assert df.iloc[2].to_dict() == {"doc_id": "text", "row_id": "s2-0", "sentence_id": 2,
                                    "sentence": "Inflation was caused by the storms.", "cause": "storm",
                                    "effect": "inflation", "verb": "cause", "prob": 1.0}


def test_self_loop_is_rejected():
    assert extract_triples("Stress causes stress.").empty


def test_cyclic_text_gives_acyclic_dag_keeping_earlier_sentences():
    g = triples_to_dag(extract_triples("Rain causes floods. Floods cause rain."))
    assert set(g.edges) == {("rain", "flood")}
    g = triples_to_dag(extract_triples("Rain causes floods. Floods cause damage. Damage causes rain."))
    assert nx.is_directed_acyclic_graph(g)
    assert set(g.edges) == {("rain", "flood"), ("flood", "damage")}
    assert g.edges["flood", "damage"] == {"prob": 1.0, "doc_id": "text", "row_id": "s1-0", "sentence_id": 1}


def test_dag_feeds_bayesian_network():
    g = triples_to_dag(extract_triples("Heavy storms cause floods. Floods reduce crop yields."))
    bn = noisy_or_bn(g)
    assert ate(bn, "heavy storm", "flood") > 0.5
