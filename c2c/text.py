"""Free-text stage: causal verbs in a spaCy dependency parse -> (cause, effect) triples with provenance -> one DAG."""
import functools

import networkx as nx
import pandas as pd
import spacy

from c2c.graph import add_one_direction, break_cycles

CAUSAL_VERBS = {"cause", "lead", "result", "increase", "reduce", "raise", "lower", "trigger", "produce",
                "induce", "prevent", "drive"}
EFFECT_PREP = {"lead": "to", "result": "in"}  # "X leads to Y", "X results in Y"
COLUMNS = ["doc_id", "row_id", "sentence_id", "sentence", "cause", "effect", "verb", "prob"]


@functools.cache
def _nlp():
    return spacy.load("en_core_web_sm")


def _name(token, chunks) -> str:
    """Noun chunk around token, without determiners, lowercased and lemmatised: 'the storms' -> 'storm'."""
    span = chunks.get(token.i, token.doc[token.i:token.i + 1])
    return " ".join(t.lemma_.lower() for t in span if t.pos_ != "DET")


def _pobj(prep):
    return next((c for c in prep.children if c.dep_ == "pobj"), None)


def _pair(verb):
    """(cause, effect) tokens for one causal verb, or None. Active: nsubj -> dobj (or pobj of to/in);
    passive: agent pobj -> nsubjpass."""
    kids = {c.dep_: c for c in verb.children}  # one child per relation; conjunctions ("X and Y") are skipped
    if "nsubjpass" in kids and "agent" in kids:
        return _pobj(kids["agent"]), kids["nsubjpass"]
    effect = kids.get("dobj")
    if effect is None and verb.lemma_ in EFFECT_PREP:
        preps = [c for c in verb.children if c.dep_ == "prep" and c.lower_ == EFFECT_PREP[verb.lemma_]]
        effect = _pobj(preps[0]) if preps else None
    return (kids["nsubj"], effect) if "nsubj" in kids else None


def extract_triples(text: str) -> pd.DataFrame:
    doc = _nlp()(text)
    chunks = {c.root.i: c for c in doc.noun_chunks}
    rows = []
    for s_id, sent in enumerate(doc.sents):
        k = 0
        for verb in (t for t in sent if t.pos_ == "VERB" and t.lemma_ in CAUSAL_VERBS):
            pair = _pair(verb)
            if not pair or None in pair:
                continue
            cause, effect = (_name(t, chunks) for t in pair)
            if cause == effect:
                continue
            rows.append({"doc_id": "text", "row_id": f"s{s_id}-{k}", "sentence_id": s_id, "sentence": sent.text,
                         "cause": cause, "effect": effect, "verb": verb.lemma_, "prob": 1.0})
            k += 1
    return pd.DataFrame(rows, columns=COLUMNS)


def triples_to_dag(df: pd.DataFrame) -> nx.DiGraph:
    """Review rules of graph.build_dag: one direction per pair (the earlier sentence wins a tie),
    cycles broken by dropping the edge from the latest sentence."""
    g = nx.DiGraph()
    for r in df.sort_values("sentence_id", kind="stable").itertuples():
        g.add_nodes_from([r.cause, r.effect])
        add_one_direction(g, r.cause, r.effect, prob=float(r.prob), doc_id=r.doc_id, row_id=r.row_id,
                          sentence_id=int(r.sentence_id))
    return break_cycles(g, weakest=lambda d: -d["sentence_id"])
