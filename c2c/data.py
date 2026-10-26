"""Corr2Cause loading, leakage-safe splits, and strict parsing of premise/hypothesis text into symbols."""
import hashlib
import random
import re

import pandas as pd
from datasets import load_dataset

VAR = r"\b[A-Z]\b"
TRAIN_ROWS = 30_000  # fixed subsample so AML and DistilBERT see identical rows; full train is 205k

HYPOTHESES = [
    (r"([A-Z]) directly causes ([A-Z])\.", "direct"),
    (r"([A-Z]) is a cause for ([A-Z]), but not a direct one\.", "indirect"),
    (r"([A-Z]) causes something else which causes ([A-Z])\.", "indirect"),
    (r"There exists at least one collider \(i\.e\., common effect\) of ([A-Z]) and ([A-Z])\.", "collider"),
    (r"There exists at least one confounder \(i\.e\., common cause\) of ([A-Z]) and ([A-Z])\.", "confounder"),
]


def load_splits() -> dict[str, pd.DataFrame]:
    ds = load_dataset("causalnlp/corr2cause")
    out = {}
    for split in ("train", "validation", "test"):
        df = ds[split].to_pandas()
        parts = df["input"].str.removeprefix("Premise: ").str.split("\nHypothesis: ", n=1, expand=True)
        df["premise"], df["hypothesis"] = parts[0], parts[1]
        df["doc_id"] = df["premise"].map(lambda p: hashlib.md5(p.encode()).hexdigest()[:10])
        df["row_id"] = f"{split}-" + df.index.astype(str)
        out[split] = df
    # a premise is a document and must live in exactly one split. The official splits leak:
    # 3 validation premises also occur in test, 2 in train. Test stays untouched; the others give way.
    test_docs = set(out["test"].doc_id)
    out["validation"] = out["validation"][~out["validation"].doc_id.isin(test_docs)].reset_index(drop=True)
    held_out = set(out["validation"].doc_id) | test_docs
    out["train"] = out["train"][~out["train"].doc_id.isin(held_out)].reset_index(drop=True)
    return out


def train_subset(train: pd.DataFrame, n: int = TRAIN_ROWS, seed: int = 0) -> pd.DataFrame:
    return train.sample(n=n, random_state=seed).reset_index(drop=True)


def parse_premise(premise: str) -> tuple[list[str], list[tuple[str, str, frozenset]]]:
    """-> (variables, independencies as (x, y, conditioning_set)). Raises on any sentence it does not understand."""
    head, body = premise.split(" are as follows: ", 1)
    variables = re.findall(VAR, re.search(r"variables, (.+?)\.", head)[1])
    indep = []
    for s in re.split(r"(?<=\.) ", body):
        s = s.removeprefix("However, ")
        if re.fullmatch(r"[A-Z] correlates with [A-Z]\.", s):
            continue  # dependence is the default: the premise lists every independence
        if m := re.fullmatch(r"([A-Z]) is independent of ([A-Z])\.", s):
            indep.append((m[1], m[2], frozenset()))
        elif m := re.fullmatch(r"([A-Z]) and ([A-Z]) are independent given ([A-Z](?:, [A-Z])*(?: and [A-Z])?)\.", s):
            indep.append((m[1], m[2], frozenset(re.findall(VAR, m[3]))))
        else:
            raise ValueError(f"unrecognised premise sentence: {s!r}")
    return variables, indep


def parse_hypothesis(hypothesis: str) -> tuple[str, str, str]:
    """-> candidate causal triple (relation, x, y)."""
    for pattern, relation in HYPOTHESES:
        if m := re.fullmatch(pattern, hypothesis):
            return relation, m[1], m[2]
    raise ValueError(f"unrecognised hypothesis: {hypothesis!r}")


def rename_variables(premise: str, hypothesis: str, seed: int | str) -> tuple[str, str]:
    """Consistently permute variable letters. Labels are invariant; the dataset's A<B<C causal-order shortcut is not."""
    variables, _ = parse_premise(premise)
    shuffled = variables[:]
    random.Random(seed).shuffle(shuffled)
    mapping = dict(zip(variables, shuffled))
    sub = lambda text: re.sub(VAR, lambda m: mapping.get(m[0], m[0]), text)
    return sub(premise), sub(hypothesis)


def renamed(df: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """One permutation per document, so all hypotheses of a premise stay consistent with each other."""
    out = df.copy()
    pairs = [rename_variables(p, h, f"{seed}-{d}") for p, h, d in zip(df.premise, df.hypothesis, df.doc_id)]
    out["premise"], out["hypothesis"] = zip(*pairs)
    return out
