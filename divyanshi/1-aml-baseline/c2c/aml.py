"""AML baseline: surface features read off the parsed premise -> logistic regression, one weight set per relation."""
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from c2c.data import parse_hypothesis, parse_premise


def pair_features(premise: str, hypothesis: str) -> dict[str, float]:
    variables, indep = parse_premise(premise)
    relation, x, y = parse_hypothesis(hypothesis)
    separated = {}
    for a, b, s in indep:
        separated.setdefault(frozenset((a, b)), []).append(s)
    adjacent = lambda u, v: frozenset((u, v)) not in separated
    seps = separated.get(frozenset((x, y)), [])
    feats = {
        "adjacent": float(adjacent(x, y)),
        "marginal_corr": float(frozenset() not in seps),
        "n_sepsets": float(len(seps)),
        "n_common_neighbours": float(sum(adjacent(x, z) and adjacent(y, z) for z in variables if z not in (x, y))),
        "n_vars": float(len(variables)),
        "x_before_y": float(x < y),  # deliberately included: the renamed-variable test exposes this shortcut
    }
    # crossing with the relation lets one linear model learn a separate rule per hypothesis type
    return {f"{relation}|{k}": v for k, v in feats.items()} | {f"bias|{relation}": 1.0}


def train_aml(df):
    X = [pair_features(p, h) for p, h in zip(df.premise, df.hypothesis)]
    model = make_pipeline(DictVectorizer(), LogisticRegression(max_iter=5000, class_weight="balanced"))
    return model.fit(X, df.label)


def predict_proba(model, df):
    return model.predict_proba([pair_features(p, h) for p, h in zip(df.premise, df.hypothesis)])[:, 1]
