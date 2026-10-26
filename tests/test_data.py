import pandas as pd
import pytest

from c2c.data import load_splits, parse_hypothesis, parse_premise, rename_variables, renamed

COLLIDER = ("Suppose there is a closed system of 3 variables, A, B and C. All the statistical relations among "
            "these 3 variables are as follows: A correlates with C. B correlates with C. However, A is independent of B.")
LONG = ("Suppose there is a closed system of 4 variables, A, B, C and D. All the statistical relations among these "
        "4 variables are as follows: A correlates with B. A correlates with C. A correlates with D. B correlates with C. "
        "B correlates with D. C correlates with D. However, B and D are independent given A. B and D are independent "
        "given A and C. C and D are independent given A. C and D are independent given A and B.")


def test_parse_premise_reads_variables_and_every_independence():
    variables, indep = parse_premise(LONG)
    assert variables == ["A", "B", "C", "D"]
    assert indep == [("B", "D", frozenset("A")), ("B", "D", frozenset("AC")),
                     ("C", "D", frozenset("A")), ("C", "D", frozenset("AB"))]
    assert parse_premise(COLLIDER)[1] == [("A", "B", frozenset())]


def test_parse_premise_rejects_unknown_sentences():
    with pytest.raises(ValueError, match="unrecognised"):
        parse_premise(COLLIDER.replace("A is independent of B.", "A is unrelated to B."))


@pytest.mark.parametrize("text,triple", [
    ("A directly causes B.", ("direct", "A", "B")),
    ("A is a cause for B, but not a direct one.", ("indirect", "A", "B")),
    ("B causes something else which causes A.", ("indirect", "B", "A")),
    ("There exists at least one collider (i.e., common effect) of A and B.", ("collider", "A", "B")),
    ("There exists at least one confounder (i.e., common cause) of C and D.", ("confounder", "C", "D")),
])
def test_parse_hypothesis(text, triple):
    assert parse_hypothesis(text) == triple


def test_rename_touches_only_variable_letters():
    p, h = rename_variables(COLLIDER, "A directly causes C.", seed=3)
    assert p.startswith("Suppose there is a closed system of 3 variables,") and "However," in p
    assert sorted(parse_premise(p)[0]) == ["A", "B", "C"]
    assert parse_hypothesis(h)[0] == "direct"


def test_renamed_uses_one_permutation_per_premise():
    df = pd.DataFrame({"premise": [COLLIDER] * 3, "doc_id": ["d"] * 3,
                       "hypothesis": ["A directly causes C.", "B directly causes C.", "C directly causes A."]})
    out = renamed(df, seed=1)
    assert out.premise.nunique() == 1  # every row of the document got the same letter mapping
    assert len({parse_hypothesis(h)[2] for h in out.hypothesis[:2]}) == 1  # A->C and B->C still share an effect


def test_splits_share_no_premise_and_every_row_parses():
    s = load_splits()
    assert not set(s["train"].doc_id) & set(s["validation"].doc_id)
    assert not set(s["train"].doc_id) & set(s["test"].doc_id)
    assert not set(s["validation"].doc_id) & set(s["test"].doc_id)
    for df in s.values():
        for p in df.premise.unique():
            parse_premise(p)
        for h in df.hypothesis.unique():
            parse_hypothesis(h)
