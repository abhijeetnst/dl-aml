from c2c.data import load_splits, rename_variables
from c2c.oracle import entailed, equivalence_class

COLLIDER = ("Suppose there is a closed system of 3 variables, A, B and C. All the statistical relations among "
            "these 3 variables are as follows: A correlates with C. B correlates with C. However, A is independent of B.")
CHAIN = ("Suppose there is a closed system of 3 variables, A, B and C. All the statistical relations among "
         "these 3 variables are as follows: A correlates with B. A correlates with C. B correlates with C. "
         "However, A and C are independent given B.")


def test_collider_is_fully_oriented():
    assert len(equivalence_class(COLLIDER)) == 1
    assert entailed(COLLIDER, "A directly causes C.") == 1
    assert entailed(COLLIDER, "C directly causes A.") == 0
    assert entailed(COLLIDER, "There exists at least one collider (i.e., common effect) of A and B.") == 1
    assert entailed(COLLIDER, "There exists at least one confounder (i.e., common cause) of A and B.") == 0


def test_chain_leaves_direction_unidentified():
    assert len(equivalence_class(CHAIN)) == 3  # A->B->C, A<-B<-C, A<-B->C
    assert entailed(CHAIN, "A directly causes B.") == 0
    assert entailed(CHAIN, "B directly causes A.") == 0
    assert entailed(CHAIN, "There exists at least one collider (i.e., common effect) of A and C.") == 0


def test_oracle_is_invariant_to_variable_renaming():
    for hyp in ["A directly causes C.", "There exists at least one collider (i.e., common effect) of A and B."]:
        for seed in range(5):
            assert entailed(*rename_variables(COLLIDER, hyp, seed)) == entailed(COLLIDER, hyp)


def test_oracle_agrees_with_most_test_labels():
    test = load_splits()["test"]
    agree = sum(entailed(p, h) == y for p, h, y in zip(test.premise, test.hypothesis, test.label))
    assert agree / len(test) >= 0.95  # measured 0.962; the rest are audited in the notebook
