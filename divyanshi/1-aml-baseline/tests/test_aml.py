from c2c.aml import pair_features

COLLIDER = ("Suppose there is a closed system of 3 variables, A, B and C. All the statistical relations among "
            "these 3 variables are as follows: A correlates with C. B correlates with C. However, A is independent of B.")


def test_pair_features_read_structure_off_the_premise():
    f = pair_features(COLLIDER, "A directly causes C.")
    assert (f["direct|adjacent"], f["direct|marginal_corr"], f["direct|x_before_y"], f["bias|direct"]) == (1, 1, 1, 1)
    f = pair_features(COLLIDER, "There exists at least one collider (i.e., common effect) of A and B.")
    assert (f["collider|adjacent"], f["collider|n_sepsets"], f["collider|n_common_neighbours"]) == (0, 1, 1)
    assert not any(k.startswith("direct|") for k in f)
