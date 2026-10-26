# Prashant: start here

You own the **symbolic oracle**, our reference system, and the **label audit**.
Read `TEAM_GUIDE.md` for the whole project. This file covers your part and your exact commands.

## Your contribution

| Commit | Files | What it adds |
|---|---|---|
| 1 | `c2c/oracle.py`, `tests/test_oracle.py` | PC algorithm → Markov equivalence class → entailment |
| 2 | `review/label_audit_review.csv` | verdicts on 10 rows where the oracle and the dataset label disagree |

## How to explain it (2-minute version)

> "The oracle answers: *given only the premise, is this causal claim logically forced?*
>
> A Corr2Cause premise lists **every** independence among its variables, so we have a perfect
> independence test. That's exactly the input the **PC algorithm** needs.
>
> Step 1, the skeleton: two variables are connected if the premise never says they're independent.
> For each separated pair we remember the conditioning set that separated them, the *sepset*.
>
> Step 2, v-structures: if X and Y aren't connected but share a neighbour Z, and Z is **not** in their
> sepset, the arrows must be X → Z ← Y. That's a collider.
>
> Step 3, the equivalence class: the skeleton plus the v-structures don't fix every arrow. So we try every
> way of orienting the remaining edges, 2^|edges| options, and keep those that are acyclic and have exactly
> the same v-structures. Those DAGs are the **Markov equivalence class**: every graph the data can't tell
> apart.
>
> Step 4, entailment: a hypothesis is valid only if it holds in **every** DAG of the class. That's the
> dataset's own definition. 'Direct' means the edge X→Y exists, 'indirect' means X is an ancestor of Y but
> not a parent, 'collider' means a common child, and 'confounder' means a common parent."

Then the audit:

> "The oracle agrees with 96.2% of test labels. I looked at the disagreements: 44 rows across 12
> premises. In the 10 we reviewed, 5 dataset labels contradict every graph consistent with the premise,
> so the label is wrong. The other 5 match a looser definition of the relation. So part of the oracle's
> 'error' is really label noise."

## Key design decisions (and why)

- **Brute force over orientations instead of Meek's rules.** At most 6 variables means 15 edges and
  32,768 orientations. That's simple and obviously correct. Results are cached per premise
  (`lru_cache`). For bigger graphs you would switch to a CPDAG with Meek's orientation rules.
- **"Holds in every DAG" semantics.** A claim the data can't distinguish is not entailed. That's why the
  chain A–B–C gives 3 DAGs and "A directly causes B" is **not** valid there.
- **We keep the strict "indirect" definition** (ancestor but not parent). On test, the looser definition
  "a directed path of length ≥ 2" matches more labels (99.2% vs 95.1% of the 388 indirect rows). On
  validation the difference is one row (333 vs 332 of 337). Switching because of test would be tuning on
  test, so we didn't.

## What the tests prove (`tests/test_oracle.py`)

- A collider premise (A ⊥ B, both correlate with C) gives exactly 1 DAG, A → C ← B. "A directly causes C"
  is valid; "C directly causes A" isn't.
- A chain premise (A ⊥ C | B) gives 3 DAGs, so neither direction of A–B is entailed.
- Renaming the letters never changes the answer.
- The oracle agrees with ≥ 95% of test labels (measured: 96.2%).

## Likely questions

- **Why is the oracle's F1 only 0.871 if it's "perfect"?** It's perfect *relative to the premise*. F1 is
  measured against dataset labels, which have noise and a different "indirect" definition (its indirect
  recall is 0.455). Overall it agrees with 96.2% of rows.
- **Why does it score the same on shuffled letters?** Logic doesn't depend on names. That's the point of
  the comparison: the learned models drop to ~0.27, the oracle stays at 0.871.
- **What is a Markov equivalence class?** The set of DAGs that imply exactly the same independences. Two
  DAGs are equivalent iff they have the same skeleton and the same v-structures (Verma & Pearl).
- **Why include a system that isn't learned?** It's the upper bound for what the text supports, it exposes
  the shortcut in the learned models, and it let us find label errors in the dataset.
- **Complexity?** O(2^E) per premise, E ≤ 15, cached. About 0.1 s for the whole test set (1,162 rows).

## How to push

Wait until Abhijeet says **step 1 is pushed** and you have **accepted the GitHub collaborator invite**.
Windows: use **Git Bash**.

Extract `prashant.zip` in the folder where you'll clone, so `prashant/` sits next to `dl-aml/`.

```bash
git clone https://github.com/abhijeetnst/dl-aml.git
cd dl-aml
git config user.name "Prashant"
git config user.email "your-github-email@example.com"   # the email on YOUR GitHub account
cp -R ../prashant/1-oracle/. .
git add c2c/oracle.py tests/test_oracle.py
git commit -m "feat: symbolic oracle via PC skeleton and equivalence-class enumeration"
cp -R ../prashant/2-label-audit/. .
```

**Recommended before the next commit:** open `review/label_audit_review.csv`. For each of the 10 rows,
read the premise and the `evidence`, and fill the empty `team_verdict` column with `oracle` (the label is
wrong), `label` (the oracle is wrong) or `unclear`. This is what makes it a real human review. Save as CSV.

```bash
git add review/label_audit_review.csv
git commit -m "docs: audit of test rows where the oracle and the dataset labels disagree"
git pull --rebase
git push
```

### Check (optional, large downloads)

```bash
python3 -m venv .venv && source .venv/bin/activate      # Windows Git Bash: source .venv/Scripts/activate
pip install -r requirements.txt
pytest -q tests/test_data.py tests/test_oracle.py
```
