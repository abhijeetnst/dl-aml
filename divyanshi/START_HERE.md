# Divyanshi: start here

You own the **AML baseline**, the **graph stage** that turns predictions into causal graphs, and the
**edge review**. Read `TEAM_GUIDE.md` for the whole project. This file covers your part and your exact
commands.

## Your contribution

| Commit | Files | What it adds |
|---|---|---|
| 1 | `c2c/aml.py`, `tests/test_aml.py` | AML baseline: structural features → logistic regression |
| 2 | `c2c/graph.py`, `tests/test_graph.py` | threshold tuning, review gate, one DAG per premise, edge metrics |
| 3 | `review/edge_review.csv` | evidence-backed verdicts on 38 predicted edges (3 premises) |

## How to explain it (2-minute version)

**AML baseline (`aml.py`):**

> "The baseline doesn't read raw text. It reads features off the parsed premise for the two variables
> in the hypothesis: are they *adjacent* (never declared independent)? Are they *marginally correlated*?
> How many conditioning sets separate them? How many common neighbours do they have? How many variables
> are there? And one deliberate trap: is X's letter before Y's (`x_before_y`)?
>
> Every feature is **crossed with the relation type**, so `direct|adjacent` and `collider|adjacent` are
> separate features with a bias per relation. That lets a single logistic regression learn a different
> rule for each kind of hypothesis. It uses `class_weight='balanced'` because only ~18% of rows are
> positive."

**Graph stage (`graph.py`):**

> "Each system gives a probability per hypothesis. `best_threshold` picks the F1-optimal cut-off on
> **validation** from the precision-recall curve. Then `build_dag` takes all rows of one premise and keeps
> only 'X directly causes Y' claims above the threshold, under three review rules:
> no self-loops; if both X→Y and Y→X pass, keep the more confident one; if there's a cycle, drop its
> weakest edge until the graph is acyclic. Every edge keeps its **provenance**: probability, `doc_id` and
> `row_id`, so you can trace it back to the sentence it came from.
> `gold_edges` builds the true graph from the direct-cause labels, and `edge_scores` gives micro-averaged
> edge precision, recall and F1."

## Key design decisions (and why)

- **Features from the parsed structure, not bag-of-words.** These are the signals PC itself uses
  (adjacency, sepsets, common neighbours), so the baseline is a "learned approximation" of the oracle.
- **`x_before_y` is there on purpose.** It's the only feature renaming changes. On shuffled letters AML
  falls from 0.617 to 0.279 F1, which proves it relies on the alphabetical shortcut.
- **One threshold per system, tuned on validation only.** AML: 0.766, DistilBERT: 0.218, oracle: 1.0.
- **Only "direct" claims become edges.** Indirect, collider and confounder are path-level claims, not
  single arrows.
- **Cycle breaking at the weakest edge** keeps the most-trusted structure, and a Bayesian network needs a
  DAG.

## Results you should know

| | AML | DistilBERT | Oracle |
|---|---|---|---|
| Triple F1 (test) | 0.617 | 0.605 | 0.871 |
| AML F1 per relation: direct / indirect / collider / confounder | 0.703 / 0.452 / 0.667 / 0.000 | | |
| Edge precision / recall / F1 | 0.542 / 1.000 / 0.703 | 0.505 / 0.769 / 0.610 | 0.890 / 1.000 / 0.942 |

Edge review (`review/edge_review.csv`, 3 premises): AML 15 supported / 6 not, and all 6 connect variables
whose direction the premise leaves open. DistilBERT 10 / 7, and 5 of those 7 connect variables the premise
says are **independent**.

## Likely questions

- **Why is AML's confounder F1 exactly 0?** It's a threshold artefact, not "learned nothing". There is one
  global threshold (0.766), and AML's highest confounder score is 0.765, just below it. Its confounder
  AUC is 0.80, so it does rank them reasonably.
- **Why did the review rules remove nothing (`removed_by_review` = 0)?** Both text models only propose
  edges from an earlier letter to a later one. Edges that follow one fixed order can't form a cycle or a
  two-way pair. The rules are still proven by unit tests: a 3-cycle is broken at its weakest edge,
  opposite directions keep the more confident one, and self-loops and non-direct claims are dropped.
- **Why micro-average edge scores?** Premises have different numbers of edges; micro-averaging weights
  every edge equally instead of every premise.
- **Why is AML's edge recall 1.0 but precision 0.54?** It accepts 120 of the 125 adjacent pairs, always in
  the alphabetical direction. That finds every true edge, plus many edges whose direction the premise
  can't decide.

## How to push

Wait until Abhijeet says **step 1 is pushed** and you have **accepted the GitHub collaborator invite**.
Windows: use **Git Bash**.

Extract `divyanshi.zip` in the folder where you'll clone, so `divyanshi/` sits next to `dl-aml/`.

```bash
git clone https://github.com/abhijeetnst/dl-aml.git
cd dl-aml
git config user.name "Divyanshi"
git config user.email "your-github-email@example.com"   # the email on YOUR GitHub account
cp -R ../divyanshi/1-aml-baseline/. .
git add c2c/aml.py tests/test_aml.py
git commit -m "feat: AML baseline with relation-crossed pattern features"
cp -R ../divyanshi/2-graph-stage/. .
git add c2c/graph.py tests/test_graph.py
git commit -m "feat: reviewed-edge DAG builder with provenance and edge metrics"
cp -R ../divyanshi/3-edge-review/. .
```

**Recommended before the next commit:** open `review/edge_review.csv`. For each of the 38 rows, read the
`evidence`, decide whether the edge is supported, and fill the empty `team_verdict` column with `ok` or
`wrong`. This is what makes it a real human review. Save as CSV.

```bash
git add review/edge_review.csv
git commit -m "docs: evidence-backed review of predicted causal edges"
git pull --rebase
git push
```

### Check (optional, large downloads)

```bash
python3 -m venv .venv && source .venv/bin/activate      # Windows Git Bash: source .venv/Scripts/activate
pip install -r requirements.txt
pytest -q tests/test_aml.py tests/test_graph.py
```
