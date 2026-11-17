# Neuro-Symbolic Causal Inference on Corr2Cause (DL/AML Project 1)

[![tests](https://github.com/abhijeetnst/dl-aml/actions/workflows/tests.yml/badge.svg)](https://github.com/abhijeetnst/dl-aml/actions/workflows/tests.yml)

Live app: https://corr2cause-dl-aml.streamlit.app (Streamlit Community Cloud)

Text → candidate causal triples (AML baseline vs DistilBERT) → reviewed DAG per premise →
pgmpy Bayesian network → intervention queries. A PC-algorithm oracle is the symbolic reference.

## Architecture
```
Corr2Cause row: premise (correlations + independences) + hypothesis ("A directly causes C.")
  │
  │  c2c/data.py    load, leakage-safe splits, strict parsers, letter shuffling
  ▼
candidate causal triple (relation, X, Y)  ── scored by three systems ──┐
  ├─ c2c/aml.py     AML baseline: pattern features → logistic regression
  ├─ c2c/bert.py    DL model: fine-tuned DistilBERT (premise, hypothesis) → P(valid)
  └─ c2c/oracle.py  symbolic reference: PC skeleton → equivalence class → entailment
  ▼  P(valid) per triple, one threshold per system tuned on validation
c2c/graph.py   review gate → one DAG per premise (with provenance) → edge precision/recall
  ▼
c2c/bn.py      gold DAG + noisy-OR CPDs → 5,000 simulated rows → fit each system's DAG (pgmpy MLE)
               → ATE = P(Y=1 | do(X=1)) − P(Y=1 | do(X=0)), compared with the gold network
  ▼
project1_causal.py / .ipynb  runs every stage → results/*.csv, results/dag_comparison.png

c2c/text.py    free English text → (cause, verb, effect) triples → DAG
c2c/demo.py             app logic (no UI): the three systems on a Corr2Cause row or a CSV batch; free text → DAG → ATE
demo/streamlit_app.py   Streamlit UI over c2c/demo.py (Streamlit Community Cloud entrypoint)
```
Each `c2c/` module has its own test file in `tests/`; `tests/test_demo.py` also runs the Streamlit app.

## Reproduce
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest tests -q                             # 43 tests; 2 skip without the local model
python -m c2c.bert                          # ~1.5 h on an Apple M3 Pro (MPS); resumes if interrupted
jupytext --to ipynb project1_causal.py
jupyter nbconvert --to notebook --execute --inplace project1_causal.ipynb
```

Edit `project1_causal.py`, then regenerate the `.ipynb`; never hand-edit the notebook file.

## Using the app
Three tabs:
- **Corr2Cause**: pick a test-set example or type a premise and hypothesis in the Corr2Cause template, then Check.
  AML, DistilBERT and the PC oracle each give P(valid) and a verdict; the PC pattern is drawn. Rename variables
  shuffles the letters to show the text models' alphabetical-order shortcut.
- **Batch**: upload a CSV with columns `premise`, `hypothesis` and an optional 0/1 `label`
  (a sample CSV can be downloaded from the tab). Every row is scored by all three systems; rows that do not parse get a
  message in the `error` column. With labels, precision, recall and F1 per system are shown. Results download as CSV.
- **Free text**: English sentences with causal verbs become a graph; pick X and Y for the ATE of do(X) on Y.

Limits: premise up to 6,000 characters and 6 variables; free text up to 5,000 characters and 50 causal statements;
batch files up to 500 rows and 2 MB. Privacy: inputs are processed in memory and not stored or logged; errors are logged
with their type and traceback only.

## Run the app locally
```bash
python scripts/export_artifacts.py   # optional: local artifacts/; otherwise they download from the Hub
streamlit run demo/streamlit_app.py  # http://localhost:8501
python scripts/screenshots.py        # with the app running: browser check + docs/img/app_*.png (needs playwright)
```
The model is read from `models/distilbert-c2c` if present, else from `abhijeetnst/distilbert-c2c` on the Hub
(override with `MODEL_REPO`).

## Deploy
Model and artifacts go to the Hub model repo `abhijeetnst/distilbert-c2c`:
```bash
hf auth login
bash scripts/upload_model.sh
```
App: on share.streamlit.io choose New app → repository `abhijeetnst/dl-aml`, branch `main`, main file
`demo/streamlit_app.py`, Python 3.12 (under Advanced settings). It installs `demo/requirements.txt` (CPU-only torch).
Theme, the 2 MB upload cap and disabled usage stats are in `.streamlit/config.toml`.

Keep-alive: Community Cloud puts apps to sleep after a period without traffic. `.github/workflows/keepalive.yml` runs
every 6 hours (and on demand): `scripts/keepalive.py` opens the app in headless Chromium, clicks
"Yes, get this app back up!" if it is asleep, and fails unless the app's title appears. GitHub pauses scheduled
workflows after 60 days without repository activity; re-enable it from the Actions tab if that happens.

## Free-text extraction
`c2c/text.py` parses English sentences with spaCy (`en_core_web_sm`) and keeps verbs from a fixed causal list
("causes", "leads to", "raises", "prevents", "is caused by", ...). Subject and object become cause and effect; noun
phrases are lemmatised so "floods" and "flood" are one node. Triples become a DAG with the same review rules as
`c2c/graph.py`: one direction per pair (the earlier sentence wins), and a cycle loses the edge from the latest
sentence. The app then puts a noisy-OR network on that DAG and reports the ATE for any pair.

Limits:
- Pattern-based and English only; causal verbs outside the list are ignored.
- One cause and one effect per verb; "X and Y cause Z" keeps only the first noun of each side.
- The small spaCy model misparses some verbs (e.g. "results in" yields no triple).
- Noisy-OR assumes every edge raises its effect; "reduces" or "prevents" is treated as a positive cause.

## Results
Triple extraction (test / renamed test), graph edges and intervention queries: see
`results/extraction_metrics.csv`, `results/graph_metrics.csv`, `results/intervention_queries.csv`
and notebook sections 2, 4 and 5.

```
         set           system relation  precision  recall    f1    n
        test AML (pattern LR)      all      0.596   0.639 0.617 1162
        test       DistilBERT      all      0.479   0.822 0.605 1162
        test  Symbolic oracle      all      0.920   0.828 0.871 1162
test_renamed AML (pattern LR)      all      0.279   0.278 0.279 1162
test_renamed       DistilBERT      all      0.224   0.322 0.264 1162
test_renamed  Symbolic oracle      all      0.920   0.828 0.871 1162
         set           system  edge_precision  edge_recall  edge_f1  tp  fp  fn
        test AML (pattern LR)           0.542        1.000    0.703  65  55   0
        test       DistilBERT           0.505        0.769    0.610  50  49  15
        test  Symbolic oracle           0.890        1.000    0.942  65   8   0
test_renamed AML (pattern LR)           0.200        0.369    0.259  24  96  41
test_renamed       DistilBERT           0.161        0.138    0.149   9  47  56
test_renamed  Symbolic oracle           0.890        1.000    0.942  65   8   0
```

Mean |ATE error| over 76 intervention queries per set (5 premises): test AML 0.005, DistilBERT 0.063, oracle 0.005;
renamed test AML 0.238, DistilBERT 0.189, oracle 0.004.

## Review sheets
`review/edge_review.csv` and `review/label_audit_review.csv` hold a verdict and the premise evidence for each row
(the independence statement or collider that decides it). The `draft_verdict` column was derived from that evidence
with the oracle's equivalence-class reasoning, so it is not independent of the oracle. `team_verdict` is blank until
the team fills it per `review/REVIEW_CHECKLIST.md`; only then can the report call it a human review.
The notebook never writes to `review/`, so re-running it keeps these files.

## Data caveats we found
- The official splits leak: 3 validation premises also occur in test, 2 in train. We removed them from validation/train.
- The "B directly causes A" template is never positive: causes are ordered alphabetically, a shortcut.
  The renamed-variable test set removes it.
- 37% of training pairs exceed 512 tokens; DistilBERT sees a truncated premise.
- The symbolic oracle disagrees with 44 of 1,162 test labels; `review/label_audit_review.csv` has our verdicts
  for the first 10.

## Team
Team T043, Section D.

| Member | Contribution |
|---|---|
| Abhijeet | Project setup and codebase structure (`c2c/` package layout, pinned environment, test setup), data layer (`c2c/data.py`: loading, leakage-safe splits, parsers, letter shuffling), end-to-end notebook and results, free-text extraction (`c2c/text.py`), Streamlit app and deploy (`c2c/demo.py`, `demo/`, `scripts/`), CI, presentation and report |
| Prashant Pandey | Symbolic oracle (`c2c/oracle.py`: PC algorithm, Markov equivalence class, entailment) and the label audit |
| Divyanshi Khanka | AML baseline (`c2c/aml.py`), graph stage (`c2c/graph.py`: review gate, DAG per premise, edge metrics) and the edge review |
| Shubham Jain | DistilBERT fine-tuning and inference (`c2c/bert.py`), resumable training |
| Rahul Kumar | Bayesian networks and intervention queries (`c2c/bn.py`: noisy-OR gold BN, pgmpy fitting, ATE) |
