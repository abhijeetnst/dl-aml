---
title: Corr2Cause Causal Inference
emoji: 🔗
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: 6.32.0
app_file: app.py
python_version: "3.12"
pinned: false
license: mit
---

# Corr2Cause causal inference

Two tabs:
- **Corr2Cause**: paste a premise (correlations and independences) and a causal hypothesis. A pattern-feature
  logistic regression (AML), a fine-tuned DistilBERT and a PC-algorithm oracle each say whether the premise
  entails it; the PC skeleton is drawn with its v-structures.
- **Free text**: plain English causal sentences become a DAG; pick X and Y for an interventional
  ATE = P(Y=1 | do(X=1)) − P(Y=1 | do(X=0)) under a noisy-OR network.

| Test set, positive-class F1 | AML | DistilBERT | Symbolic oracle |
|---|---|---|---|
| original letters | 0.617 | 0.605 | **0.871** |
| variables renamed | 0.279 | 0.264 | **0.871** |
| DAG edge F1 | 0.703 | 0.610 | **0.942** |

The text models learn the dataset's alphabetical-order shortcut; the oracle does not depend on letter names.

Code, report and tests: https://github.com/abhijeetnst/dl-aml
