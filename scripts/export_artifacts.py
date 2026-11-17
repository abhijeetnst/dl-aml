"""Export the trained AML model and validation-tuned thresholds for the demo app.
Run from the repo root:  python scripts/export_artifacts.py"""
import json
import pathlib
import sys

import joblib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from c2c import aml, bert  # noqa: E402
from c2c.data import load_splits, train_subset  # noqa: E402
from c2c.graph import best_threshold  # noqa: E402

OUT = pathlib.Path("artifacts")

if __name__ == "__main__":
    s = load_splits()
    val = s["validation"]
    model = aml.train_aml(train_subset(s["train"]))  # same rows and seed as project1_causal.py
    thresholds = {"aml": best_threshold(val.label, aml.predict_proba(model, val)),
                  "bert": best_threshold(val.label, bert.predict_proba(val))}
    OUT.mkdir(exist_ok=True)
    joblib.dump(model, OUT / "aml.joblib")
    (OUT / "thresholds.json").write_text(json.dumps(thresholds, indent=2))
    print(thresholds)
