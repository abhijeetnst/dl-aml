"""DL model: fine-tune DistilBERT on (premise, hypothesis) -> valid / not valid.
Train from the shell (long job):  python -m c2c.bert"""
import os

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import f1_score
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, Trainer,
                          TrainingArguments)
from transformers.trainer_utils import get_last_checkpoint

BASE = "distilbert-base-uncased"
MAX_LEN = 512
MODEL_DIR = "models/distilbert-c2c"


def encode(tok, premises, hypotheses, **kw):
    # cut the premise, never the hypothesis; long 6-variable premises exceed 512 tokens
    return tok(list(premises), list(hypotheses), truncation="only_first", max_length=MAX_LEN, **kw)


def to_dataset(tok, df: pd.DataFrame) -> Dataset:
    ds = Dataset.from_pandas(df[["premise", "hypothesis", "label"]].reset_index(drop=True))
    return ds.map(lambda b: encode(tok, b["premise"], b["hypothesis"]), batched=True,
                  remove_columns=["premise", "hypothesis"])


def train_bert(train_df, val_df, out_dir=MODEL_DIR, epochs=2, seed=42, max_steps=-1):
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForSequenceClassification.from_pretrained(BASE, num_labels=2)
    args = TrainingArguments(
        out_dir, num_train_epochs=epochs, max_steps=max_steps, learning_rate=5e-5, weight_decay=0.01,
        per_device_train_batch_size=16, per_device_eval_batch_size=32, eval_strategy="epoch",
        save_strategy="epoch", load_best_model_at_end=True, metric_for_best_model="f1", save_total_limit=1,
        seed=seed, report_to=[], logging_steps=100)
    trainer = Trainer(
        model=model, args=args, train_dataset=to_dataset(tok, train_df), eval_dataset=to_dataset(tok, val_df),
        data_collator=DataCollatorWithPadding(tok), processing_class=tok,
        compute_metrics=lambda p: {"f1": f1_score(p.label_ids, p.predictions.argmax(-1))})
    # an interrupted run picks up from its last epoch checkpoint; delete out_dir to retrain from scratch
    last = get_last_checkpoint(out_dir) if os.path.isdir(out_dir) else None
    trainer.train(resume_from_checkpoint=last)
    trainer.save_model(out_dir)


def predict_proba(df: pd.DataFrame, model_dir=MODEL_DIR, batch_size=32) -> np.ndarray:
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device).eval()
    out = []
    for i in range(0, len(df), batch_size):
        b = df.iloc[i:i + batch_size]
        enc = encode(tok, b.premise, b.hypothesis, padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            out.append(model(**enc).logits.softmax(-1)[:, 1].cpu().numpy())
    return np.concatenate(out)


if __name__ == "__main__":
    from c2c.data import load_splits, train_subset
    s = load_splits()
    train_bert(train_subset(s["train"]), s["validation"])
