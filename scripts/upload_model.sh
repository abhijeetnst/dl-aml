#!/usr/bin/env bash
# Publish the DistilBERT model and app artifacts to a Hub model repo. Run `hf auth login` first.
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL_REPO=${HF_USER:-abhijeetnst}/distilbert-c2c

hf repos create "$MODEL_REPO" --exist-ok
hf upload "$MODEL_REPO" models/distilbert-c2c . \
  --include model.safetensors --include config.json --include tokenizer.json --include tokenizer_config.json \
  --include special_tokens_map.json --include vocab.txt
hf upload "$MODEL_REPO" artifacts . --include aml.joblib --include thresholds.json
