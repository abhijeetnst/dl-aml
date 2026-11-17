#!/usr/bin/env bash
# Publish the model + artifacts to a Hub model repo and the demo to a Gradio Space. Run `hf auth login` first.
set -euo pipefail
cd "$(dirname "$0")/.."

HF_USER=${HF_USER:-abhijeetnst}
MODEL_REPO=$HF_USER/distilbert-c2c
SPACE=$HF_USER/corr2cause-demo

hf repos create "$MODEL_REPO" --exist-ok
hf upload "$MODEL_REPO" models/distilbert-c2c . \
  --include model.safetensors --include config.json --include tokenizer.json --include tokenizer_config.json \
  --include special_tokens_map.json --include vocab.txt
hf upload "$MODEL_REPO" artifacts . --include aml.joblib --include thresholds.json

hf repos create "$SPACE" --repo-type space --space-sdk gradio --exist-ok
hf spaces variables add "$SPACE" -e "MODEL_REPO=$MODEL_REPO"

stage=$(mktemp -d)
trap 'rm -rf "$stage"' EXIT
cp app.py "$stage/"
cp -R c2c "$stage/"
cp space/README.md space/requirements.txt "$stage/"
hf upload "$SPACE" "$stage" . --repo-type space --exclude "**/__pycache__/**" --exclude "*.pyc"

echo "https://huggingface.co/spaces/$SPACE"
