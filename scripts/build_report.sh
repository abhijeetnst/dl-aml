#!/usr/bin/env bash
# Render docs/report/report.html to docs/T043_Project1_Report.pdf with headless Chrome, then set the PDF metadata.
# Usage: bash scripts/build_report.sh    (CHROME and PYTHON override the defaults; metadata step needs pypdf)
set -euo pipefail
cd "$(dirname "$0")/.."

CHROME="${CHROME:-/Applications/Google Chrome.app/Contents/MacOS/Google Chrome}"
PYTHON="${PYTHON:-.venv/bin/python}"
SRC="docs/report/report.html"
OUT="docs/T043_Project1_Report.pdf"

"$CHROME" --headless=new --disable-gpu --no-pdf-header-footer --run-all-compositor-stages-before-draw \
  --print-to-pdf="$PWD/$OUT" "file://$PWD/$SRC" 2>/dev/null

"$PYTHON" - "$OUT" <<'EOF'
import sys
from pypdf import PdfReader, PdfWriter

path = sys.argv[1]
writer = PdfWriter(clone_from=PdfReader(path))
writer.add_metadata({"/Title": "Neuro-Symbolic Causal Inference on Corr2Cause", "/Author": "Team T043",
                     "/Subject": "DL/AML Project 1 report", "/Creator": "Team T043", "/Producer": "Team T043"})
with open(path, "wb") as f:
    writer.write(f)
print(f"{path}: {len(writer.pages)} pages")
EOF
