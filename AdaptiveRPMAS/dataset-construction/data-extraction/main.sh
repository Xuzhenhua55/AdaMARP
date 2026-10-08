#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

python "${BASE_DIR}/dataset-construction/data-extraction/main.py" \
  --input "${BASE_DIR}/dataset-construction/data-extraction/source-books/books_example.jsonl" \
  --output_dir "${BASE_DIR}/dataset-construction/data-extraction" \
  --num_workers 1 \
  --model gpt-4o-mini \
  --max_books 100 \
  "$@"
