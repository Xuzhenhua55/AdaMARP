#!/bin/bash
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${BASE_DIR}/evaluation-framework"

python generate_evaluation_configs.py \
  --jsonl "${BASE_DIR}/dataset_analysis/final_data/evaluation_data/AdaBenchEvaluate.jsonl" \
  --output "${BASE_DIR}/evaluation-framework/evaluation-configs/gpt-4o-mini-Enhance-Enhance.jsonl"
