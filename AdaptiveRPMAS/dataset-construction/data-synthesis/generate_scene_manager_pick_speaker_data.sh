#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
export SCENE_MANAGER_NUM_WORKERS="${SCENE_MANAGER_NUM_WORKERS:-32}"
export USE_LLM_REASON="${USE_LLM_REASON:-1}"
export LLM_REASON_MODEL="${LLM_REASON_MODEL:-openai:gpt-4o-mini}"
exec "${PYTHON:-python}" "$SCRIPT_DIR/generate_scene_manager_pick_speaker_data.py" \
  --prompt-version "${1:-${PROMPT_VERSION:-enhance}}" \
  --output-path "${2:-${OUTPUT_PATH:-$BASE_DIR/dataset_analysis/final_data/training_data/AdaSMSet.jsonl}}"
