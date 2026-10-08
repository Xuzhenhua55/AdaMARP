#!/bin/bash

# Base path
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${BASE_DIR}/evaluation-framework"

# Batch generate configuration files for all models
python generate_all_model_configs_for_scene_managers.py \
  --jsonl "${BASE_DIR}/dataset_analysis/final_data/evaluation_data/AdaBenchEvaluate.jsonl" \
  --output-dir "${BASE_DIR}/evaluation-framework/evaluation-configs/evaluate-scene_manager"

python generate_all_model_configs_for_actor_models.py \
  --jsonl "${BASE_DIR}/dataset_analysis/final_data/evaluation_data/AdaBenchEvaluate.jsonl" \
  --output-dir "${BASE_DIR}/evaluation-framework/evaluation-configs/evaluate-actor_model"
