#!/usr/bin/env bash
set -euo pipefail
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-$HOME/.cache/modelscope}"
export MEGATRON_LM_PATH="${MEGATRON_LM_PATH:?Set MEGATRON_LM_PATH to the installed Megatron-LM directory}"
export MASTER_PORT=29501

# Set BASE_DIR to project root
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NPROC_PER_NODE=4 \
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3} \
megatron sft \
    --model "${CHECKPOINT_PATH:?Set CHECKPOINT_PATH to a local scene-manager checkpoint}" \
    --load_safetensors true \
    --save_safetensors true \
    --dataset "${DATASET_PATH:-${BASE_DIR}/dataset_analysis/final_data/training_data/AdaSMSet.jsonl}" \
    --split_dataset_ratio 0.05 \
    --tensor_model_parallel_size 4 \
    --pipeline_model_parallel_size 1 \
    --context_parallel_size 1 \
    --sequence_parallel true \
    --micro_batch_size 24 \
    --global_batch_size 48 \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --finetune false \
    --cross_entropy_loss_fusion true \
    --lr 1e-6 \
    --lr_warmup_fraction 0.05 \
    --min_lr 1e-7 \
    --max_epochs 8 \
    --save "${OUTPUT_PATH:-${BASE_DIR}/output/Qwen2.5-7B-Instruct/scene_manager/enhanced}" \
    --save_interval 50 \
    --max_length 16384 \
    --truncation_strategy left \
    --num_workers 8 \
    --no_save_optim false \
    --no_save_rng false \
    --dataset_num_proc 256 \
    --model_author ${MODEL_AUTHOR:-your-name} \
    --model_name ${MODEL_NAME:-qwen2.5-7b-instruct-adaptiveRPMAS-scene_manager-enhanced}

# tensor_model_parallel_size
# load_safetensors
# save_safetensors
