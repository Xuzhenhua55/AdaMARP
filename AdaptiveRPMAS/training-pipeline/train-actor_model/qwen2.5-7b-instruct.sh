#!/usr/bin/env bash
set -euo pipefail
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-$HOME/.cache/modelscope}"
export MEGATRON_LM_PATH="${MEGATRON_LM_PATH:?Set MEGATRON_LM_PATH to the installed Megatron-LM directory}"
export MASTER_PORT=29520

# Set BASE_DIR to project root
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NPROC_PER_NODE=2 \
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1} \
megatron sft \
    --model "${MODEL_PATH:?Set MODEL_PATH to a local actor model}" \
    --load_safetensors true \
    --save_safetensors true \
    --dataset "${DATASET_PATH:-${BASE_DIR}/dataset_analysis/final_data/training_data/AdaRPSet.jsonl}" \
    --split_dataset_ratio 0.05 \
    --tensor_model_parallel_size 2 \
    --pipeline_model_parallel_size 1 \
    --context_parallel_size 1 \
    --sequence_parallel true \
    --micro_batch_size 24 \
    --global_batch_size 48 \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --finetune true \
    --cross_entropy_loss_fusion true \
    --lr 1e-6 \
    --lr_warmup_fraction 0.05 \
    --min_lr 1e-7 \
    --max_epochs 8 \
    --save "${OUTPUT_PATH:-${BASE_DIR}/output/Qwen2.5-7B-Instruct/role_playing}" \
    --save_interval 50 \
    --max_length 16384 \
    --truncation_strategy left \
    --num_workers 8 \
    --no_save_optim true \
    --no_save_rng true \
    --dataset_num_proc 256 \
    --model_author ${MODEL_AUTHOR:-your-name} \
    --model_name ${MODEL_NAME:-qwen2.5-7b-instruct-adaptiveRPMAS-role_playing}

# tensor_model_parallel_size
# load_safetensors
# save_safetensors
