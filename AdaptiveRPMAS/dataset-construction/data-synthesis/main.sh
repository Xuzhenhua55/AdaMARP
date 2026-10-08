#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
# python synthesis.py --output sample_preview.jsonl --num_samples 1 --model openai:gpt-4o-mini

samples_dir="synthesis-data"
mkdir -p "${samples_dir}"

topics=(
  "Adventure"
  "Quest"
  "Rescue"
  "Battle"
  "Escape"
  "Exploration"
  "Mystery"
  "Investigation"
  "Crime-solving"
  "Puzzle-solving"
  "Conspiracy"
  "Romance"
  "Friendship"
  "Rivalry"
  "Betrayal"
  "Reconciliation"
  "Negotiation"
  "Strategy"
  "Magic"
  "Apocalypse"
)

num_threads=32
for topic in "${topics[@]}"; do
  output_file="${samples_dir}/samples_${topic}.jsonl"
  target_num=500

  # Count existing samples (by line count)
  if [ -f "${output_file}" ]; then
    existing_num=$(grep -cve '^[[:space:]]*$' "${output_file}" || true)
  else
    existing_num=0
  fi

  # If target number is reached or exceeded, skip
  if [ "${existing_num}" -ge "${target_num}" ]; then
    echo "Topic ${topic}: already has ${existing_num} samples (>= ${target_num}), skip."
    continue
  fi

  # Calculate how many more samples are needed
  remaining=$((target_num - existing_num))
  echo "Topic ${topic}: existing ${existing_num}, need ${remaining} more -> start synthesis."

  python main.py \
    --output "${output_file}" \
    --num_samples "${remaining}" \
    --model openai:gpt-4o-mini \
    --dialogue_topic "${topic}" \
    --num_threads "${num_threads}"
done
