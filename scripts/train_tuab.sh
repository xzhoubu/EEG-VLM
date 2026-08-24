#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

ADAPTATION="${ADAPTATION:-lora}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"
COMMON=(--data_dir "${EEG_VLM_DATA_DIR:-${ROOT_DIR}/data}/TUAB" --split_mode cross-sub --window_size 10 --qwen_model_size 4b --single_image_kind combined --image_size 896 --seed 42)

if [[ "${ADAPTATION}" == "lora" ]]; then
  accelerate launch --num_processes "${NUM_PROCESSES}" src/tuab_vlm.py train_lora "${COMMON[@]}" --lora_rank 8 --lora_alpha 16 --lora_dropout 0.1
elif [[ "${ADAPTATION}" == "head" ]]; then
  accelerate launch --num_processes "${NUM_PROCESSES}" src/tuab_vlm_cls.py train_head "${COMMON[@]}"
else
  echo "ADAPTATION must be lora or head" >&2
  exit 2
fi
