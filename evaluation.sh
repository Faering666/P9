# SPDX-License-Identifier: MIT
# Copyright (c) 2025 <Hugin J. Zachariasen, Magnus H. Jensen, Martin C. B. Nielsen, Tobias S. Madsen>.

#!/bin/bash
set -euo pipefail

run_eval() {
  local prediction_path=""
  local truth_path=""
  local data_type=""
  local model=""
  local result_path=""

  # Parse args
  while [[ $# -gt 0 ]]; do
    case "$1" in
      -p|--pred)
        [[ $# -ge 2 ]] || { echo "Error: $1 requires a value" >&2; return 2; }
        prediction_path="$2"; shift 2 ;;
      -gt|--truth)
        [[ $# -ge 2 ]] || { echo "Error: $1 requires a value" >&2; return 2; }
        truth_path="$2"; shift 2 ;;
      -ty|--type)
        [[ $# -ge 2 ]] || { echo "Error: $1 requires a value" >&2; return 2; }
        data_type="$2"; shift 2 ;;
      -m|--model)
        [[ $# -ge 2 ]] || { echo "Error: $1 requires a value" >&2; return 2; }
        model="$2"; shift 2 ;;
      -o|--out)
        [[ $# -ge 2 ]] || { echo "Error: $1 requires a value" >&2; return 2; }
        result_path="$2"; shift 2 ;;
      *)
        echo "Unknown argument: $1" >&2
        echo "Run: run_eval --help" >&2
        return 2
        ;;
    esac
  done

  # Validate required args
  if [[ -z "$prediction_path" || -z "$truth_path" || -z "$data_type" || -z "$model" || -z "$result_path" ]]; then
    echo "Error: Missing required arguments." >&2
    echo "Usage: run_eval -p <prediction_path> -t <truth_path> -d <data_type> -m <model_path> -o <result_path>" >&2
    return 2
  fi

  echo "=== Beginning predictions ==="
  python ./src/eval.py \
    --model "$model" \
    --data_path "$truth_path" \
    --data_type "$data_type" \
    --save_path "$prediction_path"

  echo "=== Beginning evaluation ==="
  python ./src/run_validation.py \
    --pred_path "$prediction_path" \
    --truth_path "$truth_path" \
    --type "$data_type" \
    --result_path "$result_path"
  echo "\n\n\n\n"
}

run_eval \
  --model "./checkpoints/hyggestue-19-12-2025/stage1_best_final.pth" \
  --pred "results/" \
  --truth "./data/Sri-Lanka-Aligned/" \
  --type "Sri-Lanka" \
  --out "my_results.json"