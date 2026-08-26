#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

manifest="experiments/capability_locked_manifest.json"
results_dir="results/capability"
log_dir="tmp/capability_logs"
mkdir -p "$results_dir" "$log_dir"

benchmarks=(mmlu_pro gsm8k ifeval hellaswag)
conditions=(baseline recirculation)

for benchmark in "${benchmarks[@]}"; do
  for condition in "${conditions[@]}"; do
    output="$results_dir/${benchmark}_${condition}.json"
    log="$log_dir/locked_${benchmark}_${condition}.log"
    if [[ -e "$output" ]]; then
      echo "Preserving existing result: $output"
      continue
    fi
    echo "Running $benchmark / $condition"
    HF_HUB_DISABLE_PROGRESS_BARS=1 uv run python -m recirculation.run_capability \
      --condition "$condition" \
      --benchmark "$benchmark" \
      --manifest "$manifest" \
      --output "$output" >"$log" 2>&1
  done
done

uv run python -m recirculation.summarize_capability \
  --manifest "$manifest" \
  --results-dir "$results_dir" \
  --output "$results_dir/comparison.json"
