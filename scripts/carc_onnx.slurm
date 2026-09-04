#!/bin/bash
#SBATCH --job-name=visionflow-onnx
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=vineetha@usc.edu
#
# ONNX export + execution-provider benchmark only.
#
# Split out from carc_gpu_benchmark.slurm so the one outstanding measurement can
# be collected without re-running the latency and accuracy sweeps, which are
# already done. Short queue, short wall time.
#
#   mkdir -p logs
#   sbatch --account=<your_account> scripts/carc_onnx.slurm
#
# The first TensorRT session spends ~10 minutes building an engine with no
# output. That is expected, not a hang.
#
# Read the provider table carefully. ORT falls back silently: a provider can be
# listed by get_available_providers() and still fail at session creation, after
# which ORT quietly runs on CPU. The harness reads provider_used back from the
# live session and flags any mismatch, so a row saying "this is NOT a
# TensorrtExecutionProvider measurement" is CPU latency wearing a TensorRT label.

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$HOME/visionflow}"
mkdir -p logs benchmarks/results
source scripts/carc_env.sh
vf_env_report

python -m visionflow.onnx_export --verify --benchmark --runs 50 \
  --out benchmarks/results/onnx_report_cuda.json \
  --markdown benchmarks/results/onnx_table_cuda.md

echo
echo "=== results ==="
cat benchmarks/results/onnx_table_cuda.md
