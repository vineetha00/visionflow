#!/bin/bash
#SBATCH --job-name=visionflow-gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a40:1               # 48GB, comfortably fits the 2.25B tier at fp16/INT8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# NOTE: `mkdir -p logs` before your first sbatch. Slurm opens these files before
# the script body runs, so the mkdir further down cannot create the directory in
# time -- the job starts with nowhere to write its output.
#
# VisionFlow GPU benchmark for USC CARC (Discovery).
#
# Fills the four rows this project cannot measure on Apple Silicon: bitsandbytes
# INT4/INT8 (CUDA-only kernels) and the CUDA / TensorRT ONNX execution providers.
#
# Submit with:   sbatch --account=<your_account> scripts/carc_gpu_benchmark.slurm
# Watch with:    squeue --me   /   tail -f logs/visionflow-gpu-<jobid>.out
#
# The account is passed at submit time rather than hardcoded here, so that
# `git pull` never conflicts with a local edit of this file (and so the repo
# carries nobody's allocation ID). Find yours with `myaccount`.
#
# Before submitting, run the one-time setup in the README section "GPU run on
# CARC" -- in particular, pre-download the model weights on the LOGIN node.
# Compute nodes may not have outbound internet, and this script sets
# HF_HUB_OFFLINE=1 so a missing cache fails immediately and loudly rather than
# hanging on a network timeout partway through a 4-hour allocation.

set -euo pipefail

# --- Scratch-backed caches -------------------------------------------------
# /home1 quota is small and the artifacts here are not: SmolVLM-2.25B is ~4.5GB
# and the exported ONNX graph carries a ~1.7GB external-data file.
export HF_HUB_OFFLINE=1

cd "${SLURM_SUBMIT_DIR:-$HOME/visionflow}"
mkdir -p logs benchmarks/results

source scripts/carc_env.sh
vf_env_report

python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" \
  || { echo "no GPU visible -- wrong partition or --gres?" >&2; exit 1; }

# --- 1. Latency / throughput / memory --------------------------------------
echo "=== vf bench ==="
vf bench --repeats 3 --max-new-tokens 256 \
  --out benchmarks/results/bench_report_cuda.json \
  --markdown benchmarks/results/bench_table_cuda.md

# --- 2. Accuracy per quantization level ------------------------------------
# The run that matters. On MPS, fp16 cost 71 points of exact field accuracy
# versus fp32 on identical weights. INT4/INT8 are more aggressive than fp16, so
# whether NF4 holds up better than plain fp16 is genuinely unknown. INT4 and
# INT8 are forced to load separately here, so one cannot answer for the other.
echo "=== vf accuracy ==="
vf accuracy --max-new-tokens 384 \
  --out benchmarks/results/accuracy_report_cuda.json \
  --markdown benchmarks/results/accuracy_table_cuda.md

# --- 3. ONNX export + execution providers ----------------------------------
# First TensorRT run spends ~10 minutes building an engine. That is expected,
# not a hang. Check the output table: if TensorRT is missing, ORT falls back to
# CUDA or CPU *silently*, and the harness flags that row rather than letting a
# CPU number be published as a TensorRT one.
echo "=== onnx export + providers ==="
python -m visionflow.onnx_export --verify --benchmark --runs 50 \
  --out benchmarks/results/onnx_report_cuda.json \
  --markdown benchmarks/results/onnx_table_cuda.md

echo
echo "=== results ==="
for f in benchmarks/results/*_cuda.md; do echo; echo "--- $f"; cat "$f"; done
echo
echo "Done. Commit benchmarks/results/ and push, or copy the *_cuda.md tables back."
