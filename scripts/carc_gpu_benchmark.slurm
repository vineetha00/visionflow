#!/bin/bash
#SBATCH --job-name=visionflow-gpu
#SBATCH --account=CHANGE_ME            # <-- your PI/project account (see `myaccount`)
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1              # a40 / v100 also fine; >=16GB VRAM keeps the 2.25B tier
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# VisionFlow GPU benchmark for USC CARC (Discovery).
#
# Fills the four rows this project cannot measure on Apple Silicon: bitsandbytes
# INT4/INT8 (CUDA-only kernels) and the CUDA / TensorRT ONNX execution providers.
#
# Submit with:   sbatch scripts/carc_gpu_benchmark.slurm
# Watch with:    squeue --me   /   tail -f logs/visionflow-gpu-<jobid>.out
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
export HF_HOME="${HF_HOME:-/scratch1/$USER/hf}"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

mkdir -p logs benchmarks/results

# --- Environment -----------------------------------------------------------
# Module names and versions drift on CARC; check `module avail cuda` and
# `module avail conda` and adjust. The CUDA module must be loaded before the
# bitsandbytes and onnxruntime-gpu imports, or they fall back to CPU silently.
module purge
module load gcc/11.3.0 || true
module load cuda/11.8.0 || module load cuda || true
module load conda || true

source "${VISIONFLOW_VENV:-$HOME/visionflow/.venv}/bin/activate"

echo "=== host ==="
hostname
nvidia-smi || { echo "no GPU visible -- wrong partition or --gres?" >&2; exit 1; }
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
python -c "import bitsandbytes; print('bitsandbytes', bitsandbytes.__version__)" || echo "WARNING: bitsandbytes missing -- INT4/INT8 rows will record a load failure"
python -c "import onnxruntime as ort; print('ORT providers:', ort.get_available_providers())" || true
echo

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
