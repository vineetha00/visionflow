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
export HF_HOME="${HF_HOME:-/scratch1/$USER/hf}"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

mkdir -p logs benchmarks/results

# ONNX Runtime's CUDA and TensorRT providers need libcublas/libcudnn on the
# loader path, and CARC's cuda module does not put them there -- the provider
# .so fails to load, ORT falls back to CPU, and `get_available_providers()`
# still lists the provider the whole time. torch's +cu12x wheel already bundles
# exactly these libraries as nvidia-* pip packages, so point the loader at them.
# Appended after venv activation below (needs the venv's python to locate them).
_add_torch_cuda_libs() {
  local libs
  libs="$(python - <<'PY' 2>/dev/null || true
import os
try:
    import nvidia
except ImportError:
    raise SystemExit
p = os.path.dirname(nvidia.__file__)
print(":".join(os.path.join(p, d, "lib") for d in os.listdir(p)
                if os.path.isdir(os.path.join(p, d, "lib"))))
PY
)"
  [ -n "$libs" ] && export LD_LIBRARY_PATH="${libs}:${LD_LIBRARY_PATH:-}"
  # TensorRT ships its own libnvinfer via the tensorrt-cu12 wheel, if installed.
  local trt
  trt="$(python -c 'import os,tensorrt_libs; print(os.path.dirname(tensorrt_libs.__file__))' 2>/dev/null || true)"
  [ -n "$trt" ] && export LD_LIBRARY_PATH="${trt}:${LD_LIBRARY_PATH:-}"
}

# --- Environment -----------------------------------------------------------
# CUDA is hidden behind a compiler in CARC's Lmod hierarchy -- `module load cuda`
# alone fails with "exist but cannot be loaded as requested". This pairing was
# found on Discovery with:
#   for g in $(module -t avail gcc | grep '^gcc/'); do
#     module purge; module load "$g" cuda/12.4.1 && echo "$g"; done
# Override without editing this file: export VISIONFLOW_MODULES="gcc/... cuda/..."
#
# CUDA 12.x deliberately, not 11.8: current torch and onnxruntime-gpu wheels are
# built against CUDA 12 (ORT >=1.19 needs CUDA 12 + cuDNN 9), and pairing a cu12
# torch with an 11.8 module leaves bitsandbytes unable to find a runtime -- which
# surfaces as INT4/INT8 load-failure rows, i.e. exactly the measurement this job
# exists to produce.
#
# No `|| true` here on purpose. An earlier version tolerated module-load failures,
# which meant a wrong version name was swallowed and the job ran to completion on
# CPU -- producing plausible numbers that were not GPU numbers. Under `set -e` a
# bad module name now kills the job in the first seconds instead.
# Only the CUDA *major* version has to match the torch build. CUDA 12.x provides
# minor-version compatibility, so this 12.4.1 module runs a +cu126 torch happily;
# each cuda module sits under a different compiler in the hierarchy and 12.6.3 is
# not reachable from gcc/12.3.0, so there is nothing to gain by chasing an exact
# minor match.
#
# The major version does matter. PyPI's default torch wheel on this stack was
# +cu130, which had to be replaced: onnxruntime-gpu targets CUDA 12 + cuDNN 9 and
# would not have found CUDA 13 libraries, so the TensorRT/CUDA provider rows would
# have come back unavailable -- and bitsandbytes CUDA 13 support is immature,
# which is the other half of what this job measures. Install torch from
# https://download.pytorch.org/whl/cu126 and keep this module on CUDA 12.
: "${VISIONFLOW_MODULES:=gcc/12.3.0 cuda/12.4.1 conda}"
module purge
module load ${VISIONFLOW_MODULES}
module list 2>&1

source "${VISIONFLOW_VENV:-$HOME/visionflow/.venv}/bin/activate"
_add_torch_cuda_libs

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
