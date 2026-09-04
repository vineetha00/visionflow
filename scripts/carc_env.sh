#!/bin/bash
# Shared environment setup for VisionFlow jobs on USC CARC (Discovery).
# Source this from a Slurm script, or from an interactive shell:
#
#   source scripts/carc_env.sh
#
# Every setting here exists because something failed without it. The comments
# record which failure, since all of them were silent or misleading.

# --- Caches ----------------------------------------------------------------
# /home1 quota will not hold a 4.5GB checkpoint plus a ~1.7GB ONNX external-data
# file. /scratch1 will.
export HF_HOME="${HF_HOME:-/scratch1/$USER/hf}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

# --- Modules ---------------------------------------------------------------
# CUDA is hidden behind a compiler in CARC's Lmod hierarchy: a bare
# `module load cuda` fails with "exist but cannot be loaded as requested".
# gcc/12.3.0 + cuda/12.4.1 is the pairing that resolves on Discovery. Find
# another with:
#   for g in $(module -t avail gcc | grep '^gcc/'); do
#     module purge; module load "$g" cuda/<ver> && echo "$g"; done
#
# Only the CUDA *major* version must match the torch build -- CUDA 12.x has
# minor-version compatibility, so this 12.4.1 module runs a +cu126 torch fine.
# The major version genuinely matters: pip's default torch wheel here resolves
# to +cu130, and onnxruntime-gpu 1.22 wants CUDA 12 (libcublas.so.12). Install
# torch from https://download.pytorch.org/whl/cu126 to keep them aligned.
#
# No `|| true`. An earlier version tolerated module-load failures, so a wrong
# version name was swallowed and the job ran to completion on CPU, producing
# plausible numbers that were not GPU numbers.
: "${VISIONFLOW_MODULES:=gcc/12.3.0 cuda/12.4.1 conda}"
module purge
module load ${VISIONFLOW_MODULES}

# --- Virtualenv ------------------------------------------------------------
source "${VISIONFLOW_VENV:-$HOME/visionflow/.venv}/bin/activate"

# --- CUDA libraries for ONNX Runtime ---------------------------------------
# ORT's CUDA and TensorRT providers need libcublas/libcudnn on the loader path.
# CARC's cuda module does not put them there, so the provider .so fails to load
# and ORT falls back to CPU -- while get_available_providers() keeps listing the
# provider, which is what made this take several rounds to diagnose. torch's
# +cu126 wheel already bundles exactly those libraries as nvidia-* pip packages.
_vf_libs="$(python - <<'PY' 2>/dev/null || true
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
[ -n "$_vf_libs" ] && export LD_LIBRARY_PATH="${_vf_libs}:${LD_LIBRARY_PATH:-}"

# TensorRT's own libnvinfer, from the tensorrt-cu12 wheel if it is installed.
_vf_trt="$(python -c 'import os,tensorrt_libs; print(os.path.dirname(tensorrt_libs.__file__))' 2>/dev/null || true)"
[ -n "$_vf_trt" ] && export LD_LIBRARY_PATH="${_vf_trt}:${LD_LIBRARY_PATH:-}"
unset _vf_libs _vf_trt

# --- Report ----------------------------------------------------------------
vf_env_report() {
  echo "=== environment ==="
  hostname
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null \
    || echo "no GPU visible (login node, or missing --gres)"
  python -c "import torch; print('torch', torch.__version__, '| cuda avail:', torch.cuda.is_available())"
  python -c "import bitsandbytes; print('bitsandbytes', bitsandbytes.__version__)" \
    2>/dev/null || echo "bitsandbytes MISSING -- INT4/INT8 rows will be load failures"
  python -c "import onnxruntime as ort; print('onnxruntime', ort.__version__, ort.get_available_providers())" \
    2>/dev/null || echo "onnxruntime not installed"
  echo
}
