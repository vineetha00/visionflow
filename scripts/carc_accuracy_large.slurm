#!/bin/bash
#SBATCH --job-name=visionflow-acc
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=vineetha@usc.edu
#
# Large-n accuracy sweep on real document/chart data.
#
# The hand-built set in benchmarks/labeled_set.json has 21 fields. At that size
# one field is 4.8 percentage points, and the CI columns show the consequence:
# only one accuracy comparison in the whole README clears its intervals. This
# job re-runs the same comparisons at n=150 so the rest can either be confirmed
# or dropped.
#
# GPU is what makes it affordable: INT4 runs ~3.4s/sample against fp32-on-CPU's
# ~92s, so a sweep that would take days on CPU takes hours here.
#
# Submit with:
#   mkdir -p logs
#   sbatch --account=<your_account> scripts/carc_accuracy_large.slurm
#
# See scripts/carc_gpu_benchmark.slurm for the one-time environment setup; this
# script assumes the same venv and cached weights.

set -euo pipefail

: "${VISIONFLOW_N:=150}"

cd "${SLURM_SUBMIT_DIR:-$HOME/visionflow}"
mkdir -p logs benchmarks/results
source scripts/carc_env.sh
vf_env_report

python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" \
  || { echo "no GPU visible -- wrong partition or --gres?" >&2; exit 1; }

# Dataset build needs the network, so it is NOT run under HF_HUB_OFFLINE. If
# compute nodes are firewalled, run this one command on the login node first --
# it is deterministic given --seed, so the job will find the same set already
# built and skip straight to evaluation.
# Test the manifest AND its images, not just the manifest. The labeled_set.json
# files are committed while the images are gitignored, so on a fresh clone the
# JSON is present and the PNGs are not -- checking only the JSON makes this skip
# the build and then fail on missing image files deep into the allocation.
for SOURCE in docvqa chartqa; do
  MANIFEST="benchmarks/datasets/${SOURCE}_${VISIONFLOW_N}/labeled_set.json"
  if python -c "import sys; from visionflow.datasets import is_built; sys.exit(0 if is_built('$MANIFEST') else 1)"; then
    echo "=== ${SOURCE} (n=${VISIONFLOW_N}) already built ==="
  else
    echo "=== building ${SOURCE} (n=${VISIONFLOW_N}) ==="
    vf dataset "$SOURCE" -n "$VISIONFLOW_N" --seed 0
  fi
done

# CUDA rows only. The CPU/fp32 full-precision reference is the scientifically
# interesting comparison but costs ~92s/sample -- at n=150 that is ~4 hours for
# that row alone, which would not fit alongside everything else. Run it
# separately with --only cpu and a longer wall time if the fp32-vs-quantized
# gap is what you are after.
for SOURCE in docvqa chartqa; do
  echo
  echo "=== vf accuracy: ${SOURCE} n=${VISIONFLOW_N} (CUDA rows) ==="
  vf accuracy \
    --labeled-set "benchmarks/datasets/${SOURCE}_${VISIONFLOW_N}/labeled_set.json" \
    --only cuda \
    --max-new-tokens 128 \
    --out "benchmarks/results/accuracy_${SOURCE}${VISIONFLOW_N}_cuda.json" \
    --markdown "benchmarks/results/accuracy_${SOURCE}${VISIONFLOW_N}_cuda.md"
done

echo
echo "=== results ==="
for f in benchmarks/results/accuracy_*"${VISIONFLOW_N}"_cuda.md; do
  echo; echo "--- $f"; cat "$f"
done
