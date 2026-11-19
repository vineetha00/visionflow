"""Run VisionFlow benchmarks on Modal (serverless GPU) — no queue, no Slurm.

    modal run scripts/modal_run.py --task accuracy            # n=150 DocVQA, CUDA rows, speculative twins
    modal run scripts/modal_run.py --task accuracy --source chartqa
    modal run scripts/modal_run.py --task bench                # vf bench, every backend/quant
    modal run scripts/modal_run.py --task bench --attn eager   # kernel A/B: run once per kernel
    modal run scripts/modal_run.py --task onnx                 # export + TensorRT/CUDA/CPU providers
    modal run scripts/modal_run.py --task all

One-time setup, done by you (these touch your account and I can't do them):

    pip install modal
    modal setup            # opens a browser; needs a Modal account with a card on file

Results persist in a Modal Volume and print to your terminal. Download them with:

    modal volume get visionflow-results / ./modal_results

Design decisions, each one a failure that happened on CARC:

- **The GPU is an L40S, not an H100.** Every existing number in the README (INT4 1.37s,
  TensorRT 11.90ms) is on an L40S. Same hardware keeps the dataset continuous; an
  H100 would be faster and incomparable. $1.95/hr against $3.95/hr, too.
- **The CUDA stack is pinned to what worked.** cu126 torch, onnxruntime-gpu 1.22,
  tensorrt-cu12 10.9.0.34, on a CUDA 12.6 devel base. Letting pip resolve freely
  gave +cu130 torch and TensorRT 11 on CARC, both of which fail *silently* -- ORT
  keeps listing the provider and runs on CPU. The harness catches that, but it
  is better not to produce it.
- **The repo is an editable install from a clone.** accuracy.py resolves dataset
  image paths against its own file location; installed into site-packages, those
  paths point nowhere. `pip install -e /root/visionflow` keeps them valid.
- **Weights live in a Volume, not the image.** Modal's own guidance, and it means
  the 4.5GB checkpoint downloads once rather than on every image rebuild.
- **flash-attn is a separate image layer.** Its build imports torch and needs the
  devel toolchain -- hence the nvidia/cuda devel base and --no-build-isolation. It
  can take 10-30 minutes and is cached afterwards. Being its own layer means a
  failed build only loses the FlashAttention bench row, not every task.
- **Functions cd into the repo and write to the Volume explicitly.** Background
  commits usually catch results, but an explicit commit before return is the
  difference between "usually" and "always".
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

import modal

APP_NAME = "visionflow"
REPO_URL = "https://github.com/vineetha00/visionflow"
REPO_DIR = "/root/visionflow"
GPU = "L40S"                       # matches every existing GPU number in the README
HF_CACHE = "/cache/hf"
RESULTS = "/results"
MINUTES = 60

app = modal.App(APP_NAME)

hf_cache = modal.Volume.from_name("visionflow-hf-cache", create_if_missing=True)
results_vol = modal.Volume.from_name("visionflow-results", create_if_missing=True)

# CUDA 12.6 devel base: the full toolkit is needed for flash-attn and for TensorRT's
# provider to find libcublas/libcudnn as system libraries rather than via pip paths.
base_image = (
    modal.Image.from_registry("nvidia/cuda:12.6.3-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .apt_install("git", "libgl1", "libglib2.0-0")
    .pip_install("packaging", "ninja", "wheel", "setuptools")
    # torch FIRST, from the cu126 index, so nothing downstream drags in +cu130.
    .pip_install(
        "torch==2.13.0", "torchvision==0.28.0",
        index_url="https://download.pytorch.org/whl/cu126",
    )
    .run_commands(
        f"git clone --depth 1 {REPO_URL} {REPO_DIR}",
        f"pip install -e '{REPO_DIR}[cuda,data,eval]'",
        # ORT 1.22 <-> TensorRT 10.9 <-> CUDA 12.x. Pinned together on purpose.
        "pip install 'onnxruntime-gpu<1.23' 'onnx>=1.16' 'onnxscript>=0.1' "
        "'tensorrt-cu12==10.9.0.34'",
    )
    .env({
        "HF_HOME": HF_CACHE,
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTHONUNBUFFERED": "1",
    })
)

# FlashAttention-2 as its own layer: a slow, fragile build that should only cost
# the row that needs it if it fails.
flash_image = base_image.pip_install(
    "flash-attn>=2.5", extra_options="--no-build-isolation",
)


def _pull_latest() -> str:
    """Images cache the clone; pull main so a run reflects the current repo."""
    subprocess.run(["git", "-C", REPO_DIR, "pull", "--ff-only", "-q"], check=False)
    rev = subprocess.run(["git", "-C", REPO_DIR, "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    return rev


def _preflight() -> None:
    """The four things that each silently failed at least once on CARC."""
    import torch

    print(f"torch {torch.__version__} | cuda: {torch.cuda.is_available()} | "
          f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no GPU'}")
    try:
        import bitsandbytes as bnb
        print(f"bitsandbytes {bnb.__version__}")
    except Exception as e:
        print(f"bitsandbytes MISSING ({e}) -- INT4/INT8 rows will be load failures")
    try:
        import onnxruntime as ort
        print(f"onnxruntime {ort.__version__} {ort.get_available_providers()}")
    except Exception as e:
        print(f"onnxruntime unavailable ({e})")
    try:
        import flash_attn  # noqa: F401
        print("flash-attn: present")
    except Exception:
        print("flash-attn: absent (engine will select SDPA and record it)")


def _run(cmd: list[str]) -> int:
    print("$", " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=REPO_DIR).returncode


def _save_results(tag: str) -> list[str]:
    """Copy benchmarks/results/*.{md,json} into the Volume under a stamped dir."""
    import shutil

    stamp = time.strftime("%Y%m%d-%H%M%S")
    dest = Path(RESULTS) / f"{stamp}-{tag}"
    dest.mkdir(parents=True, exist_ok=True)
    src = Path(REPO_DIR) / "benchmarks" / "results"
    copied = []
    for f in sorted(src.glob("*")):
        if f.is_file() and f.suffix in (".md", ".json"):
            shutil.copy2(f, dest / f.name)
            copied.append(f.name)
    results_vol.commit()
    print(f"\nsaved {len(copied)} file(s) to volume visionflow-results:{dest.relative_to(RESULTS)}")
    for name in copied:
        if name.endswith(".md"):
            print(f"\n--- {name} ---")
            print((dest / name).read_text())
    return copied


def _ensure_dataset(source: str, n: int) -> None:
    """Build the labeled set if its manifest OR images are missing."""
    probe = (
        "import sys; from visionflow.datasets import is_built; "
        f"sys.exit(0 if is_built('benchmarks/datasets/{source}_{n}/labeled_set.json') else 1)"
    )
    if subprocess.run(["python", "-c", probe], cwd=REPO_DIR).returncode != 0:
        _run(["vf", "dataset", source, "-n", str(n), "--seed", "0"])


COMMON = dict(
    gpu=GPU,
    volumes={HF_CACHE: hf_cache, RESULTS: results_vol},
    cpu=8,
    memory=32768,
)


@app.function(image=base_image, timeout=4 * 60 * MINUTES, **COMMON)
def accuracy(source: str = "docvqa", n: int = 150, speculative: bool = True,
             max_new_tokens: int = 128, only: str = "cuda") -> list[str]:
    rev = _pull_latest()
    print(f"repo @ {rev}")
    _preflight()
    _ensure_dataset(source, n)
    cmd = [
        "vf", "accuracy",
        "--labeled-set", f"benchmarks/datasets/{source}_{n}/labeled_set.json",
        "--max-new-tokens", str(max_new_tokens),
        "--out", f"benchmarks/results/accuracy_{source}{n}_cuda.json",
        "--markdown", f"benchmarks/results/accuracy_{source}{n}_cuda.md",
    ]
    if only:
        cmd += ["--only", only]
    if speculative:
        cmd.append("--speculative")
    _run(cmd)
    return _save_results(f"accuracy-{source}{n}")


@app.function(image=base_image, timeout=90 * MINUTES, **COMMON)
def bench(attn: str = "", repeats: int = 3, max_new_tokens: int = 256) -> list[str]:
    rev = _pull_latest()
    print(f"repo @ {rev}")
    _preflight()
    suffix = f"_{attn}" if attn else ""
    cmd = [
        "vf", "bench", "--repeats", str(repeats), "--max-new-tokens", str(max_new_tokens),
        "--out", f"benchmarks/results/bench_report_cuda{suffix}.json",
        "--markdown", f"benchmarks/results/bench_table_cuda{suffix}.md",
    ]
    if attn:
        cmd += ["--attn", attn]
    _run(cmd)
    return _save_results(f"bench{suffix}")


@app.function(image=flash_image, timeout=90 * MINUTES, **COMMON)
def bench_flash(repeats: int = 3, max_new_tokens: int = 256) -> list[str]:
    """Same as bench, on the image with flash-attn built. Auto-select picks it up."""
    rev = _pull_latest()
    print(f"repo @ {rev}")
    _preflight()
    _run([
        "vf", "bench", "--repeats", str(repeats), "--max-new-tokens", str(max_new_tokens),
        "--attn", "flash_attention_2",
        "--out", "benchmarks/results/bench_report_cuda_flash.json",
        "--markdown", "benchmarks/results/bench_table_cuda_flash.md",
    ])
    return _save_results("bench_flash")


@app.function(image=base_image, timeout=60 * MINUTES, **COMMON)
def onnx(runs: int = 50) -> list[str]:
    rev = _pull_latest()
    print(f"repo @ {rev}")
    _preflight()
    _run([
        "python", "-m", "visionflow.onnx_export", "--verify", "--benchmark",
        "--runs", str(runs),
        "--out", "benchmarks/results/onnx_report_cuda.json",
        "--markdown", "benchmarks/results/onnx_table_cuda.md",
    ])
    return _save_results("onnx")


@app.local_entrypoint()
def main(task: str = "accuracy", source: str = "docvqa", n: int = 150,
         attn: str = "", no_speculative: bool = False):
    """Dispatch. `--task all` runs accuracy on both datasets, bench, bench_flash, onnx."""
    tasks = {
        "accuracy": lambda: accuracy.remote(source=source, n=n, speculative=not no_speculative),
        "bench": lambda: bench.remote(attn=attn),
        "bench_flash": lambda: bench_flash.remote(),
        "onnx": lambda: onnx.remote(),
    }
    if task == "all":
        accuracy.remote(source="docvqa", n=n, speculative=not no_speculative)
        accuracy.remote(source="chartqa", n=n, speculative=not no_speculative)
        bench.remote()
        onnx.remote()
        try:
            bench_flash.remote()
        except Exception as e:  # a failed flash-attn build should not sink the rest
            print(f"bench_flash failed: {e}")
        return
    if task not in tasks:
        raise SystemExit(f"unknown --task {task!r}; choose from {sorted(tasks)} or 'all'")
    tasks[task]()
    print("\nDownload everything with:\n  modal volume get visionflow-results / ./modal_results")
