"""Command-line entry point: `visionflow <image> --prompt "..." [--mode text|json|kv]`"""

from __future__ import annotations

import argparse
import json
import sys

from .pipeline import VisionFlow


USAGE_EXTRA = """
Subcommands:
  bench     Benchmark every backend/quantization on this machine (`vf bench --help`)
  accuracy  Measure extraction accuracy per quantization level (`vf accuracy --help`)
"""


def main():
    # Subcommands are dispatched before argparse so that the original positional
    # form (`visionflow image.png --prompt "..."`) keeps working unchanged.
    argv = sys.argv[1:]
    if argv and argv[0] == "bench":
        from .bench import main as bench_main

        return bench_main(argv[1:])
    if argv and argv[0] == "accuracy":
        from .accuracy import main as accuracy_main

        return accuracy_main(argv[1:])

    parser = argparse.ArgumentParser(
        prog="visionflow",
        description="Local VLM image -> structured output",
        epilog=USAGE_EXTRA,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("image", help="Path to an image file")
    parser.add_argument("--prompt", required=True, help="What to extract or ask about the image")
    parser.add_argument("--mode", choices=["text", "json", "kv"], default="text")
    parser.add_argument("--fields", nargs="*", help="Field names for --mode kv")
    parser.add_argument("--model", default=None, help="Override the HF model id")
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference")
    args = parser.parse_args()

    kwargs = {"force_cpu": args.cpu}
    if args.model:
        kwargs["model_id"] = args.model
    vf = VisionFlow(**kwargs)

    if args.mode == "text":
        print(vf.text(args.image, args.prompt))
    elif args.mode == "json":
        result = vf.json(args.image, args.prompt)
        print(json.dumps(result.parsed if result.ok else {"error": result.error}, indent=2))
        if not result.ok:
            sys.exit(1)
    else:
        if not args.fields:
            parser.error("--mode kv requires --fields")
        print(json.dumps(vf.key_value(args.image, args.prompt, args.fields), indent=2))


if __name__ == "__main__":
    main()
