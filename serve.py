"""Iteration 0: serve GPT-OSS-20B with vLLM and no derived compile cache."""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import sys
from typing import Sequence


MODEL_ID = "openai/gpt-oss-20b"
MODEL_REVISION = "6cee5e81ee83917806bbde320786a8fb61efebee"
SERVED_MODEL_NAME = "gpt-oss-20b"


def build_command(args: argparse.Namespace) -> list[str]:
    command = [
        args.vllm_executable,
        "serve",
        MODEL_ID,
        "--revision",
        MODEL_REVISION,
        "--served-model-name",
        SERVED_MODEL_NAME,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--tensor-parallel-size",
        "1",
        "--max-model-len",
        "8192",
        "--gpu-memory-utilization",
        "0.9",
    ]
    command.extend(args.vllm_args)
    return command


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--vllm-executable", default="vllm")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("vllm_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.vllm_args[:1] == ["--"]:
        args.vllm_args = args.vllm_args[1:]
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    command = build_command(args)
    print(shlex.join(command), flush=True)
    if args.dry_run:
        return 0

    executable = shutil.which(args.vllm_executable)
    if executable is None:
        print("vllm not found; run `uv sync --extra server`", file=sys.stderr)
        return 2

    command[0] = executable
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)
    environment["VLLM_DISABLE_COMPILE_CACHE"] = "1"
    environment["VLLM_NO_USAGE_STATS"] = "1"
    environment["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    os.execvpe(executable, command, environment)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
