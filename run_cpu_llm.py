#!/usr/bin/env python3
"""
CPU-only inference helper that wraps build/bin/llama-cli for quick tests.

Usage (after activating the llama-cpu conda env and landing on a CPU node):
    python run_cpu_llm.py --prompt "Say hello in one sentence." --n-predict 32

Pass --use-python if you explicitly want to try the python llama_cpp bindings;
the default path sticks to the CLI to avoid hardware-specific binary issues.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Quick CPU inference using llama.cpp")
    prompt_group = parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument("--prompt", help="Inline prompt text.")
    prompt_group.add_argument("--prompt-file", type=Path, help="Read prompt from file.")

    parser.add_argument(
        "--n-predict",
        type=int,
        default=None,
        help="Limit tokens generated (omit to let the model stop naturally).",
    )
    parser.add_argument("--threads", type=int, help="CPU threads (default: SLURM_CPUS_PER_TASK or os.cpu_count).")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size (default 64).")
    parser.add_argument("--ctx-size", type=int, default=4096, help="Context window (default 4096).")
    parser.add_argument("--model", type=Path, help="Override GGUF model path.")
    parser.add_argument("--verbose", action="store_true", help="Show full llama-cli output.")
    parser.add_argument(
        "--use-python",
        action="store_true",
        help="Attempt to use python llama_cpp bindings instead of the CLI.",
    )
    parser.add_argument(
        "--extra-args",
        nargs=argparse.REMAINDER,
        help="Extra flags passed to llama-cli after '--'.",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream tokens to stdout as they generate (CLI backend; python backend uses streaming iterator).",
    )
    parser.add_argument("--temperature", type=float, default=0.8, help="Temperature (python backend only).")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p (python backend only).")
    parser.add_argument("--seed", type=int, help="Random seed (python backend only).")
    parser.add_argument("--grammar-file", type=Path, help="Path to a GBNF grammar file (CLI backend).")
    parser.add_argument(
        "--stop",
        action="append",
        default=[],
        help="Add a stop sequence for generation (CLI backend). Repeatable.",
    )
    return parser


def resolve_paths() -> tuple[Path, Path]:
    script_dir = Path(__file__).resolve().parent
    llama_dir = script_dir / "llm" / "llama.cpp"
    if not llama_dir.exists():
        raise SystemExit(f"Expected llama.cpp checkout at {llama_dir}.")
    return script_dir, llama_dir


def read_prompt(args: argparse.Namespace) -> str:
    if args.prompt is not None:
        return args.prompt
    try:
        return args.prompt_file.read_text(encoding="utf-8")
    except OSError as exc:  # pragma: no cover - user input
        raise SystemExit(f"Failed to read prompt file: {exc}") from exc


def select_model(args: argparse.Namespace, llama_dir: Path) -> Path:
    if args.model:
        model_path = Path(args.model)
    else:
        model_path = llama_dir / "models" / "mistral-7b-instruct" / "mistral-7b-instruct-v0.2.Q4_K_M.gguf"
    if not model_path.is_file():
        raise SystemExit(f"Model not found: {model_path}")
    return model_path


def python_backend(args: argparse.Namespace, model_path: Path, threads: int) -> None:
    try:
        from llama_cpp import Llama  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise SystemExit("llama-cpp-python not installed in this environment. Use --use-cli.") from exc

    if args.verbose:
        print("Using python llama_cpp bindings", file=sys.stderr)
        print(f"Model: {model_path}", file=sys.stderr)
        print(f"Threads: {threads}, Batch: {args.batch_size}, Ctx: {args.ctx_size}", file=sys.stderr)

    llm = Llama(
        model_path=str(model_path),
        n_ctx=args.ctx_size,
        n_batch=args.batch_size,
        n_threads=threads,
        seed=args.seed or -1,
        verbose=args.verbose,
    )
    prompt_text = read_prompt(args)

    max_tokens = args.n_predict if args.n_predict is not None else max(1, args.ctx_size - 1)

    if args.stream:
        stream = llm(
            prompt=prompt_text,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=max_tokens,
            stream=True,
        )
        for chunk in stream:
            piece = chunk["choices"][0]["text"]
            if piece:
                print(piece, end="", flush=True)
        print()
        return

    completion = llm(
        prompt=prompt_text,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=max_tokens,
    )
    text = completion["choices"][0]["text"]
    print(text.strip())


def cli_backend(args: argparse.Namespace, llama_dir: Path, model_path: Path, threads: int) -> int:
    prompt_text = read_prompt(args)
    prompt_lines = {
        line.strip()
        for line in prompt_text.splitlines()
        if line.strip()
    }

    llama_cli = llama_dir / "build" / "bin" / "llama-cli"
    if not llama_cli.is_file():
        raise SystemExit(f"Could not find llama-cli at {llama_cli}. Rebuild llama.cpp with -DLLAMA_BUILD_EXAMPLES=ON.")

    cmd: List[str] = [
        str(llama_cli),
        "--model",
        str(model_path),
        "--threads",
        str(threads),
        "--batch-size",
        str(args.batch_size),
        "--ctx-size",
        str(args.ctx_size),
        "--no-conversation",
        "--simple-io",
        "--prompt",
        prompt_text,
    ]

    predict_limit = args.n_predict if args.n_predict is not None else max(1, args.ctx_size - 1)
    cmd.extend(["--n-predict", str(predict_limit)])

    # Map select sampling/options for CLI backend when provided.
    if args.temperature is not None:
        cmd.extend(["--temp", str(args.temperature)])
    if args.top_p is not None:
        cmd.extend(["--top-p", str(args.top_p)])
    if args.grammar_file:
        cmd.extend(["--grammar-file", str(args.grammar_file)])
    if args.stop:
        for stop_token in args.stop:
            cmd.extend(["--stop", stop_token])

    if args.extra_args:
        cmd.append("--")
        cmd.extend(args.extra_args)

    if args.verbose:
        print("Running:", " ".join(cmd), file=sys.stderr)

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(threads)
    env.setdefault("LLAMA_LOG_LEVEL", "error")

    if args.stream:
        process = subprocess.Popen(
            cmd,
            cwd=llama_dir,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            env=env,
            bufsize=1,
        )
        if not process.stdout:
            return process.wait()
        try:
            printing = False
            printed_any = False
            stop_print = False
            buffer = ""
            debug_prefixes = (
                "build:",
                "main:",
                "llama_model_loader:",
                "print_info:",
                "load:",
                "llama_context:",
                "llama_kv_cache:",
                "common_init_from_params:",
                "system_info:",
                "sampler",
                "generate:",
                "llama_perf",
                "llama_memory",
            )
            while True:
                chunk = process.stdout.read(1024)
                if chunk == "":
                    break
                buffer += chunk
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    if not printing:
                        if "generate:" in line:
                            printing = True
                        continue
                    if stop_print:
                        continue
                    stripped = line.strip()
                    if not printed_any and stripped == "":
                        continue
                    if stripped in prompt_lines:
                        continue
                    if any(stripped.startswith(prefix) for prefix in debug_prefixes):
                        stop_print = True
                        continue
                    print(line, end="\n", flush=True)
                    printed_any = True
            if printing and not stop_print and buffer:
                stripped = buffer.strip()
                if stripped and stripped not in prompt_lines and stripped != prompt_text.strip() and not any(
                    stripped.startswith(prefix) for prefix in debug_prefixes
                ):
                    print(buffer, end="", flush=True)
                    printed_any = True
        finally:
            process.stdout.close()
        ret = process.wait()
        if printed_any:
            print()
        return ret

    result = subprocess.run(
        cmd,
        cwd=llama_dir,
        text=True,
        capture_output=not args.verbose,
        stdin=subprocess.DEVNULL,
        env=env,
    )

    if args.verbose:
        return result.returncode

    if result.returncode != 0:
        sys.stderr.write(result.stderr or "llama-cli failed without stderr output.\n")
        return result.returncode

    combined = (result.stdout or "") + ("\n" + result.stderr if result.stderr else "")
    lines = []
    debug_prefixes = (
        "build:",
        "main:",
        "llama_",
        "print_info:",
        "load:",
        "load_tensors:",
        "llama_context:",
        "llama_kv_cache:",
        "common_init_from_params:",
        "system_info:",
        "sampler",
        "generate:",
        "llama_perf",
        "llama_memory",
    )
    for raw in combined.splitlines():
        stripped = raw.strip()
        if not stripped:
            continue
        if set(stripped) == {"."}:
            continue
        if stripped == prompt_text.strip() or stripped in prompt_lines:
            continue
        if any(stripped.startswith(prefix) for prefix in debug_prefixes):
            continue
        if "=" in stripped and any(ch.isdigit() for ch in stripped):
            continue
        lines.append(stripped)

    if lines:
        print("\n".join(lines))
    else:
        # Fallback to raw stdout in case filtering removed everything
        if combined.strip():
            print(combined.strip())
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    _, llama_dir = resolve_paths()
    model_path = select_model(args, llama_dir)

    threads = args.threads or int(os.environ.get("SLURM_CPUS_PER_TASK") or os.cpu_count() or 1)

    if args.use_python:
        python_backend(args, model_path, threads)
        return 0

    return cli_backend(args, llama_dir, model_path, threads)


if __name__ == "__main__":
    raise SystemExit(main())
