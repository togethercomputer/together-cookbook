# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "deepagents>=0.6,<0.8",
#     "langchain-together",
# ]
# ///
"""Ralph loop on Together AI's Nemotron 3 Ultra.

The "Ralph" pattern (named after Geoff Huntley's `while :; do cat PROMPT.md |
agent; done`) runs an agent on the SAME task over and over, with a FRESH
context window every iteration. The agent's only memory is the filesystem:
each iteration it re-orients by reading what previous iterations left behind,
makes progress, saves everything to files, and exits. Because context never
accumulates, the loop can run indefinitely without summarization or
compaction.

This version uses Deep Agents (`create_deep_agent`) directly with
`nvidia/nemotron-3-ultra-550b-a55b` served on Together AI. Files persist in a
local work directory; the agent is jailed to it via
`FilesystemBackend(virtual_mode=True)`.

Usage:
    export TOGETHER_API_KEY=...
    uv run ralph_loop.py "Build a beginner's Python course as markdown files"
    uv run ralph_loop.py "..." --iterations 10 --work-dir ./my-project

The agent signals completion by writing DONE.md to the workspace root, which
stops the loop early.
"""

import argparse
import sys
from pathlib import Path

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend

DEFAULT_MODEL = "together:nvidia/nemotron-3-ultra-550b-a55b"

SYSTEM_PROMPT = """\
You are an autonomous builder agent running inside a "Ralph loop": you are \
invoked repeatedly on the same task, and each invocation starts with a fresh \
context window. The filesystem is your ONLY memory between iterations.

Rules:
- ALWAYS start by listing the workspace (ls) and reading what already exists, \
especially PROGRESS.md. Never redo finished work.
- Make concrete, incremental progress this iteration, then stop. Do not try \
to finish everything in one pass.
- Save ALL work to files. Anything not written to a file is lost when this \
iteration ends.
- Keep a PROGRESS.md up to date: what is done, what is in flight, and what \
the next iteration should do first.
- When the task is FULLY complete, write DONE.md summarizing what was built. \
Only do this when there is genuinely nothing left to do.
"""

ITERATION_PROMPT = """\
## Ralph Iteration {n}{total}

You are in a fresh session with no memory of previous iterations. Your \
previous work (if any) is in the filesystem. Check what exists and keep \
building.

TASK:
{task}

Make progress and save it to files. You'll be called again.
"""


def build_agent(model: str, work_dir: Path):
    return create_deep_agent(
        model=model,
        system_prompt=SYSTEM_PROMPT,
        backend=FilesystemBackend(root_dir=str(work_dir), virtual_mode=True),
    )


def run_iteration(agent, task: str, n: int, max_iterations: int) -> str:
    total = f"/{max_iterations}" if max_iterations else ""
    prompt = ITERATION_PROMPT.format(n=n, total=total, task=task)
    result = agent.invoke({"messages": prompt})
    # Nemotron occasionally ends a tool-heavy turn with empty content, so
    # report the last non-empty assistant text from the run instead.
    for msg in reversed(result["messages"]):
        if msg.type == "ai" and isinstance(msg.content, str) and msg.content.strip():
            return msg.content
    return "(iteration produced no text summary — progress is in the files)"


def print_workspace(work_dir: Path) -> None:
    print("\nWorkspace contents:")
    for path in sorted(work_dir.rglob("*")):
        if path.is_file() and ".git" not in path.parts:
            print(f"  {path.relative_to(work_dir)}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a Deep Agent in a Ralph loop on Together AI."
    )
    parser.add_argument("task", help="The task to work on every iteration")
    parser.add_argument(
        "--iterations",
        type=int,
        default=3,
        help="Number of iterations (0 = run until DONE.md or Ctrl+C; default 3)",
    )
    parser.add_argument(
        "--work-dir",
        default="./ralph_workspace",
        help="Persistent workspace directory (default ./ralph_workspace)",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"LangChain model string (default {DEFAULT_MODEL})",
    )
    args = parser.parse_args()

    work_dir = Path(args.work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    agent = build_agent(args.model, work_dir)

    print(f"Ralph loop | model={args.model}")
    print(f"Workspace: {work_dir}\n")

    n = 1
    try:
        while args.iterations == 0 or n <= args.iterations:
            print(f"=== Iteration {n} ===")
            try:
                answer = run_iteration(agent, args.task, n, args.iterations)
            except Exception as exc:  # keep looping on transient API errors
                print(f"  iteration failed ({exc}); continuing")
                n += 1
                continue
            preview = answer if len(answer) <= 1500 else answer[:1500] + " ..."
            print(preview + "\n")
            if (work_dir / "DONE.md").exists():
                print("DONE.md found — task complete, stopping loop.")
                break
            n += 1
    except KeyboardInterrupt:
        print(f"\nStopped during iteration {n}.")

    print_workspace(work_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
