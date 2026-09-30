#!/usr/bin/env python3
"""Discovery + execution harness for cookbook examples (DX-1042).

Discovers two kinds of units and executes them by tier:

1. Standalone notebooks: every ``*.ipynb`` outside a manifest folder is a
   unit of kind ``notebook``. Default tier is ``weekly``; overrides (tier
   changes and exclusions, each with a reason) live in ``.ci/tiers.yaml``.
2. Manifest folders: a directory containing a ``ci.yaml`` manifest is a
   self-describing demo. The manifest declares its entrypoints (kind
   ``notebook``, ``script``, or ``app``), their tiers, and how to run or
   build them. Auto-discovery does not descend into manifest folders, so
   the manifest is authoritative for everything inside.

Execution per kind:
- ``notebook``: papermill if importable, else ``jupyter nbconvert --execute``.
- ``script``: the manifest ``command`` if given, else ``python <path>``.
- ``app``: each command in the manifest ``build`` list (build/typecheck at
  minimum; apps are never served in CI).

Failures on units listed in ``.ci/known_failures.yaml`` are reported as
warnings, not failures, so a flaky upstream doesn't page anyone.

Usage:
    python .ci/run_examples.py --list [--tier weekly|monthly|excluded|all]
    python .ci/run_examples.py --tier weekly
    python .ci/run_examples.py --tier all --only Thinking_Augmented
    python .ci/run_examples.py --tier weekly --dry-run

Requires: pyyaml. For notebooks: papermill + ipykernel (or jupyter/nbconvert).
``TOGETHER_BASE_URL`` is always unset for child processes so runs hit the
production API. ``TOGETHER_API_KEY`` must be set unless --list/--dry-run.
"""

from __future__ import annotations

import argparse
import dataclasses
import fnmatch
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
TIERS_FILE = REPO_ROOT / ".ci" / "tiers.yaml"
KNOWN_FAILURES_FILE = REPO_ROOT / ".ci" / "known_failures.yaml"
MANIFEST_NAME = "ci.yaml"

VALID_TIERS = ("weekly", "monthly", "excluded")
VALID_KINDS = ("notebook", "script", "app")

# Directories auto-discovery never descends into.
SKIP_DIRS = {
    ".git",
    ".ci",
    ".github",
    "archived",  # superseded notebooks (see archived/README.md); never executed
    ".ipynb_checkpoints",
    "node_modules",
    "images",
    "assets",
    "datasets",
    "data",
    "__pycache__",
    ".venv",
    "venv",
}

# Units run sequentially, so the default keeps the worst case (every unit
# hanging to timeout) inside the workflow's timeout-minutes. Slow-but-bounded
# units carry an explicit `timeout` in tiers.yaml or their manifest.
DEFAULT_TIMEOUT = 600  # seconds per unit


@dataclasses.dataclass
class Unit:
    path: str  # repo-relative, POSIX separators
    kind: str  # notebook | script | app
    tier: str  # weekly | monthly | excluded
    reason: str = ""  # required for excluded units
    timeout: int = DEFAULT_TIMEOUT
    command: str = ""  # script/app override
    build: list[str] = dataclasses.field(default_factory=list)  # app kind
    source: str = "auto"  # auto | manifest | tiers.yaml


def load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_tier_overrides() -> dict[str, dict]:
    data = load_yaml(TIERS_FILE)
    overrides = {}
    for entry in data.get("overrides", []):
        overrides[entry["path"]] = entry
    return overrides


def load_known_failures() -> dict[str, dict]:
    data = load_yaml(KNOWN_FAILURES_FILE)
    return {e["path"]: e for e in data.get("known_failures", [])}


def parse_manifest(folder: Path) -> list[Unit]:
    rel_folder = folder.relative_to(REPO_ROOT).as_posix()
    manifest = load_yaml(folder / MANIFEST_NAME)
    units = []
    for entry in manifest.get("entrypoints", []):
        kind = entry.get("kind", "notebook")
        tier = entry.get("tier", "weekly")
        if kind not in VALID_KINDS:
            raise ValueError(f"{rel_folder}/{MANIFEST_NAME}: bad kind {kind!r}")
        if tier not in VALID_TIERS:
            raise ValueError(f"{rel_folder}/{MANIFEST_NAME}: bad tier {tier!r}")
        units.append(
            Unit(
                path=f"{rel_folder}/{entry['path']}",
                kind=kind,
                tier=tier,
                reason=entry.get("reason", ""),
                timeout=int(entry.get("timeout", DEFAULT_TIMEOUT)),
                command=entry.get("command", ""),
                build=list(entry.get("build", [])),
                source="manifest",
            )
        )
    return units


def discover() -> list[Unit]:
    """Walk the repo: manifest folders are units; loose notebooks are units."""
    units: list[Unit] = []
    overrides = load_tier_overrides()

    def walk(folder: Path) -> None:
        if (folder / MANIFEST_NAME).exists() and folder != REPO_ROOT:
            units.extend(parse_manifest(folder))
            return  # manifest is authoritative; don't auto-discover inside
        for child in sorted(folder.iterdir()):
            if child.is_dir():
                if child.name in SKIP_DIRS or child.name.startswith("."):
                    continue
                walk(child)
            elif child.suffix == ".ipynb":
                rel = child.relative_to(REPO_ROOT).as_posix()
                override = overrides.get(rel, {})
                tier = override.get("tier", "weekly")
                if tier not in VALID_TIERS:
                    raise ValueError(f"tiers.yaml: bad tier {tier!r} for {rel}")
                units.append(
                    Unit(
                        path=rel,
                        kind="notebook",
                        tier=tier,
                        reason=override.get("reason", ""),
                        timeout=int(override.get("timeout", DEFAULT_TIMEOUT)),
                        source="tiers.yaml" if override else "auto",
                    )
                )

    walk(REPO_ROOT)

    # Warn about stale override entries pointing at deleted files.
    discovered = {u.path for u in units}
    for path in overrides:
        if path not in discovered:
            print(f"[warn] tiers.yaml entry has no matching notebook: {path}")
    return units


def run_cmd(cmd: list[str] | str, timeout: int, cwd: Path) -> tuple[int, str]:
    env = os.environ.copy()
    env.pop("TOGETHER_BASE_URL", None)  # always hit the production API
    shell = isinstance(cmd, str)
    try:
        proc = subprocess.run(
            cmd,
            shell=shell,
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        output = proc.stdout + proc.stderr
        return proc.returncode, output
    except subprocess.TimeoutExpired as exc:
        partial = (exc.stdout or "") + (exc.stderr or "")
        if isinstance(partial, bytes):
            partial = partial.decode(errors="replace")
        return 124, f"{partial}\n[timeout after {timeout}s]"


def notebook_runner() -> str:
    try:
        import papermill  # noqa: F401

        return "papermill"
    except ImportError:
        if shutil.which("jupyter"):
            return "nbconvert"
    raise SystemExit(
        "No notebook runner available: pip install papermill ipykernel "
        "(or jupyter nbconvert)."
    )


def execute(unit: Unit, runner: str) -> tuple[int, str]:
    abs_path = REPO_ROOT / unit.path
    cwd = abs_path.parent  # notebooks use relative asset paths
    if unit.kind == "notebook":
        with tempfile.TemporaryDirectory() as tmp:
            out_nb = Path(tmp) / abs_path.name
            if runner == "papermill":
                cmd = [
                    sys.executable,
                    "-m",
                    "papermill",
                    "--no-progress-bar",
                    str(abs_path),
                    str(out_nb),
                ]
            else:
                cmd = [
                    "jupyter",
                    "nbconvert",
                    "--to",
                    "notebook",
                    "--execute",
                    "--output",
                    str(out_nb),
                    str(abs_path),
                ]
            return run_cmd(cmd, unit.timeout, cwd)
    if unit.kind == "script":
        cmd = unit.command or f"{sys.executable} {abs_path.name}"
        return run_cmd(cmd, unit.timeout, cwd)
    if unit.kind == "app":
        # Build/typecheck only; apps are never served in CI.
        combined = ""
        for step in unit.build or [unit.command]:
            if not step:
                return 1, "app unit has no build commands"
            code, out = run_cmd(step, unit.timeout, cwd)
            combined += f"$ {step}\n{out}\n"
            if code != 0:
                return code, combined
        return 0, combined
    return 1, f"unknown kind {unit.kind!r}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tier",
        default="weekly",
        choices=["weekly", "monthly", "excluded", "all"],
        help="which tier to select (default: weekly). 'all' selects "
        "weekly+monthly; excluded units never run unless --tier excluded "
        "is combined with --list.",
    )
    parser.add_argument("--list", action="store_true", help="list units, don't run")
    parser.add_argument(
        "--dry-run", action="store_true", help="print what would run, don't run"
    )
    parser.add_argument(
        "--only",
        default="",
        help="only units whose path contains this substring (or glob)",
    )
    parser.add_argument(
        "--timeout", type=int, default=0, help="override per-unit timeout (seconds)"
    )
    args = parser.parse_args()

    units = discover()

    if args.tier == "all":
        selected = [u for u in units if u.tier in ("weekly", "monthly")]
    else:
        selected = [u for u in units if u.tier == args.tier]
    if args.only:
        pat = args.only if any(c in args.only for c in "*?[") else f"*{args.only}*"
        selected = [u for u in selected if fnmatch.fnmatch(u.path, pat)]

    if args.list:
        for u in selected:
            note = f"  # {u.reason}" if u.reason else ""
            print(f"{u.tier:8s} {u.kind:8s} {u.path}{note}")
        print(f"\n{len(selected)} unit(s) selected out of {len(units)} discovered")
        return 0

    if args.tier == "excluded":
        print("Refusing to execute the excluded tier; use --list to inspect it.")
        return 2

    if args.dry_run:
        for u in selected:
            print(f"[dry-run] would run {u.kind} {u.path} (timeout {u.timeout}s)")
        return 0

    if not os.environ.get("TOGETHER_API_KEY"):
        print("TOGETHER_API_KEY is not set; refusing to live-run examples.")
        return 2

    known = load_known_failures()
    runner = notebook_runner()
    if args.timeout:
        for u in selected:
            u.timeout = args.timeout

    passed, failed, warned = [], [], []
    for unit in selected:
        print(f"::group::{unit.kind} {unit.path}")
        code, output = execute(unit, runner)
        if code == 0:
            print("PASS")
            passed.append(unit)
        elif unit.path in known:
            entry = known[unit.path]
            print(output[-4000:])
            print(
                f"WARN (known failure since {entry.get('date', '?')}: "
                f"{entry.get('reason', 'no reason recorded')})"
            )
            warned.append(unit)
        else:
            print(output[-4000:])
            print("FAIL")
            failed.append(unit)
        print("::endgroup::")

    # Known-failure entries that now pass should be pruned from the allowlist.
    for unit in passed:
        if unit.path in known:
            print(f"[note] {unit.path} passes again; remove it from known_failures.yaml")

    print(
        f"\nTested: {len(selected)} | Passed: {len(passed)} | "
        f"Failed: {len(failed)} | Known-failure warnings: {len(warned)}"
    )
    for u in failed:
        print(f"  [FAIL] {u.path}")
    for u in warned:
        print(f"  [KNOWN] {u.path}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
