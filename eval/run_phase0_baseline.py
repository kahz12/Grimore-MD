"""Plan by default; --execute builds the Phase-0 baseline in an isolated workspace.

The workspace receives a copy of exactly the twelve files the Phase-0 datasets
declare (corpus_version phase0-test-vault-2026-09-22), verified by SHA-256, plus
the given grimore.toml and an index of its own. The reference vault, the
developer's database and earlier reports are never written. Each step runs in
a fresh `grimore` process from the workspace. Full-model steps use
--checkpoint, so relaunching with the same workspace resumes an interrupted
run; a report with generation failures is set aside and its failed cases are
asked again. Human review of the exported queues is still required.
"""
# ruff: noqa: E402 -- direct execution must find the repository package
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from grimore.utils.config import load_config

SOURCE_VAULT = REPO_ROOT / "test_vault" / "test_vault"
# The corpus the Phase-0 datasets were validated against. A changed file is a
# different corpus: re-pin deliberately (and say so) instead of measuring it.
CORPUS_SHA256 = {
    "acta_revision_humedal.md": "c439acb2c109028aa275f2f9f582828206c9feabfce63f4e7ab3bf3771441972",
    "ai_evolution.md": "71bff23c898e1a3c79a3f52864bc549012953947883e607589a91e50fa0a0fc1",
    "apple_chemistry.md": "727b0841a0fa144e93e8a73e1296d183c7ea7fae197359596146a9086ddf3a51",
    "arquitectura_de_software.md": "3e2fc753df656c4cbc061f30da11b076828a2f988983be16ce95079a3d5e5568",
    "astronomia.md": "aae8ace3b46dcd804cb4dd8d5aa9638a697058b9df8aaa757ca5a5412825fe0d",
    "biografia_socrates.md": "3f5856a8125e65d0b0b8694c3e7eb91bd9e45565f27f4500645e7e9aafda6566",
    "biologia_ranas.md": "8d98578c8a58d6d2348814f7efc68ccf32c47926449535e860bef53115962009",
    "gothic_architecture.txt": "4153beb20f8115f90b67c2883bbf1655300a47486446b528c49fcf5cc04682c5",
    "informe_humedal_la_esperanza.pdf": "6e8eae1f1df2eb110f9a92db53240e165bfa8f2dddcbae5063d49a9fc387e34d",
    "kotlin_fundamentals.md": "5e68c84fce319f6c016aa4a585692defb4763a080d43e6b53e24d79c59aa43ba",
    "roman_empire.md": "98c05b49210c7c29fbc4b61d940da8bccf32e18a856ab81af8ffb3b6d7cdf443",
    "social_engineering_defense.md": "b6b115ee435f9f8e1dee5efbb984a76ce28195c5c2178c1fe410e37f51f9ed1c",
}
MARKER = "phase0-workspace.json"
SPLITS = ("tuning", "evaluation")
STEPS = ("scan", "tuning_retrieval", "evaluation_retrieval", "tuning_full", "evaluation_full")


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def corpus_problems(source: Path) -> list[str]:
    problems = []
    for name, expected in CORPUS_SHA256.items():
        path = source / name
        if not path.is_file():
            problems.append(f"missing: {name}")
        elif sha256(path) != expected:
            problems.append(f"changed: {name}")
    return problems


def config_problems(config: Path) -> list[str]:
    """The copied config must keep the vault and index inside the workspace."""
    settings = load_config(str(config))
    problems = []
    if Path(settings.vault.path).as_posix().removeprefix("./").rstrip("/") != "test_vault":
        problems.append('[vault] path must be "./test_vault"')
    if Path(settings.memory.db_path).is_absolute():
        problems.append("[memory] db_path must be relative to the workspace")
    return problems


def step_argv(step: str) -> list[str]:
    if step == "scan":
        return ["scan", "--no-dry-run"]
    split, mode = step.split("_")
    argv = ["eval", "--golden", str(REPO_ROOT / f"eval/phase0_{split}.yaml"), "--json",
            "--export", f"reports/{step}.json", "--history", "reports/history.jsonl"]
    if mode == "retrieval":
        argv.append("--retrieval-only")
    else:
        argv += ["--checkpoint", f"reports/{step}.checkpoint.jsonl"]
    return argv


def plan(config: Path, workspace: Path | None) -> dict:
    config = config.resolve()
    problems = corpus_problems(SOURCE_VAULT) + config_problems(config)
    settings = load_config(str(config))
    return {
        "version": 1, "status": "planned-not-executed" if not problems else "blocked",
        "problems": problems, "workspace": str(workspace) if workspace else None,
        "config": str(config), "config_sha256": sha256(config),
        "model_llm_local": settings.cognition.model_llm_local,
        "model_embeddings_local": settings.cognition.model_embeddings_local,
        "corpus_version": "phase0-test-vault-2026-09-22", "corpus_sha256": CORPUS_SHA256,
        "steps": [{"id": step, "argv": ["grimore", *step_argv(step)]} for step in STEPS],
        "limitations": [
            "Default action only prepares a plan; no models, files or processes.",
            "Twelve controlled documents: retrieval scores near the ceiling.",
            "Judge metrics are directional; human claim/abstention review is pending.",
            "Latency belongs to this machine; compare it only on the same hardware.",
        ],
    }


def prepare(schedule: dict, workspace: Path) -> None:
    """Create the workspace once; a later launch must find the same inputs."""
    marker = workspace / MARKER
    if marker.exists():
        recorded = json.loads(marker.read_text(encoding="utf-8"))
        if (recorded["config_sha256"] != sha256(workspace / "grimore.toml")
                or recorded["corpus_sha256"] != CORPUS_SHA256):
            raise ValueError(f"{workspace} was prepared with other inputs; use a new workspace")
        return
    if workspace.exists() and any(workspace.iterdir()):
        raise ValueError(f"{workspace} exists and is not a Phase-0 workspace")
    vault = workspace / "test_vault" / "test_vault"
    vault.mkdir(parents=True)
    (workspace / "reports").mkdir()
    for name in CORPUS_SHA256:
        shutil.copy2(SOURCE_VAULT / name, vault / name)
    shutil.copy2(schedule["config"], workspace / "grimore.toml")
    marker.write_text(json.dumps({
        "prepared_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config_sha256": schedule["config_sha256"], "corpus_sha256": CORPUS_SHA256,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _finished(workspace: Path, step: str) -> bool:
    if step == "scan":
        return (workspace / "reports/scan.done").exists()
    report = workspace / f"reports/{step}.json"
    if not report.exists():
        return False
    summary = json.loads(report.read_text(encoding="utf-8"))
    indexed = summary["provenance"]["corpus"]["notes"]
    if indexed != len(CORPUS_SHA256):
        raise ValueError(f"{step} ran against {indexed} indexed notes, not {len(CORPUS_SHA256)}; "
                         "check reports/scan.out before measuring")
    if not summary["aggregate"].get("generation_failure_n", 0):
        return True
    # Contaminated: keep it for the record, rerun so the checkpoint retries.
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report.rename(report.with_name(f"{step}.contaminated-{stamp}.json"))
    return False


def _run(workspace: Path, step: str) -> dict:
    print(f"Running {step}", file=sys.stderr, flush=True)
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with (workspace / f"reports/{step}.out").open("a", encoding="utf-8") as out, \
            (workspace / f"reports/{step}.log").open("a", encoding="utf-8") as log:
        result = subprocess.run([sys.executable, "-m", "grimore", *step_argv(step)],
                                cwd=workspace, env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
                                stdin=subprocess.DEVNULL, stdout=out, stderr=log)
    attempt = {"id": step, "started": started, "exit_code": result.returncode,
               "finished": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    if result.returncode == 0 and step == "scan":
        (workspace / "reports/scan.done").write_text(attempt["finished"] + "\n", encoding="utf-8")
    attempt["status"] = "completed" if result.returncode == 0 and _finished(workspace, step) else "incomplete"
    return attempt


def execute(schedule: dict, workspace: Path, *, retries: int = 2) -> dict:
    """Run the missing steps in order. An eval step that ends with generation
    failures is relaunched up to ``retries`` times; its checkpoint means only
    the failed cases are asked again."""
    if schedule["problems"]:
        raise ValueError("; ".join(schedule["problems"]))
    workspace = workspace.resolve()
    prepare(schedule, workspace)
    attempts = []
    for step in STEPS:
        if _finished(workspace, step):
            attempts.append({"id": step, "status": "already-completed"})
            continue
        for _ in range(1 + (retries if step != "scan" else 0)):
            attempts.append(_run(workspace, step))
            if attempts[-1]["status"] == "completed":
                break
        if attempts[-1]["status"] != "completed":
            break
    done = {a["id"] for a in attempts if a["status"] != "incomplete"}
    complete = done == set(STEPS)
    return {"version": 1, "status": "completed-measurements" if complete else "incomplete-measurements",
            "workspace": str(workspace), "attempts": attempts,
            "human_review": "pending", "resume": "rerun the same command to continue"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "grimore.toml")
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--retries", type=int, default=2,
                        help="relaunches of an eval step that ended with generation failures")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.execute and args.workspace is None:
        parser.error("--execute requires --workspace (new, or one this script prepared)")
    if args.retries < 0:
        parser.error("--retries must be zero or more")
    try:
        schedule = plan(args.config, args.workspace)
        result = execute(schedule, args.workspace, retries=args.retries) if args.execute else schedule
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not args.execute:
        return 0 if not schedule["problems"] else 1
    return 0 if result["status"] == "completed-measurements" else 1


if __name__ == "__main__":
    raise SystemExit(main())
