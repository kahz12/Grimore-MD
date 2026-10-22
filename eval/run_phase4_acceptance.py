"""Plan by default; --execute explicitly starts local-model acceptance measurements.

Each run uses a fresh process and the answer bank's temporary vault/index.
Original documents/configuration and historical reports are never overwritten.
This is a technical control suite, not representative documents or a pilot.
"""
# ruff: noqa: E402 -- direct execution must find the repository package
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import json
from pathlib import Path
import subprocess
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.benchmark_answers import DEFAULT_CASES, DEFAULT_PDF, fingerprints, sha256
from eval.compare_answers import compare, load_report
from grimore.cognition.eval import load_golden
from grimore.utils.config import load_config
from grimore.utils.security import SecurityGuard

DEFAULT_NEGATIVE = REPO_ROOT / "eval/phase4_unanswerable_answers.yaml"


def plan(positive: Path, negative: Path, pdf: Path, config: Path, *, repeat=3, model=None, timeout_s=600):
    if type(repeat) is not int or not 1 <= repeat <= 10:
        raise ValueError("repeat must be an integer from 1 to 10")
    if type(timeout_s) is not int or timeout_s < 1:
        raise ValueError("timeout must be a positive integer")
    positive, negative, pdf, config = (p.resolve() for p in (positive, negative, pdf, config))
    settings = load_config(str(config))
    manifests = {"positive": positive, "negative": negative}
    counts, inputs = {}, {}
    for role, manifest in manifests.items():
        cases = load_golden(manifest)
        if not cases or any(c.split != "tuning" or c.document_formats != ["pdf"]
                            or c.negative != (role == "negative") for c in cases):
            raise ValueError("requires separate positive/negative PDF tuning manifests, never held-out evaluation")
        counts[role] = sum(1 + len(c.follow_ups) for c in cases)
        # Validate follow-up roles too; this control must not silently mix denominators.
        if any(f.negative != (role == "negative") for c in cases for f in c.follow_ups):
            raise ValueError("follow-up roles must match their manifest")
        inputs[role] = fingerprints(manifest, pdf, config)
    runs = []
    for repetition in range(1, repeat + 1):
        for role, order in (("positive", "ranked"), ("positive", "rotate-first-last"), ("negative", "ranked")):
            # Alternate baseline-first/candidate-first to expose order effects,
            # without claiming that cache/residency is thereby controlled.
            for enabled in ([False, True] if repetition % 2 else [True, False]):
                name = f"{role}_{order}_r{repetition}_{'on' if enabled else 'off'}"
                runs.append({"id": name, "role": role, "repetition": repetition,
                             "context_order": order, "enabled": enabled,
                             "cases": str(manifests[role]), "report": name + ".json"})
    return {"version": 1, "status": "planned-not-executed", "repeat": repeat,
            "pdf": str(pdf), "config": str(config), "model": model or settings.cognition.model_llm_local,
            "sufficiency_timeout_s": timeout_s, "runs": runs, "inputs": inputs,
            "oracle_turns": sum(counts[r["role"]] for r in runs),
            "maximum_model_calls_excluding_rewrites_embeddings": sum(
                counts[r["role"]] * (2 if r["enabled"] else 1) for r in runs),
            "orchestrator_sha256": sha256(Path(__file__)),
            "limitations": ["Default action only prepares a plan; no models or downloads.",
                            "One PDF control, not representative validation or a pilot.",
                            "Human review is still required; no automatic semantic acceptance.",
                            "Repeated fresh client processes do not reset server cache/residency.",
                            "Technical failures are retained; a missing result is never a success."]}


def _write_new(path: Path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def execute(schedule: dict, output_dir: Path, *, run_timeout_s=7200):
    if type(run_timeout_s) is not int or run_timeout_s < 1:
        raise ValueError("run timeout must be a positive integer")
    # Validate provenance BEFORE creating any files or starting a child process.
    def unchanged(run):
        return fingerprints(Path(run["cases"]), Path(schedule["pdf"]), Path(schedule["config"])) == schedule["inputs"][run["role"]]

    if not all(unchanged(run) for run in schedule["runs"]):
        raise ValueError("inputs changed after planning")
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_new(output_dir / "plan.json", schedule)
    observations = []
    completed = {}
    pairs = []
    interrupted = False
    for run in schedule["runs"]:
        if not unchanged(run):
            observations.append({"id": run["id"], "status": "not-started", "reason": "inputs_changed"})
            break
        report = output_dir / run["report"]
        argv = [sys.executable, str(REPO_ROOT / "eval/benchmark_answers.py"),
                "--cases", run["cases"], "--pdf", schedule["pdf"], "--config", schedule["config"],
                "--model", schedule["model"], "--context-order", run["context_order"],
                "--sufficiency-timeout-s", str(schedule["sufficiency_timeout_s"]), "--output", str(report)]
        if run["enabled"]:
            argv.append("--sufficiency-check")
        started = time.monotonic()
        print(f"Measuring {run['id']}", file=sys.stderr, flush=True)
        observation = {"id": run["id"], "argv": argv, "status": "failed"}
        try:
            result = subprocess.run(argv, cwd=REPO_ROOT, capture_output=True, text=True, timeout=run_timeout_s)
            observation["exit_code"] = result.returncode
            observation["diagnostic"] = SecurityGuard.redact_for_log(result.stderr[-32_000:])
            if result.returncode == 0 and report.is_file():
                loaded = load_report(report)
                measured = loaded["controlled_run"]
                if (measured.get("inputs_unchanged") is True and unchanged(run)
                        and measured.get("inputs") == schedule["inputs"][run["role"]]
                        and (measured.get("sufficiency_control") or {}).get("enabled") is run["enabled"]
                        and (measured.get("cognition") or {}).get("model_llm_local") == schedule["model"]):
                    observation["status"] = "completed"
                    observation["report_sha256"] = sha256(report)
                    key = (run["role"], run["context_order"], run["repetition"])
                    completed.setdefault(key, {})[run["enabled"]] = loaded
                else:
                    observation["reason"] = "unstable_inputs_or_unmatched_measurement"
            else:
                observation["reason"] = "process_failed_or_missing_report"
        except subprocess.TimeoutExpired:
            observation["reason"] = "client_process_timeout"
        except KeyboardInterrupt:
            observation["reason"] = "interrupted"
            interrupted = True
        except (OSError, ValueError, TypeError, KeyError):
            observation["reason"] = "invalid_report_or_process_failure"
        observation["elapsed_s"] = time.monotonic() - started
        _write_new(output_dir / (run["id"] + ".status.json"), observation)
        observations.append(observation)
        if interrupted or observation.get("reason") == "client_process_timeout":
            break
    for key, reports in completed.items():
        if set(reports) == {False, True}:
            try:
                result = compare(reports[False], reports[True])
            except (ValueError, TypeError, KeyError):
                result = {"version": 1, "comparable": False, "reasons": ["invalid_completed_report"],
                          "semantic_acceptance": "pending-human-review"}
            name = f"compare_{key[0]}_{key[1]}_r{key[2]}.json"
            _write_new(output_dir / name, result)
            pairs.append({"report": name, "comparable": result["comparable"]})
    complete = len(observations) == len(schedule["runs"]) and all(o["status"] == "completed" for o in observations)
    summary = {"version": 1, "status": "completed-measurements" if complete else "incomplete-measurements",
               "planned_runs_n": len(schedule["runs"]), "attempts": observations, "pairs": pairs,
               "semantic_acceptance": "pending-human-review", "interrupted": interrupted}
    _write_new(output_dir / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positive-cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--negative-cases", type=Path, default=DEFAULT_NEGATIVE)
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "grimore.toml")
    parser.add_argument("--model")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--sufficiency-timeout-s", type=int, default=600)
    parser.add_argument("--run-timeout-s", type=int, default=7200)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.execute and args.output_dir is None:
        parser.error("--execute requires a NEW --output-dir")
    if not args.execute and args.output_dir is not None:
        parser.error("--output-dir requires --execute; planning writes no files")
    try:
        with redirect_stdout(sys.stderr):
            schedule = plan(args.positive_cases, args.negative_cases, args.pdf, args.config,
                            repeat=args.repeat, model=args.model, timeout_s=args.sufficiency_timeout_s)
            result = execute(schedule, args.output_dir, run_timeout_s=args.run_timeout_s) if args.execute else schedule
    except (OSError, ValueError, TypeError, KeyError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if not args.execute or result["status"] == "completed-measurements" else 1


if __name__ == "__main__":
    raise SystemExit(main())
