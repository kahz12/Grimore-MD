"""Compare extraction quality, wall time and peak RSS by hardware profile.

Each measurement runs in a fresh process so ``ru_maxrss`` belongs to one
engine/variant. Optional engines are reported as skipped when unavailable;
this harness never installs or silently substitutes them.

Usage::

    venv/bin/python eval/benchmark_extraction.py --repeat 3 --include-legacy
    venv/bin/python eval/benchmark_extraction.py --include-ocr --profile-label laptop-cpu
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = Path(__file__).with_name("phase4_extraction.yaml")
ENGINES = ("pypdf", "pdfplumber", "pymupdf")
MEASURED_CODE = (
    "eval/benchmark_extraction.py",
    "eval/check_extraction.py",
    "grimore/ingest/adapters/base.py",
    "grimore/ingest/adapters/pdf.py",
    "grimore/ingest/pdf_layout.py",
    "grimore/ingest/parser.py",
    "grimore/utils/ocr.py",
)


def _version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _cpu_model() -> str | None:
    info = Path("/proc/cpuinfo")
    if info.is_file():
        try:
            for line in info.read_text(encoding="utf-8").splitlines():
                if line.startswith("model name"):
                    return line.partition(":")[2].strip() or None
        except OSError:
            pass
    return platform.processor() or None


def _tool_probe(name: str, arguments: list[str]) -> dict:
    """Identify optional binaries without letting a diagnostic hang the run."""
    executable = shutil.which(name)
    if executable is None:
        return {"path": None, "status": "missing", "output": None}
    try:
        result = subprocess.run(
            [executable, *arguments], capture_output=True, text=True,
            timeout=5, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return {"path": executable, "status": "error", "output": None}
    return {
        "path": executable,
        "status": "ok" if result.returncode == 0 else "error",
        "output": (result.stdout or result.stderr).strip()[:8192],
    }


def _language_data(probe: dict) -> dict:
    """Hash installed traineddata, not just model names that may hide different weights."""
    lines = (probe.get("output") or "").splitlines()
    if not lines or probe.get("status") != "ok":
        return {}
    match = re.fullmatch(r'List of available languages in "(.+)" \(\d+\):', lines[0])
    if match is None or not Path(match[1]).is_absolute():
        return {}
    root = Path(match[1])
    result = {}
    for language in lines[1:]:
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", language):
            continue
        try:
            digest = _sha256(root / f"{language}.traineddata")
        except OSError:
            digest = None
        result[language] = digest
    return result


def hardware_profile(label: str | None = None) -> dict:
    """Record enough host detail to avoid comparing unlike machines blindly."""
    try:
        ram_gib = round(
            os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / (1024 ** 3), 2,
        )
    except (AttributeError, OSError, ValueError):
        ram_gib = None
    language_probe = _tool_probe("tesseract", ["--list-langs"])
    return {
        "label": label or "auto",
        "system": platform.system(),
        "release": platform.release(),
        "architecture": platform.machine(),
        "cpu_model": _cpu_model(),
        "logical_cpus": os.cpu_count(),
        "ram_gib": ram_gib,
        "python": platform.python_version(),
        "ocr_environment": {
            name: os.environ.get(name) for name in ("OMP_THREAD_LIMIT", "OMP_NUM_THREADS")
        },
        "external_tools": {
            "tesseract": _tool_probe("tesseract", ["--version"]),
            "tesseract_languages": language_probe,
            "pdfinfo": _tool_probe("pdfinfo", ["-v"]),
            "pdftoppm": _tool_probe("pdftoppm", ["-v"]),
            "pdftocairo": _tool_probe("pdftocairo", ["-v"]),
        },
        "ocr_language_data_sha256": _language_data(language_probe),
        "packages": {
            "pypdf": _version("pypdf"),
            "pdfplumber": _version("pdfplumber"),
            "pymupdf": _version("PyMuPDF"),
            "pytesseract": _version("pytesseract"),
            "pdf2image": _version("pdf2image"),
            "pdfminer.six": _version("pdfminer.six"),
            "Pillow": _version("Pillow"),
            "pypdfium2": _version("pypdfium2"),
            "cryptography": _version("cryptography"),
        },
    }


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def input_fingerprints(manifest: Path) -> dict:
    """Identify the exact corpus and extraction code behind a measurement."""
    from eval.check_extraction import load_manifest

    root, documents = load_manifest(manifest)
    return {
        "manifest_sha256": _sha256(manifest),
        "documents": [
            {
                "id": item["id"], "path": item["path"],
                "size_bytes": (root / item["path"]).stat().st_size,
                "sha256": _sha256(root / item["path"]),
            }
            for item in documents
        ],
        "code_sha256": {path: _sha256(REPO_ROOT / path) for path in MEASURED_CODE},
    }


def _unavailable(engine: str, ocr: bool, ocr_language: str = "eng") -> list[str]:
    missing = []
    if engine == "pdfplumber" and importlib.util.find_spec("pdfplumber") is None:
        missing.append("pdfplumber extra")
    if engine == "pymupdf" and all(
        importlib.util.find_spec(module) is None for module in ("pymupdf", "fitz")
    ):
        missing.append("pymupdf extra (AGPL-3.0)")
    if ocr:
        for module in ("pytesseract", "pdf2image"):
            if importlib.util.find_spec(module) is None:
                missing.append(f"{module} extra")
        if shutil.which("tesseract") is None:
            missing.append("tesseract binary")
        if shutil.which("pdfinfo") is None:
            missing.append("poppler pdfinfo binary")
        if not (shutil.which("pdftoppm") or shutil.which("pdftocairo")):
            missing.append("poppler binary")
        if not missing:
            from grimore.utils.ocr import check_ocr_languages

            if failure := check_ocr_languages(shutil.which("tesseract"), ocr_language):
                missing.append(failure)
    return missing


def _peak_rss_mb(*, children: bool = False) -> float | None:
    try:
        import resource
    except ImportError:
        return None
    scope = resource.RUSAGE_CHILDREN if children else resource.RUSAGE_SELF
    value = resource.getrusage(scope).ru_maxrss
    divisor = 1024 ** 2 if sys.platform == "darwin" else 1024
    return round(value / divisor, 2)


def worker(manifest: Path, engine: str, ocr: bool, legacy: bool,
           ocr_language: str = "eng", ocr_max_pages: int = 0) -> dict:
    """Execute one quality pass in this process and return its cost."""
    sys.path.insert(0, str(REPO_ROOT))
    from eval.check_extraction import evaluate

    before_rss = _peak_rss_mb()
    start = time.perf_counter()
    # The adapter may log page warnings to stdout. Keep the worker's stdout
    # reserved for its single JSON record, even when extraction is incomplete.
    with redirect_stdout(sys.stderr):
        report = evaluate(
            manifest, pdf_engine=engine, ocr=ocr, legacy_pypdf_plain=legacy,
            ocr_language=ocr_language,
            ocr_max_pages=ocr_max_pages,
        )
    elapsed = time.perf_counter() - start
    peak_rss = _peak_rss_mb()
    child_peak_rss = _peak_rss_mb(children=True)
    return {
        "wall_s": round(elapsed, 6),
        "peak_rss_mb": peak_rss,
        "child_peak_rss_mb": child_peak_rss,
        "rss_increase_mb": (
            round(max(0.0, peak_rss - before_rss), 2)
            if peak_rss is not None and before_rss is not None else None
        ),
        "quality": {
            "evaluation_semantics": report["evaluation_semantics"],
            "ocr_max_pages": report["ocr_max_pages"],
            "passed": report["passed"],
            "totals": report["totals"],
            "ocr_sections": report["ocr_sections"],
            "page_diagnostics": [
                {"document_id": item["id"], "ocr_pages": item.get("ocr_pages", []),
                 "extraction_issues": item.get("extraction_issues", [])}
                for item in report["documents"] if "error" not in item
            ],
            "document_errors": [
                item["id"] for item in report["documents"] if "error" in item
            ],
            "failed_checks": [
                {"document_id": item["id"], **check}
                for item in report["documents"]
                for check in item["checks"] if not check["passed"]
            ],
        },
    }


def _run_once(manifest: Path, engine: str, ocr: bool, legacy: bool,
              timeout_s: int, ocr_language: str = "eng", ocr_max_pages: int = 0) -> dict:
    command = [
        sys.executable, str(Path(__file__).resolve()), "--worker",
        "--manifest", str(manifest), "--engine", engine,
        "--ocr-language", ocr_language,
        "--ocr-max-pages", str(ocr_max_pages),
    ]
    if ocr:
        command.append("--ocr")
    if legacy:
        command.append("--legacy")
    try:
        completed = subprocess.run(
            command, cwd=REPO_ROOT, text=True, capture_output=True,
            timeout=timeout_s, check=False,
        )
    except subprocess.TimeoutExpired:
        return {"error": f"timeout after {timeout_s}s"}
    if completed.returncode != 0:
        return {"error": (completed.stderr or completed.stdout).strip()[-1000:]}
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError:
        return {"error": f"invalid worker output: {completed.stdout[-500:]}"}


def benchmark(
    manifest: Path, *, engines: list[str] | tuple[str, ...] = ENGINES,
    repeat: int = 3, include_ocr: bool = False, include_legacy: bool = False,
    timeout_s: int = 300, profile_label: str | None = None,
    ocr_language: str = "eng",
    ocr_max_pages: int = 0,
) -> dict:
    """Benchmark requested variants; unavailable extras remain explicit skips."""
    if repeat < 1 or repeat > 20:
        raise ValueError("repeat must be between 1 and 20")
    if timeout_s < 1:
        raise ValueError("timeout must be positive")
    if not engines or any(engine not in ENGINES for engine in engines):
        raise ValueError(f"engines must be selected from {ENGINES}")

    sys.path.insert(0, str(REPO_ROOT))
    from grimore.utils.ocr import validate_ocr_language, validate_ocr_max_pages

    validate_ocr_language(ocr_language)
    validate_ocr_max_pages(ocr_max_pages)
    manifest = manifest.resolve()
    inputs = input_fingerprints(manifest)
    variants = [(engine, False, False) for engine in engines]
    if include_ocr:
        variants += [(engine, True, False) for engine in engines]
    if include_legacy:
        variants.append(("pypdf", False, True))

    rows = []
    for engine, ocr, legacy in variants:
        name = "pypdf-legacy-plain" if legacy else engine + ("+ocr" if ocr else "")
        missing = _unavailable(engine, ocr, ocr_language)
        if missing:
            rows.append({"variant": name, "status": "skipped", "reason": ", ".join(missing)})
            continue
        samples = [
            _run_once(manifest, engine, ocr, legacy, timeout_s, ocr_language, ocr_max_pages)
            for _ in range(repeat)
        ]
        errors = [sample["error"] for sample in samples if "error" in sample]
        if errors:
            rows.append({
                "variant": name, "status": "error", "errors": errors,
                "successful_samples": len(samples) - len(errors),
            })
            continue
        quality = samples[0]["quality"]
        peaks = [sample["peak_rss_mb"] for sample in samples]
        child_peaks = [sample["child_peak_rss_mb"] for sample in samples]
        increases = [sample["rss_increase_mb"] for sample in samples]
        rows.append({
            "variant": name,
            "status": "ok",
            "repeats": repeat,
            "wall_s_median": round(statistics.median(s["wall_s"] for s in samples), 6),
            "peak_rss_mb_median": (
                round(statistics.median(peaks), 2) if all(p is not None for p in peaks)
                else None
            ),
            "child_peak_rss_mb_median": (
                round(statistics.median(child_peaks), 2)
                if all(p is not None for p in child_peaks) else None
            ),
            "rss_increase_mb_median": (
                round(statistics.median(increases), 2)
                if all(p is not None for p in increases) else None
            ),
            "quality": quality,
            "quality_consistent": all(sample["quality"] == quality for sample in samples),
            "ocr_exercised": bool(quality["ocr_sections"]) if ocr else None,
            "samples": samples,
        })
    return {
        "version": 2,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "manifest": str(manifest),
        "inputs": inputs,
        "inputs_unchanged": inputs == input_fingerprints(manifest),
        "profile": hardware_profile(profile_label),
        "settings": {
            "repeat": repeat, "timeout_s": timeout_s,
            "include_ocr": include_ocr, "include_legacy": include_legacy,
            "engines": list(engines),
            "ocr_language": ocr_language,
            "ocr_max_pages": ocr_max_pages,
        },
        "results": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--engine", choices=ENGINES, default="pypdf", help=argparse.SUPPRESS)
    parser.add_argument("--engines", nargs="+", choices=ENGINES, default=list(ENGINES))
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--profile-label")
    parser.add_argument("--include-ocr", action="store_true")
    parser.add_argument("--ocr-language", default="eng", help="Tesseract language, e.g. spa+eng")
    parser.add_argument("--ocr-max-pages", type=int, default=0,
                        help="OCR attempts per PDF; 0 means unlimited")
    parser.add_argument("--include-legacy", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--ocr", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--legacy", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        if args.worker:
            result = worker(
                args.manifest.resolve(), args.engine, args.ocr, args.legacy,
                args.ocr_language, args.ocr_max_pages,
            )
        else:
            result = benchmark(
                args.manifest, engines=args.engines, repeat=args.repeat,
                include_ocr=args.include_ocr, include_legacy=args.include_legacy,
                timeout_s=args.timeout, profile_label=args.profile_label,
                ocr_language=args.ocr_language,
                ocr_max_pages=args.ocr_max_pages,
            )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
