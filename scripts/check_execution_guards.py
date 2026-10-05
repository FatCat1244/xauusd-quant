"""Re-break Stage 12 guards in an isolated source copy; never mutate research source.

Each selected regression must fail with its guard disabled. Restore the copy after
each check and verify original source bytes afterward. Writes a new versioned record.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUDIT = "tests/test_stage12_audit_repairs.py::"
ENGINE = "tests/test_execution_engine.py::"

MUTATIONS = [
    (
        "regression_base_boundary",
        "features/store.py",
        "        if before is not None:\n            boundary =",
        "        if False and before is not None:\n            boundary =",
        AUDIT + "test_regression_boundary_precedes_every_feature_materialization",
    ),
    (
        "regression_window_prefix",
        "features/store.py",
        "                window_scan = window_scan.slice(0, frame.height)",
        "                window_scan = window_scan",
        AUDIT + "test_regression_boundary_precedes_every_feature_materialization",
    ),
    (
        "ml_development_boundary",
        "ml/datasets.py",
        "before=reserved)",
        "before=None)",
        AUDIT + "test_both_development_entry_points_supply_the_reserved_boundary",
    ),
    (
        "ensemble_development_boundary",
        "ensemble/final_test.py",
        "before=reserved)",
        "before=None)",
        AUDIT + "test_both_development_entry_points_supply_the_reserved_boundary",
    ),
    (
        "interrupted_attempt",
        "ml/final_test.py",
        "        if completed or started:",
        "        if completed:",
        AUDIT + "test_interrupted_ml_attempt_is_a_look_and_requires_a_logged_repeat_reason",
    ),
    (
        "missing_null_eligibility",
        "research/ensemble_research.py",
        '        if null_status == "untested":',
        '        if False and null_status == "untested":',
        AUDIT + "test_ensemble_missing_null_never_enters_an_eligible_universe",
    ),
    (
        "undefined_null",
        "research/ensemble_research.py",
        '"failed" if failed else "untested" if unknown else "passed"',
        '"failed" if failed else "passed"',
        AUDIT + "test_undefined_null_cannot_hide_beside_a_passing_null",
    ),
    (
        "bar_close_availability",
        "execution/policy.py",
        "if utc_time(forecast.available_at_utc) < earliest:",
        "if False and utc_time(forecast.available_at_utc) < earliest:",
        ENGINE + "test_invalid_forecasts_are_rejected_explicitly",
    ),
    (
        "strict_subsequent_quote",
        "execution/engine.py",
        "o.arrival_utc < stamp < o.expires_utc",
        "o.arrival_utc <= stamp < o.expires_utc",
        ENGINE + "test_latency_never_fills_an_earlier_or_same_arrival_quote",
    ),
    (
        "nonfinite_forecast",
        "execution/policy.py",
        "forecast.value is None or not math.isfinite(forecast.value)",
        "forecast.value is None",
        ENGINE + "test_invalid_forecasts_are_rejected_explicitly",
    ),
    (
        "promotion_unknown",
        "execution/readiness.py",
        'else "historical_diagnostic_only"',
        'else "eligible_for_economic_review"',
        "tests/test_execution_readiness.py::test_missing_scientific_or_specification_evidence_cannot_promote",
    ),
    (
        "adaptive_choice_chronology",
        "execution/readiness.py",
        "elif utc_time(datetime.fromisoformat(value)) >= boundary:",
        "elif False and utc_time(datetime.fromisoformat(value)) >= boundary:",
        "tests/test_execution_readiness.py::test_every_adaptive_choice_must_precede_the_evaluation_fold",
    ),
    (
        "complete_fold_coverage",
        "execution/readiness.py",
        "if cursor < utc_time(evaluation_end):",
        "if False and cursor < utc_time(evaluation_end):",
        "tests/test_execution_readiness.py::test_audited_fold_at_start_cannot_cover_unaudited_later_forecasts",
    ),
    (
        "unknown_scientific_null",
        "execution/readiness.py",
        'elif nulls.get(name) != "passed" and null_status != "failed":',
        'elif False and nulls.get(name) != "passed" and null_status != "failed":',
        "tests/test_execution_readiness.py::test_global_v001_selection_and_inspected_history_cannot_be_papered_over",
    ),
    (
        "evidence_roles",
        "execution/runs.py",
        "        if all(\n",
        "        if True or all(\n",
        "tests/test_execution_io.py::test_unrelated_hashed_file_does_not_verify_scientific_and_specification_evidence",
    ),
    (
        "no_reserved_execution_access",
        "execution/io.py",
        "if local_time(b, config) > datetime(2022, 1, 1):",
        "if False and local_time(b, config) > datetime(2022, 1, 1):",
        "tests/test_execution_readiness.py::test_utc_is_explicit_and_stage12_cannot_open_a_new_reserved_path",
    ),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("mutation output already exists; bump version")
    originals = {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob("src/**/*.py")
    }
    results = []
    with tempfile.TemporaryDirectory(prefix="stage12_mutation_") as temporary:
        copy = Path(temporary) / "src"
        shutil.copytree(ROOT / "src", copy, ignore=shutil.ignore_patterns("__pycache__"))
        environment = dict(os.environ, PYTHONPATH=str(copy), PYTHONDONTWRITEBYTECODE="1")
        for name, relative, before, after, test in MUTATIONS:
            path = copy / "xauusd_quant" / relative
            original = path.read_bytes()
            ending = "\r\n" if b"\r\n" in original else "\n"
            anchor = before.replace("\n", ending).encode()
            replacement = after.replace("\n", ending).encode()
            if anchor not in original:
                raise RuntimeError(f"missing mutation anchor: {name}")
            try:
                path.write_bytes(original.replace(anchor, replacement, 1))
                report_path = Path(temporary) / f"{name}.xml"
                command = [sys.executable, "-m", "pytest", test, "-q", "-p", "no:cacheprovider",
                           "--junitxml", str(report_path)]
                completed = subprocess.run(
                    command,
                    cwd=ROOT,
                    env=environment,
                    capture_output=True,
                    text=True,
                    timeout=90,
                    check=False,
                )
                # A fixture/collection error or "failed" in a test name is not
                # evidence that disabling the intended guard was detected.
                report = ET.parse(report_path).getroot() if report_path.exists() else None
                failures = len(report.findall(".//testcase/failure")) if report is not None else 0
                errors = len(report.findall(".//testcase/error")) if report is not None else 0
                caught = completed.returncode == 1 and failures > 0 and errors == 0
                results.append(
                    {
                        "guard": name,
                        "command": command,
                        "test_detected_break": caught,
                        "returncode": completed.returncode,
                        "assertion_failures": failures,
                        "test_errors": errors,
                        "test_output": completed.stdout[-1800:] + completed.stderr[-800:],
                    }
                )
                print(f"{name}: {'detected' if caught else 'NOT DETECTED'}", flush=True)
            finally:
                path.write_bytes(original)
    unchanged = all(
        hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
        for path, digest in originals.items()
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(
            {"isolated_copy": True, "original_source_unchanged": unchanged, "mutations": results},
            stream,
            indent=2,
        )
        stream.write("\n")
    return 0 if unchanged and all(r["test_detected_break"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
