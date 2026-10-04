"""Summarize immutable Stage14 reports without opening market outcome sources."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def summary(root: Path, run_id: str) -> dict[str, Any]:
    directory = root / "results/robustness/runs" / run_id
    result: dict[str, Any] = {}
    for filename in (
        "verdict.json",
        "inventory.json",
        "statistics.json",
        "null_controls.json",
        "bootstrap.json",
        "monte_carlo.json",
        "econometric_sensitivity.json",
    ):
        file = directory / filename
        if not file.exists():
            continue
        body = json.loads(file.read_text(encoding="utf-8"))
        if filename == "inventory.json":
            result[filename] = {
                "historical": body["historical_recorded_statuses"],
                "synthetic": body["synthetic_recorded_statuses"],
                "registries": [{k: r.get(k) for k in ("path", "rows")} for r in body["registries"]],
                "access_history": body["metadata"]["evaluation_history"],
            }
        elif filename == "statistics.json":
            result[filename] = [
                {
                    "run": r["source_run"],
                    **{
                        k: r["effect"].get(k)
                        for k in (
                            "candidate",
                            "benchmark",
                            "rows",
                            "mean_improvement",
                            "relative_mse_improvement",
                            "positive_fold_fraction",
                        )
                    },
                }
                for r in body
                if r["effect"]
            ]
        elif filename == "null_controls.json":
            result[filename] = body["finite_monte_carlo"]
        elif filename == "bootstrap.json" and isinstance(body, dict):
            result[filename] = [
                {k: r.get(k) for k in ("block", "status", "interval", "bootstrap_mean_sd")}
                for r in body["records"]
            ]
        elif filename == "monte_carlo.json":
            result[filename] = {
                "quantiles": body.get("marked_pnl_quantiles"),
                "paths": len(body.get("records", [])),
            }
        elif filename == "econometric_sensitivity.json":
            result[filename] = {
                "state": [
                    {k: r[k] for k in ("q_multiplier", "mse", "final_covariance")}
                    for r in body["state"]["results"]
                ]
                if body["state"]
                else None,
                "intervals": body["intervals"],
                "monitor": body["monitor"],
            }
        elif filename == "verdict.json":
            result[filename] = body
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    print(json.dumps(summary(Path(__file__).resolve().parents[1], args.run_id), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
