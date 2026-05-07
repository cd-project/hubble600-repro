#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=Path, required=True)
    args = ap.parse_args()

    run_dir: Path = args.run_dir
    src = run_dir / "results_live.csv"
    if not src.exists():
        raise FileNotFoundError(f"Missing {src}")

    df = pd.read_csv(src)
    if df.empty:
        raise ValueError("results_live.csv is empty")

    df["_row"] = np.arange(len(df))
    latest = df.sort_values("_row").groupby("pairing_key", as_index=False).tail(1).copy()

    any_col = "ourdef_objective_any_success" if "ourdef_objective_any_success" in latest.columns else "ourdef_any_success"
    c_col = "ourdef_objective_success_count" if "ourdef_objective_success_count" in latest.columns else "ourdef_success_count"
    r_col = "ourdef_objective_most_true_match_ratio" if "ourdef_objective_most_true_match_ratio" in latest.columns else "ourdef_most_true_match_ratio"

    latest[any_col] = latest[any_col].astype(float)
    latest[c_col] = pd.to_numeric(latest[c_col], errors="coerce").fillna(0.0)
    latest[r_col] = pd.to_numeric(latest[r_col], errors="coerce").fillna(0.0)

    summary = (
        latest.groupby("duplication_count", as_index=False)
        .agg(
            n=("pairing_key", "count"),
            mean_r_at_5=(r_col, "mean"),
            any_at_5=(any_col, "mean"),
            mean_c_at_5=(c_col, "mean"),
        )
        .sort_values("duplication_count")
    )

    bucket = latest.copy()
    bucket["c_bucket"] = np.select(
        [bucket[c_col] == 0, bucket[c_col] == 5],
        ["C=0", "C=5"],
        default="1<=C<5",
    )
    by_bucket = (
        bucket.groupby(["duplication_count", "c_bucket"], as_index=False)
        .agg(n=("pairing_key", "count"))
        .sort_values(["duplication_count", "c_bucket"])
    )

    summary.to_csv(run_dir / "summary_by_dup.csv", index=False)
    by_bucket.to_csv(run_dir / "summary_by_c_bucket.csv", index=False)

    print("Summary by duplication:")
    for r in summary.itertuples(index=False):
        print(
            f"dup={int(r.duplication_count)} n={int(r.n)} "
            f"r@5={r.mean_r_at_5:.4f} Any@5={100*r.any_at_5:.2f}% meanC@5={r.mean_c_at_5:.3f}"
        )


if __name__ == "__main__":
    main()
