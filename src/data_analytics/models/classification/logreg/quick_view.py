#!/usr/bin/env python3
"""
Quick viewer for the latest Logistic Regression run.
- Finds the most recent artifacts/run_* folder
- Prints metrics
- Shows top-N ZIP codes by predicted probability
- (Optional) writes a small CSV with the top ZIPs for easy plotting/cards

Usage
-----
python src/data_analytics/models/classification/logreg/quick_view.py \
  --artifacts-dir src/data_analytics/models/classification/logreg/artifacts \
  --top-n 25 \
  --write-top-csv
"""
import argparse
import json
import os
from glob import glob

import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--artifacts-dir", default="src/data_analytics/models/classification/logreg/artifacts")
    p.add_argument("--top-n", type=int, default=25)
    p.add_argument("--write-top-csv", action="store_true")
    return p.parse_args()


def latest_run_dir(artifacts_dir: str) -> str:
    runs = sorted(glob(os.path.join(artifacts_dir, "run_*")))
    if not runs:
        raise SystemExit(f"No runs found under {artifacts_dir}")
    return runs[-1]


def main():
    args = parse_args()
    run_dir = latest_run_dir(args.artifacts_dir)
    print(f"[INFO] Using latest run: {run_dir}")

    # Metrics
    metrics_path = os.path.join(run_dir, "metrics.json")
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    print("\n=== Metrics ===")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k:>16}: {v:.4f}")
        else:
            print(f"{k:>16}: {v}")

    # Predictions by ZIP
    pred_csv = os.path.join(run_dir, "predictions_test_by_zip.csv")
    if not os.path.exists(pred_csv):
        raise SystemExit(f"Predictions CSV not found: {pred_csv}")

    df = pd.read_csv(pred_csv)
    df = df.sort_values("prob_ocurre_siniestro", ascending=False)

    print("\n=== Top ZIPs by risk ===")
    print(df.head(args.top_n).to_string(index=False))

    if args.write_top_csv:
        out_csv = os.path.join(run_dir, f"top_{args.top_n}_zips.csv")
        df.head(args.top_n).to_csv(out_csv, index=False)
        print(f"\n[OK] Wrote {out_csv}")


if __name__ == "__main__":
    main()
