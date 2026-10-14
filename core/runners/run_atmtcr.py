#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import subprocess
import sys

import pandas as pd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--input_csv", required=True)
    ap.add_argument("--output_csv", required=True)
    ap.add_argument("--model_name", default="original.ckpt")
    ap.add_argument("--batch_size", default="128")
    args = ap.parse_args()

    workdir = Path(args.workdir).resolve()
    input_csv = Path(args.input_csv).resolve()
    output_csv = Path(args.output_csv).resolve()
    raw_out = output_csv.with_suffix(output_csv.suffix + ".raw.tsv")
    no_header_input = output_csv.with_suffix(output_csv.suffix + ".atm_input.csv")
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    src = pd.read_csv(input_csv)
    src.to_csv(no_header_input, index=False, header=False)

    cmd = [
        sys.executable,
        "main.py",
        "--infile",
        str(workdir / "data" / "combined_dataset.csv"),
        "--indepfile",
        str(no_header_input),
        "--model_name",
        args.model_name,
        "--mode",
        "test",
        "--cuda",
        "False",
        "--batch_size",
        str(args.batch_size),
        "--heads",
        "5",
        "--pred_out",
        str(raw_out),
    ]
    subprocess.run(cmd, cwd=str(workdir), check=True)

    rows = []
    with raw_out.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) < 5:
                continue
            rows.append({
                "Peptide": row[0],
                "CDR3": row[1],
                "Actual": row[2],
                "Prediction": row[3],
                "Score": row[4],
            })

    pd.DataFrame(rows, columns=["Peptide", "CDR3", "Actual", "Prediction", "Score"]).to_csv(output_csv, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
