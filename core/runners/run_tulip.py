#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import sys

import pandas as pd


FINAL_COLS = ["CDR3a", "CDR3b", "peptide", "score", "rank", "rank_score"]


def _valid_tulip_token_text(value: object, *, allow_missing: bool = False) -> bool:
    text = "" if pd.isna(value) else str(value).strip()
    if allow_missing and text == "<MIS>":
        return True
    return 0 < len(text) <= 48


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--input_csv", required=True)
    ap.add_argument("--output_csv", required=True)
    ap.add_argument("--modelconfig", default="configs/shallow.config.json")
    ap.add_argument("--load", default="model_weights/pytorch_model.bin")
    ap.add_argument("--batch_size", default="256")
    args = ap.parse_args()

    workdir = Path(args.workdir).resolve()
    input_csv = Path(args.input_csv).resolve()
    output_csv = Path(args.output_csv).resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    tmp_dir = output_csv.parent / "_tmp_tulip"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    input_df = pd.read_csv(input_csv)
    keep = (
        input_df["CDR3a"].map(lambda x: _valid_tulip_token_text(x, allow_missing=True))
        & input_df["CDR3b"].map(lambda x: _valid_tulip_token_text(x, allow_missing=True))
        & input_df["peptide"].map(_valid_tulip_token_text)
    )
    run_input = tmp_dir / "input.csv"
    filtered_df = input_df.loc[keep].copy()
    filtered_df.to_csv(run_input, index=False)

    if filtered_df.empty:
        pd.DataFrame(columns=FINAL_COLS).to_csv(output_csv, index=False)
        return 0

    prefix = str(tmp_dir / "part_")
    cmd = [
        sys.executable,
        "predict.py",
        "--test_dir",
        str(run_input),
        "--modelconfig",
        args.modelconfig,
        "--load",
        args.load,
        "--output",
        prefix,
        "--batch_size",
        str(args.batch_size),
    ]
    subprocess.run(cmd, cwd=str(workdir), check=True)

    frames = []
    for p in sorted(tmp_dir.glob("part_*.csv")):
        try:
            frame = pd.read_csv(p)
            frame = frame.loc[:, ~frame.columns.str.startswith("Unnamed")]
            frames.append(frame)
        except Exception:
            continue

    if frames:
        out = pd.concat(frames, ignore_index=True)
    else:
        out = pd.DataFrame(columns=["CDR3a", "CDR3b", "peptide", "score", "rank"])

    if "rank" in out.columns:
        group_size = out.groupby("peptide")["rank"].transform("size")
        denom = (group_size - 1).replace(0, pd.NA)
        out["rank_score"] = (out["rank"].astype(float) / denom).fillna(1.0)
    elif "score" in out.columns:
        out["rank_score"] = out["score"]
    else:
        out["rank_score"] = []

    out.to_csv(output_csv, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
