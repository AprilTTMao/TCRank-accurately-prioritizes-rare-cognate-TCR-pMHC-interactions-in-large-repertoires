#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import shlex
import subprocess
import tempfile
import shutil
from datetime import datetime
from pathlib import Path
import re
import os

import pandas as pd


# 你要求合并时只看这几列内容，并额外加一列 peptide（从模型名解析出来）
KEEP_COLS = ["cdr3_TRA", "cdr3_TRB", "TRAV", "TRAJ", "TRBV", "TRBJ", "score", "perc_rank"]
FINAL_COLS = ["pair_id","HLA","cdr3_TRA","cdr3_TRB","TRAV","TRAJ","TRBV","TRBJ","peptide","score","perc_rank"]


def run_cmd(cmd: str, cwd: Path, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8", errors="ignore") as logf:
        p = subprocess.run(
            cmd,
            shell=True,
            cwd=str(cwd),
            stdout=logf,
            stderr=logf,
            text=True,
        )
    return p.returncode


def read_mixtcrpred_output(path: Path) -> pd.DataFrame:
    """
    Read MixTCRpred output CSV while skipping banner lines starting with '#'.
    Return empty DataFrame if unreadable.
    """
    if (not path.exists()) or path.stat().st_size == 0:
        return pd.DataFrame()

    # MixTCRpred 输出 CSV 前面有很多 ### banner 行；comment="#" 会跳过这些行
    for sep, engine in [(",", "c"), ("\t", "c"), (None, "python")]:
        try:
            df = pd.read_csv(path, sep=sep, engine=engine, comment="#")
            if df is None or df.empty:
                continue
            # 清掉 Unnamed 垃圾列（防御）
            df = df.loc[:, [c for c in df.columns if not str(c).startswith("Unnamed")]].copy()
            return df
        except Exception:
            continue

    return pd.DataFrame()


def cleanup_old_runs(tmp_base: Path, keep: int = 5):
    tmp_base.mkdir(parents=True, exist_ok=True)
    runs = [p for p in tmp_base.iterdir() if p.is_dir() and p.name.startswith("run_")]
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for p in runs[keep:]:
        try:
            shutil.rmtree(p)
        except Exception:
            pass



def format_hla(token: str) -> str:
    """Convert allele token like A0101 / A01:01 / HLA-A*01:01 -> HLA-A*01:01"""
    if token is None:
        return ""
    s = str(token).upper().replace("HLA-", "").replace("*", "").replace(":", "").replace("_", "")
    # Expect like A0101
    m = re.match(r"^([A-Z])(\d{2})(\d{2})$", s)
    if not m:
        # Try A01
        m2 = re.match(r"^([A-Z])(\d{2})$", s)
        if m2:
            return f"HLA-{m2.group(1)}*{m2.group(2)}:01"
        return str(token).strip()
    return f"HLA-{m.group(1)}*{m.group(2)}:{m.group(3)}"
def safe_token(s: str) -> str:
    """
    Make a safe filename token from model string (avoid '/' ':' spaces, etc.)
    """
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s))


def extract_peptide_from_model(model: str) -> str:
    """
    从模型名/ckpt路径提取 peptide：
      - A0101_LTDEMIAQY              -> LTDEMIAQY
      - model_A0101_LTDEMIAQY.ckpt   -> LTDEMIAQY
      - /path/to/model_A0101_LTDEMIAQY.ckpt -> LTDEMIAQY
      - /path/to/A0101_LTDEMIAQY     -> LTDEMIAQY
    """
    s = str(model).strip()
    if not s:
        return ""
    # 取 basename
    base = s.replace("\\", "/").split("/")[-1]

    # 去掉后缀
    if base.lower().endswith(".ckpt"):
        base = base[:-5]

    # 去掉 model_ 前缀
    if base.startswith("model_"):
        base = base[len("model_"):]

    # 现在期望是 A0201_PEPTIDE
    if "_" in base:
        parts = base.split("_", 1)
        if len(parts) == 2:
            pep = parts[1].strip().upper()
            return pep

    return ""


def extract_allele_from_model(model: str) -> str:
    base = str(model).replace("\\", "/").split("/")[-1]
    if base.startswith("model_"):
        base = base[len("model_"):]
    return base.split("_", 1)[0] if "_" in base else ""


def attach_pair_ids(output: pd.DataFrame, inputs: pd.DataFrame, pair_ids) -> pd.DataFrame:
    # MixTCRpred sorts and may filter its output, so row position is not an ID.
    keys = ["cdr3_TRA", "cdr3_TRB", "TRAV", "TRAJ", "TRBV", "TRBJ"]
    expected = inputs[keys].copy().fillna("").astype(str)
    actual = output[keys].copy().fillna("").astype(str)
    expected["_occurrence"] = expected.groupby(keys, dropna=False).cumcount()
    actual["_occurrence"] = actual.groupby(keys, dropna=False).cumcount()
    expected["pair_id"] = pair_ids
    matched = actual.merge(expected, on=keys + ["_occurrence"], how="left", sort=False, validate="one_to_one")
    if matched["pair_id"].isna().any():
        raise ValueError("MixTCRpred returned a receptor that cannot be matched to its input")
    result = output.copy()
    result["pair_id"] = matched["pair_id"].astype(int).to_numpy()
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True, help="conda env for running MixTCRpred itself")
    ap.add_argument("--workdir", required=True, help="MixTCRpred repo directory")
    ap.add_argument(
        "--cmd_tpl",
        required=True,
        help='Command template, e.g. python MixTCRpred.py --model {model} --input "{input_csv}" --output "{output_csv}"',
    )
    ap.add_argument("--input_csv", required=True, help="MixTCRpred_input.csv")
    ap.add_argument("--plan_csv", required=True, help="MixTCRpred_plan.csv (must contain row_index, model)")
    ap.add_argument("--output_csv", required=True, help="Final output CSV path (MixTCRpred_output.csv)")
    args = ap.parse_args()

    workdir = Path(args.workdir).resolve()

    input_df = pd.read_csv(args.input_csv)
    plan = pd.read_csv(args.plan_csv)

    if "row_index" not in plan.columns or "model" not in plan.columns:
        raise ValueError("MixTCRpred_plan.csv must contain columns: row_index, model")

    # plan_row_order: 按 plan 里 row_index 第一次出现的顺序，映射到 input_df 的 position
    plan_row_order = []
    seen = set()
    for x in plan["row_index"].tolist():
        try:
            xi = int(x)
        except Exception:
            continue
        if xi not in seen:
            seen.add(xi)
            plan_row_order.append(xi)
    pos_map = {orig: pos for pos, orig in enumerate(plan_row_order)}

    final_out = Path(args.output_csv).resolve()
    final_out.parent.mkdir(parents=True, exist_ok=True)

    tmp_base = final_out.parent / "_tmp_mixtcrpred"
    cleanup_old_runs(tmp_base, keep=5)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tmp_root = Path(tempfile.mkdtemp(prefix=f"run_{stamp}_", dir=str(tmp_base)))
    tmp_parts = tmp_root / "parts"
    tmp_logs = tmp_root / "logs"
    tmp_parts.mkdir(parents=True, exist_ok=True)
    tmp_logs.mkdir(parents=True, exist_ok=True)

    all_results = []

    # 每个 model 跑一次
    for model, g in plan.groupby("model"):
        orig_idx = []
        for x in g["row_index"].tolist():
            try:
                orig_idx.append(int(x))
            except Exception:
                continue

        # 转换为 MixTCRpred_input.csv 的 position，并同时保留对应的 pair_id（= 原始 row_index）
        pos_idx = []
        pair_ids = []
        for oi in orig_idx:
            if oi in pos_map:
                pi = pos_map[oi]
                if 0 <= pi < len(input_df):
                    pos_idx.append(pi)
                    pair_ids.append(int(oi))

        if not pos_idx:
            continue

        sub = input_df.iloc[pos_idx].copy()

        pep = extract_peptide_from_model(model)
        allele = extract_allele_from_model(model)
        hla = format_hla(allele)

        token = safe_token(model)
        tmp_in = tmp_parts / f"{token}.input.csv"
        tmp_out = tmp_parts / f"{token}.output.csv"
        tmp_log = tmp_logs / f"{token}.log"

        sub.to_csv(tmp_in, index=False)

        cmd = args.cmd_tpl.format(
            model=str(model),
            input_csv=str(tmp_in),
            output_csv=str(tmp_out),
        )

        conda = os.environ.get("CONDA_EXE") or shutil.which("conda")
        if not conda:
            raise RuntimeError("Conda executable was not found")
        full_cmd = f'{shlex.quote(conda)} run -n {shlex.quote(args.env)} --no-capture-output bash -c {shlex.quote(cmd)}'
        rc = run_cmd(full_cmd, cwd=workdir, log_path=tmp_log)
        if rc != 0:
            raise RuntimeError(f"MixTCRpred model {model} failed (exit {rc}); see {tmp_log}")

        out_df = read_mixtcrpred_output(tmp_out)
        if out_df.empty:
            raise RuntimeError(f"MixTCRpred model {model} returned no predictions; see {tmp_log}")

        # 必须包含你要求的 8 列
        if any(c not in out_df.columns for c in KEEP_COLS):
            raise RuntimeError(f"MixTCRpred model {model} returned unexpected columns: {list(out_df.columns)}")

        out_df = out_df[KEEP_COLS].copy()
        out_df = attach_pair_ids(out_df, sub, pair_ids)
        out_df["peptide"] = pep  # ✅ 来自 model 名
        out_df["HLA"] = hla      # ✅ 来自 model 名 (标准写法)

        # 严格按最终列顺序输出
        out_df = out_df[FINAL_COLS].copy()
        all_results.append(out_df)

    merged = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame(columns=FINAL_COLS)
    merged.to_csv(final_out, index=False)


if __name__ == "__main__":
    main()
