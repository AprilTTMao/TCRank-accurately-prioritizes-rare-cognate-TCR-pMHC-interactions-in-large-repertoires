#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import re
import sys
import pandas as pd

# -------------------------
# Helpers
# -------------------------

def norm_colname(s: str) -> str:
    return re.sub(r"\s+", "", str(s).strip().lower())

def pick_col(df: pd.DataFrame, wanted: str):
    w = norm_colname(wanted)
    for c in df.columns:
        if norm_colname(c) == w:
            return c
    return None

def norm_seq(s):
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return ""
    return re.sub(r"\s+", "", str(s).strip()).upper()

def safe_str(x):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    return str(x).strip()

def clean_gene_call(g: str) -> str:
    g = safe_str(g)
    if not g:
        return ""
    return g.split("*")[0].strip()

def write_rowmap(out_dir: Path, method_name: str, pair_ids, kept_mask=None, kept_indices=None):
    """Write <method>_rowmap.csv with columns: method_row_index, pair_id."""
    if kept_indices is None:
        if kept_mask is None:
            kept_indices = list(range(len(pair_ids)))
        else:
            kept_indices = [i for i, ok in enumerate(list(kept_mask)) if bool(ok)]
    pid_list = [int(pair_ids[i]) for i in kept_indices]
    rm = pd.DataFrame({
        "method_row_index": list(range(len(pid_list))),
        "pair_id": pid_list,
    })
    rm.to_csv(out_dir / f"{method_name}_rowmap.csv", index=False)


# -------------------------
# HLA parsing + formatting
# -------------------------

def _parse_hla_to_2field(h: str):
    """
    Convert common HLA strings -> (locus_letter, group2, allele2_or_None)
    Accepts:
      HLA-A*02:01, HLA-A02:01, A*02:01, A02:01, HLA-A*02, A*02, HLA-A0201, A0201
    Returns (locus, group, allele) or (None,None,None)
    """
    h = safe_str(h).upper().replace("HLA-", "").replace("_", "").replace(" ", "")
    if not h:
        return None, None, None

    # A*02:01 / A02:01 / A*02 / A02
    m = re.match(r"^([A-Z])\*?(\d{2})(?::?(\d{2}))?$", h)
    if m:
        locus, group, allele = m.group(1), m.group(2), m.group(3)
        return locus, group, allele

    # A0201
    m = re.match(r"^([A-Z])(\d{2})(\d{2})$", h)
    if m:
        locus, group, allele = m.group(1), m.group(2), m.group(3)
        return locus, group, allele

    return None, None, None

def hla_key_2field_with_prefix(h: str) -> str:
    """Key for pseudo34 map: HLA-A*02:01 (2-field)"""
    locus, group, allele = _parse_hla_to_2field(h)
    if locus is None:
        return ""
    if allele is None:
        allele = "01"
    return f"HLA-{locus}*{group}:{allele}"

def hla_for_piste(h: str) -> str:
    # PISTE example: HLA-A02:01 (no '*', must have ':01')
    locus, group, allele = _parse_hla_to_2field(h)
    if locus is None:
        return safe_str(h)
    if allele is None:
        allele = "01"
    return f"HLA-{locus}{group}:{allele}"

def hla_for_pmtnet(h: str) -> str:
    # pMTnet example: A*02:01 (no "HLA-" prefix)
    locus, group, allele = _parse_hla_to_2field(h)
    if locus is None:
        return safe_str(h)
    if allele is None:
        allele = "01"
    return f"{locus}*{group}:{allele}"

def hla_for_nettcr(h: str) -> str:
    return hla_for_pmtnet(h)

def hla_for_ergo(h: str) -> str:
    return hla_for_pmtnet(h)

def hla_for_epitcr_4field_no_prefix(h: str) -> str:
    """
    epiTCR example uses 4-field without HLA- prefix:
      A*02:01:01:01
    If user provides only 2-field, we pad :01:01
    """
    h = safe_str(h).upper().replace("HLA-", "").replace("_", "").replace(" ", "")
    if not h:
        return ""
    if re.match(r"^[A-Z]\*\d{2}:\d{2}:\d{2}:\d{2}$", h):
        return h
    locus, group, allele = _parse_hla_to_2field(h)
    if locus is None:
        return h
    if allele is None:
        allele = "01"
    return f"{locus}*{group}:{allele}:01:01"

# -------------------------
# epiTCR pseudo34 map
# -------------------------

def _pseudo34_from_full(seq: str) -> str:
    seq = norm_seq(seq)
    if len(seq) < 34:
        return ""
    return seq[:34]

def _header_to_2field_key(header: str) -> str:
    s = safe_str(header)
    m = re.search(r"(HLA-[A-Z]\*\d{2}:\d{2})", s)
    if m:
        k = m.group(1)
        locus, group, allele = _parse_hla_to_2field(k)
        if locus:
            return f"HLA-{locus}*{group}:{allele or '01'}"
    return ""

def _parse_any_fasta_like(path: Path):
    recs = []
    header = None
    seq = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                recs.append((header, "".join(seq)))
            header = line[1:].strip()
            seq = []
        else:
            seq.append(line)
    if header is not None:
        recs.append((header, "".join(seq)))
    return recs

def load_hla_pseudo34_map(hla_pseudo_csv: Path | None, hla_prot_txt: Path | None):
    mp = {}
    if hla_pseudo_csv and hla_pseudo_csv.exists():
        df = pd.read_csv(hla_pseudo_csv)
        c1 = pick_col(df, "HLA_allele") or pick_col(df, "allele")
        c2 = pick_col(df, "HLA_pseudo34") or pick_col(df, "pseudo34") or pick_col(df, "MHC34") or pick_col(df, "MHC")
        if c1 and c2:
            for _, r in df.iterrows():
                k = safe_str(r[c1])
                v = safe_str(r[c2])
                if k and len(v) == 34 and k not in mp:
                    mp[k] = v
        return mp

    if hla_prot_txt and hla_prot_txt.exists():
        recs = _parse_any_fasta_like(hla_prot_txt)
        for header, seq in recs:
            k = _header_to_2field_key(header)
            if not k or k in mp:
                continue
            pseudo = _pseudo34_from_full(seq)
            if len(pseudo) == 34:
                mp[k] = pseudo
        return mp

    return mp

def resolve_default_hla34_csv(user_arg: str) -> Path | None:
    if user_arg:
        return Path(user_arg)
    script_dir = Path(__file__).resolve().parent
    auto = script_dir / "HLA-34position.csv"
    return auto if auto.exists() else None

# -------------------------
# ANARCI
# -------------------------
ANARCI_OK = True
try:
    from anarci import anarci as anarci_run
except Exception:
    ANARCI_OK = False
    anarci_run = None

IMGT_CDR_RANGES = {"cdr1": (27, 38), "cdr2": (56, 65), "cdr3": (105, 117)}

def _cdr_from_numbering(numbering, start_pos, end_pos):
    out = []
    for (pos, _ins), aa in numbering:
        if pos is None:
            continue
        if start_pos <= pos <= end_pos and aa not in (None, "-", " "):
            out.append(aa)
    return "".join(out)

def _extract_numbering_list_from_numbered(numbered):
    if not numbered or not numbered[0]:
        return None
    dom0 = numbered[0][0]
    if isinstance(dom0, tuple) and len(dom0) >= 1 and isinstance(dom0[0], list):
        if dom0[0] and isinstance(dom0[0][0], tuple) and len(dom0[0][0]) == 2:
            return dom0[0]
    if isinstance(dom0, tuple) and len(dom0) >= 2 and isinstance(dom0[1], list):
        return dom0[1]
    if isinstance(dom0, list):
        return dom0
    for x in numbered[0]:
        if isinstance(x, tuple):
            if len(x) >= 1 and isinstance(x[0], list):
                return x[0]
            if len(x) >= 2 and isinstance(x[1], list):
                return x[1]
        if isinstance(x, list):
            return x
    return None

def extract_imgt_cdrs_and_vj(seq: str):
    seq = norm_seq(seq)
    if not seq:
        return {"cdr1": "", "cdr2": "", "cdr3": "", "v_gene": "", "j_gene": ""}, False, "empty_sequence"
    if not ANARCI_OK:
        return {"cdr1": "", "cdr2": "", "cdr3": "", "v_gene": "", "j_gene": ""}, False, "anarci_not_installed"
    try:
        numbered, details, _ = anarci_run([("q", seq)], scheme="imgt", assign_germline=True)
        numbering = _extract_numbering_list_from_numbered(numbered)
        if numbering is None:
            return {"cdr1": "", "cdr2": "", "cdr3": "", "v_gene": "", "j_gene": ""}, False, "anarci_no_numbering"

        cdr1 = _cdr_from_numbering(numbering, *IMGT_CDR_RANGES["cdr1"])
        cdr2 = _cdr_from_numbering(numbering, *IMGT_CDR_RANGES["cdr2"])
        cdr3 = _cdr_from_numbering(numbering, *IMGT_CDR_RANGES["cdr3"])

        v_gene = ""
        j_gene = ""
        try:
            if details and details[0] and isinstance(details[0][0], dict):
                germlines = details[0][0].get("germlines", {})
                v_gene = safe_str(germlines.get("v_gene", [("", ""), 0])[0][1])
                j_gene = safe_str(germlines.get("j_gene", [("", ""), 0])[0][1])
        except Exception:
            pass

        ok = True if cdr3 else False
        return {"cdr1": cdr1, "cdr2": cdr2, "cdr3": cdr3, "v_gene": v_gene, "j_gene": j_gene}, ok, ("ok" if ok else "cdr3_empty")
    except Exception as e:
        return {"cdr1": "", "cdr2": "", "cdr3": "", "v_gene": "", "j_gene": ""}, False, f"anarci_error:{e}"

# -------------------------
# Example-column checker
# -------------------------
def assert_columns_match(df_out: pd.DataFrame, example_path: Path, out_name: str):
    if not example_path.exists():
        return
    if example_path.suffix.lower() == ".xlsx":
        return
    try:
        df_ex = pd.read_csv(example_path)
        if list(df_out.columns) != list(df_ex.columns):
            raise SystemExit(f"[ERROR] {out_name} columns mismatch.\nOut: {list(df_out.columns)}\nEx: {list(df_ex.columns)}")
    except Exception:
        return

# =============================================================================
# ✅ MixTCRpred helpers
# =============================================================================

def hla_to_mixtcr_tag(h: str) -> str:
    """
    Convert HLA string to MixTCRpred ckpt tag:
      HLA-A*02:01 -> A0201
      A*02:01     -> A0201
      HLA-A02:01  -> A0201
      A0201       -> A0201
    """
    h = safe_str(h).upper().replace("HLA-", "").replace("_", "").replace(" ", "")
    if not h:
        return ""

    # A*02:01 / A02:01 / A*02 / A02
    m = re.match(r"^([A-Z])\*?(\d{2})(?::?(\d{2}))?$", h)
    if m:
        locus, g1, g2 = m.group(1), m.group(2), m.group(3) or "01"
        return f"{locus}{g1}{g2}"

    # A0201
    m = re.match(r"^([A-Z])(\d{2})(\d{2})$", h)
    if m:
        return f"{m.group(1)}{m.group(2)}{m.group(3)}"

    # A*02:01:01:01 -> take first two fields
    m = re.match(r"^([A-Z])\*?(\d{2}):(\d{2})", h)
    if m:
        return f"{m.group(1)}{m.group(2)}{m.group(3)}"

    return ""

def mixtcr_normalize_models_dir(models_dir: str) -> str:
    """
    Ensure models_dir points to MixTCRpred-main/pretrained_models.
    Accepts:
      .../MixTCRpred-main
      .../MixTCRpred-main/models
      .../MixTCRpred-main/pretrained_models
    """
    p = Path(models_dir) if models_dir else None
    if not p:
        return ""

    if p.is_file():
        return str(p.parent)

    if p.name.lower() == "pretrained_models":
        return str(p)

    if p.name.lower() == "models":
        cand = p.parent / "pretrained_models"
        if cand.exists():
            return str(cand)
        return str(p)

    if p.name.lower() == "mixtcrpred-main":
        cand = p / "pretrained_models"
        if cand.exists():
            return str(cand)
        return str(p)

    # if user passes an arbitrary dir, try child pretrained_models
    cand = p / "pretrained_models"
    if cand.exists():
        return str(cand)

    return str(p)

def mixtcr_load_index(info_csv: str, models_dir: str):
    """
    Read info_models.csv and build a set of available model keys.
    Supports both:
      - MixTCRpred_model_name like "A0201_CINGVCWTV"
      - or (MHC,Peptide) columns, where MHC like "HLA-A*02:01"
    Returns:
      info_path_used, set_of_names, diagnostics(list[str])
    """
    diag = []
    md = Path(models_dir) if models_dir else None

    # resolve info_models.csv path
    info_path = Path(info_csv) if info_csv else None
    if (info_path is None) or (not info_path.exists()):
        if md and md.exists():
            auto = md / "info_models.csv"
            if auto.exists():
                info_path = auto

    diag.append(f"[MixTCRpred][MODEL-INDEX] models_dir={models_dir}")
    diag.append(f"[MixTCRpred][MODEL-INDEX] info_csv_arg={info_csv or ''}")
    diag.append(f"[MixTCRpred][MODEL-INDEX] resolved_info_csv={str(info_path) if info_path else ''}")

    if info_path is None or (not info_path.exists()):
        diag.append("[MixTCRpred][MODEL-INDEX] ERROR: info_models.csv not found")
        return "", set(), diag

    try:
        info = pd.read_csv(info_path)
    except Exception as e:
        diag.append(f"[MixTCRpred][MODEL-INDEX] ERROR: failed to read info_models.csv: {e}")
        return str(info_path), set(), diag

    diag.append(f"[MixTCRpred][MODEL-INDEX] rows={len(info)} cols={list(info.columns)}")

    names = set()

    # Prefer MixTCRpred_model_name if present
    if "MixTCRpred_model_name" in info.columns:
        for v in info["MixTCRpred_model_name"].tolist():
            s = safe_str(v).upper()
            if s:
                names.add(s)

    # Fallback: build from MHC + Peptide
    if (not names) and ("MHC" in info.columns) and ("Peptide" in info.columns):
        for mhc, pep in zip(info["MHC"].tolist(), info["Peptide"].tolist()):
            tag = hla_to_mixtcr_tag(mhc)
            p = safe_str(pep).upper()
            if tag and p:
                names.add(f"{tag}_{p}")

    diag.append(f"[MixTCRpred][MODEL-INDEX] unique_model_keys={len(names)}")
    return str(info_path), names, diag

def mixtcr_ckpt_path(models_dir: str, model_key: str) -> str:
    """
    model_key: 'A0201_CINGVCWTV'
    ckpt:      'model_A0201_CINGVCWTV.ckpt'
    """
    md = Path(models_dir) if models_dir else None
    if md is None or (not md.exists()):
        return ""
    key = safe_str(model_key).upper()
    if not key:
        return ""
    p = md / f"model_{key}.ckpt"
    return str(p) if p.exists() else ""

# -------------------------
# MAIN
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--fill_label", default=0, type=int)

    ap.add_argument("--hla_pseudo_csv", default="", help="Optional pseudo34 CSV. If empty, will auto-use backend/HLA-34position.csv when present.")
    ap.add_argument("--hla_prot_txt", default="", help="Optional hla_prot.fasta.txt (legacy fallback if no csv)")

    ap.add_argument("--check_examples", default=1, type=int)
    ap.add_argument("--example_dir", default="", help="dir containing *_example.* files")

    ap.add_argument("--mode", required=True, choices=["cdr3", "vregion"])

    # ✅ MixTCRpred args (keep compatibility with backend)
    ap.add_argument("--mixt_models_dir", default="", help="MixTCRpred dir (will auto-resolve to pretrained_models)")
    ap.add_argument("--mixt_info_csv", default="", help="Path to info_models.csv (optional; if empty will use <models_dir>/info_models.csv)")

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ex_dir = Path(args.example_dir) if args.example_dir else out_dir

    df = pd.read_csv(args.in_csv)

    # Stable pair id from master_input.csv (preferred). If absent, fallback to row index.
    col_pid = pick_col(df, "pair_id") or pick_col(df, "PairID") or pick_col(df, "id")
    pid_ser = df[col_pid].astype(int) if col_pid else pd.Series(range(len(df)), dtype=int)

    col_a = pick_col(df, "TCR_alpha")
    col_b = pick_col(df, "TCR_beta")
    col_p = pick_col(df, "Peptide")
    col_m = pick_col(df, "MHC")

    missing = [x for x, c in [("TCR_alpha", col_a), ("TCR_beta", col_b), ("Peptide", col_p), ("MHC", col_m)] if c is None]
    if args.mode == "cdr3":
        if col_b is None or col_p is None or col_m is None:
            raise SystemExit(f"[ERROR] Missing required column(s) for cdr3 mode. Need TCR_beta + Peptide + MHC. Found: {list(df.columns)}")
        if col_a is None:
            df["TCR_alpha"] = ""
            col_a = "TCR_alpha"
    else:
        if missing:
            raise SystemExit(f"[ERROR] Missing required column(s): {missing}. Found columns: {list(df.columns)}")

    peptide = df[col_p].map(safe_str)
    peptide_raw = peptide
    mhc_raw = df[col_m].map(safe_str)

    mhc_piste = mhc_raw.map(hla_for_piste)
    mhc_pmtnet = mhc_raw.map(hla_for_pmtnet)
    mhc_nettcr = mhc_raw.map(hla_for_nettcr)
    mhc_ergo = mhc_raw.map(hla_for_ergo)

    # epiTCR pseudo34
    pseudo_csv_path = resolve_default_hla34_csv(args.hla_pseudo_csv)
    prot_txt_path = Path(args.hla_prot_txt) if args.hla_prot_txt else None

    hla_map = load_hla_pseudo34_map(
        pseudo_csv_path if (pseudo_csv_path and pseudo_csv_path.exists()) else None,
        prot_txt_path if (prot_txt_path and prot_txt_path.exists()) else None
    )

    if not hla_map:
        print(
            "[WARN] HLA pseudo34 map is EMPTY. epiTCR MHC34 may be blank. "
            "Tip: put HLA-34position.csv next to this script (backend/) or pass --hla_pseudo_csv.",
            file=sys.stderr
        )
    else:
        src = str(pseudo_csv_path) if (pseudo_csv_path and pseudo_csv_path.exists()) else (str(prot_txt_path) if prot_txt_path else "unknown")
        print(f"[INFO] Loaded HLA pseudo34 map: n={len(hla_map)} from {src}", file=sys.stderr)

    epitcr_hla_4field = mhc_raw.map(hla_for_epitcr_4field_no_prefix)
    epitcr_hla_key2 = mhc_raw.map(hla_key_2field_with_prefix)
    epitcr_mhc34 = [hla_map.get(k, "") for k in epitcr_hla_key2.tolist()]

    A1, A2, A3 = [], [], []
    B1, B2, B3 = [], [], []
    TRAV, TRAJ, TRBV, TRBJ = [], [], [], []
    failed_records = []

    for idx, row in df.iterrows():
        a_seq = norm_seq(row[col_a])
        b_seq = norm_seq(row[col_b])

        if args.mode == "cdr3":
            a = {"cdr1": "", "cdr2": "", "cdr3": a_seq, "v_gene": "", "j_gene": ""}
            b = {"cdr1": "", "cdr2": "", "cdr3": b_seq, "v_gene": "", "j_gene": ""}
            ok_a = True
            ok_b = True if b_seq else False
            reason_a, reason_b = "cdr3_direct", ("ok" if ok_b else "empty_beta_cdr3")
        else:
            a, ok_a, reason_a = extract_imgt_cdrs_and_vj(a_seq)
            b, ok_b, reason_b = extract_imgt_cdrs_and_vj(b_seq)

        A1.append(a["cdr1"]); A2.append(a["cdr2"]); A3.append(a["cdr3"])
        B1.append(b["cdr1"]); B2.append(b["cdr2"]); B3.append(b["cdr3"])
        TRAV.append(clean_gene_call(a["v_gene"]))
        TRAJ.append(clean_gene_call(a["j_gene"]))
        TRBV.append(clean_gene_call(b["v_gene"]))
        TRBJ.append(clean_gene_call(b["j_gene"]))

        reasons = []
        if not ok_a and args.mode != "cdr3":
            reasons.append(f"alpha:{reason_a}")
        if not ok_b:
            reasons.append(f"beta:{reason_b}")
        if epitcr_mhc34[idx] == "":
            k2 = epitcr_hla_key2.iloc[idx]
            reasons.append(f"epiTCR:MHC34_missing_for:{k2 or safe_str(mhc_raw.iloc[idx])}")

        if reasons:
            rec = row.to_dict()
            rec["failed_reason"] = ";".join(reasons)
            rec["row_index"] = idx
            failed_records.append(rec)

    
    # -------------------------
    # Write CDR_master.csv (same row order as master_input.csv)
    # This is used by backend/app.py to display per-CDR columns in vregion mode.
    # -------------------------
    cdr_master = pd.DataFrame({
        "pair_id": pid_ser.tolist(),
        "beta_CDR1": B1,
        "beta_CDR2": B2,
        "beta_CDR3": B3,
        "alpha_CDR1": A1,
        "alpha_CDR2": A2,
        "alpha_CDR3": A3,
    })
    cdr_master["beta_CDR123"] = cdr_master["beta_CDR1"].astype(str) + cdr_master["beta_CDR2"].astype(str) + cdr_master["beta_CDR3"].astype(str)
    cdr_master["alpha_CDR123"] = cdr_master["alpha_CDR1"].astype(str) + cdr_master["alpha_CDR2"].astype(str) + cdr_master["alpha_CDR3"].astype(str)
    cdr_master.to_csv(out_dir / "CDR_master.csv", index=False)

# -------------------------
    # peptide length masks
    # -------------------------
    pep_len = peptide.astype(str).str.len().fillna(0).astype(int)
    b3_len_for_masks = pd.Series(B3, dtype="string").fillna("").str.len()
    mask_teim = (pep_len >= 8) & (pep_len <= 12) & (b3_len_for_masks <= 20)
    mask_piste = (pep_len >= 8) & (pep_len <= 11)
    mask_itcep = (pep_len >= 8) & (pep_len <= 11)
    mask_pep_8_11 = (pep_len >= 8) & (pep_len <= 11)

    a3_ser = pd.Series(A3, dtype="string").fillna("")
    b3_ser = pd.Series(B3, dtype="string").fillna("")
    pep_ser = peptide.astype("string").fillna("")
    allele_ser = mhc_nettcr.astype("string").fillna("")
    a3_len = a3_ser.str.len()
    b3_len = b3_ser.str.len()

    mask_nettcr_nonempty = (a3_ser.str.len() > 0) & (b3_ser.str.len() > 0) & (pep_ser.str.len() > 0) & (allele_ser.str.len() > 0) & mask_pep_8_11
    mhc_ser = mhc_raw.astype("string").fillna("")
    mask_epitcr = (pep_len >= 8) & (pep_len <= 11) & (b3_len_for_masks <= 19)
    mask_imrex = (b3_len >= 10) & (b3_len <= 20) & mask_pep_8_11
    mask_tulip = (pep_ser.str.len() > 0) & (pep_ser.str.len() <= 48) & (a3_len <= 48) & (b3_len <= 48)

    # -------------------------
    # Write outputs (other methods unchanged)
    # -------------------------
    ergo = pd.DataFrame({
        "TRA": df[col_a].map(norm_seq),
        "TRB": df[col_b].map(norm_seq),
        "TRAV": TRAV,
        "TRAJ": TRAJ,
        "TRBV": TRBV,
        "TRBJ": TRBJ,
        "T-Cell-Type": ["" for _ in range(len(df))],
        "Peptide": peptide,
        "MHC": mhc_ergo
    })
    ergo.to_csv(out_dir / "ERGO-II_input.csv", index=False)
    write_rowmap(out_dir, "ERGO-II", pid_ser)

    panpep = pd.DataFrame({"Peptide": peptide, "CDR3": B3})
    panpep.to_csv(out_dir / "Panpep_input.csv", index=False)
    write_rowmap(out_dir, "PanPep", pid_ser)

    piste = pd.DataFrame({"CDR3": B3, "MT_pep": peptide, "HLA_type": mhc_piste}).loc[mask_piste].copy()
    piste.to_csv(out_dir / "PISTE_input.csv", index=False)
    write_rowmap(out_dir, "PISTE", pid_ser, kept_mask=mask_piste)

    pmtnet = pd.DataFrame({"CDR3": B3, "Antigen": peptide, "HLA": mhc_pmtnet})
    pmtnet.to_csv(out_dir / "pMTnet_input.csv", index=False)
    write_rowmap(out_dir, "pMTnet", pid_ser)

    teinet = pd.DataFrame({"CDR3.beta": B3, "Epitope": peptide, "Label": [args.fill_label] * len(df)})
    teinet.to_csv(out_dir / "TEINet_input.csv", index=False)
    write_rowmap(out_dir, "TEINet", pid_ser)

    bertrand = pd.DataFrame({"CDR3b": B3, "peptide_seq": peptide})
    bertrand.to_csv(out_dir / "bertrand_input.csv", index=False)
    write_rowmap(out_dir, "bertrand", pid_ser)

    epitcr = pd.DataFrame({
        "CDR3b": B3,
        "epitope": peptide,
        "HLA": epitcr_hla_4field,
        "binder": [args.fill_label] * len(df),
        "MHC": epitcr_mhc34
    }).loc[mask_epitcr].copy()
    epitcr.to_csv(out_dir / "epiTCR_input.csv", index=False)
    write_rowmap(out_dir, "epiTCR", pid_ser, kept_mask=mask_epitcr)

    nettcr = pd.DataFrame({
        "index": list(range(len(df))),
        "A1": A1, "A2": A2, "A3": A3,
        "B1": B1, "B2": B2, "B3": B3,
        "peptide": peptide,
        "allele": mhc_nettcr
    }).loc[mask_nettcr_nonempty].copy()
    nettcr.to_csv(out_dir / "NetTCR_input.csv", index=False)
    write_rowmap(out_dir, "NetTCR-2.2", pid_ser, kept_mask=mask_nettcr_nonempty)

    teim = pd.DataFrame({"cdr3": B3, "epitope": peptide}).loc[mask_teim].copy()
    teim.to_csv(out_dir / "TEIM_input.csv", index=False)
    write_rowmap(out_dir, "TEIM", pid_ser, kept_mask=mask_teim)

    itcep_cdr3 = pd.Series(B3, dtype=str).fillna("").str.strip().str.upper()
    itcep_cdr3 = itcep_cdr3.apply(lambda s: ("C" + s) if (s and not s.startswith("C")) else s)
    itcep_cdr3 = itcep_cdr3.apply(lambda s: (s + "F") if (s and not s.endswith(("F", "W"))) else s)

    itcep = pd.DataFrame({"peptide": peptide, "CDR3": itcep_cdr3}).loc[mask_itcep].copy()
    itcep = itcep[itcep["CDR3"] != ""].copy()
    itcep.to_csv(out_dir / "iTCep_input.csv", index=False)
    write_rowmap(out_dir, "iTCep", pid_ser, kept_mask=mask_itcep)

    atmtcr = pd.DataFrame({
        "Epitope": peptide,
        "TCR": B3,
        "Binding Affinity": [args.fill_label] * len(df),
    })
    atmtcr.to_csv(out_dir / "ATM-TCR_input.csv", index=False)
    write_rowmap(out_dir, "ATM-TCR", pid_ser)

    imrex = pd.DataFrame({
        "cdr3": B3,
        "antigen.epitope": peptide,
    }).loc[mask_imrex].copy()
    imrex.to_csv(out_dir / "ImRex_input.csv", index=False, sep=";")
    write_rowmap(out_dir, "ImRex", pid_ser, kept_mask=mask_imrex)

    tepcam = pd.DataFrame({
        "TCR": B3,
        "epitope": peptide,
        "Label": [args.fill_label] * len(df),
    })
    tepcam.to_csv(out_dir / "TEPCAM_input.csv", index=False)
    write_rowmap(out_dir, "TEPCAM", pid_ser)

    tulip_alpha = pd.Series(A3, dtype="string").fillna("").map(lambda x: safe_str(x) or "<MIS>")
    tulip_beta = pd.Series(B3, dtype="string").fillna("").map(lambda x: safe_str(x) or "<MIS>")
    tulip = pd.DataFrame({
        "CDR3a": tulip_alpha,
        "CDR3b": tulip_beta,
        "MHC": mhc_raw.map(lambda x: safe_str(x) or "<MIS>"),
        "peptide": peptide,
        "binder": [args.fill_label] * len(df),
    }).loc[mask_tulip].copy()
    tulip.to_csv(out_dir / "TULIP_input.csv", index=False)
    write_rowmap(out_dir, "TULIP", pid_ser, kept_mask=mask_tulip)

    # =============================================================================
    # ✅ MixTCRpred（核心修改）
    #   1) 使用 info_models.csv 判断模型是否存在（key = A0201_<PEPTIDE>）
    #   2) ckpt 路径固定从 pretrained_models 下读取：model_<A0201>_<PEPTIDE>.ckpt
    #   3) 输出 input/plan/og + logs
    # =============================================================================

    mixt_models_dir = mixtcr_normalize_models_dir(args.mixt_models_dir)
    info_path_used, model_keys, model_diag = mixtcr_load_index(args.mixt_info_csv, mixt_models_dir)

    # MixTCRpred 的输入表（只包含 TCR 信息；模型选择在 plan 里）
    mixtcr_all = pd.DataFrame({
        "cdr3_TRA": A3,
        "cdr3_TRB": B3,
        "TRAV": TRAV,
        "TRAJ": TRAJ,
        "TRBV": TRBV,
        "TRBJ": TRBJ
    })

    keep_idx = []
    keep_model_key = []
    keep_model_name = []   # ✅ 新增：用于 --model 参数
    keep_ckpt = []

    mixtcr_trace = []

    models_dir_exists = bool(mixt_models_dir) and Path(mixt_models_dir).exists()

    for i in range(len(df)):
        pep_i = safe_str(peptide.iloc[i]).upper()
        mhc_i = safe_str(mhc_raw.iloc[i])
        tag_i = hla_to_mixtcr_tag(mhc_i)
        key_i = f"{tag_i}_{pep_i}" if tag_i and pep_i else ""

        a3_i = safe_str(A3[i])
        b3_i = safe_str(B3[i])

        reasons = []
        status = "filtered"
        ckpt_path = ""

        # 1) 基础字段检查（需要 alpha+beta CDR3）
        if not a3_i:
            reasons.append("missing_alpha_cdr3")
        if not b3_i:
            reasons.append("missing_beta_cdr3")
        if not pep_i:
            reasons.append("missing_peptide")
        if not mhc_i:
            reasons.append("missing_mhc")

        # 2) 肽长度（MixTCRpred: 8-11）
        L = int(pep_len.iloc[i]) if i < len(pep_len) else len(pep_i)
        if not (8 <= L <= 11):
            reasons.append(f"peptide_len_out_of_range:{L} (need 8-11)")

        # 3) HLA 能否转 tag
        if mhc_i and not tag_i:
            reasons.append(f"mhc_format_unrecognized:{mhc_i} (cannot convert to A0201)")

        # 4) info_models.csv 是否可用
        if not reasons and not model_keys:
            reasons.append("model_index_empty(info_models_missing_or_empty_or_bad_columns)")

        # 5) 是否在 info_models.csv 中存在该模型 key
        if not reasons and (key_i not in model_keys):
            reasons.append(f"no_model_in_info_models:{key_i}")

        # 6) ckpt 是否存在
        exp_ckpt = f"model_{key_i}.ckpt" if key_i else ""
        if not reasons:
            if not models_dir_exists:
                reasons.append(f"models_dir_not_found:{mixt_models_dir}")
            else:
                ckpt_path = mixtcr_ckpt_path(mixt_models_dir, key_i)
                if not ckpt_path:
                    reasons.append(f"ckpt_not_found_expected:{exp_ckpt}")
                else:
                    status = "written"
                    keep_idx.append(int(i))
                    keep_model_key.append(key_i)
                    keep_model_name.append(key_i)   # ✅ 关键：--model 用 A0201_<PEP>
                    keep_ckpt.append(ckpt_path)

        mixtcr_trace.append({
            "row_index": int(i),
            "status": status,
            "filter_reason": ";".join(reasons),
            "Peptide": safe_str(peptide.iloc[i]),
            "MHC": mhc_i,
            "hla_tag": tag_i,
            "model_key": key_i,
            "expected_ckpt": exp_ckpt,
            "ckpt_path": ckpt_path,
            "cdr3_TRA": a3_i,
            "cdr3_TRB": b3_i,
            "TRAV": safe_str(TRAV[i]),
            "TRAJ": safe_str(TRAJ[i]),
            "TRBV": safe_str(TRBV[i]),
            "TRBJ": safe_str(TRBJ[i]),
        })

    # input.csv：只写 kept 行（顺序与 plan.pos 对应）
    mixtcr_out = mixtcr_all.loc[keep_idx].copy() if keep_idx else mixtcr_all.iloc[0:0].copy()
    mixtcr_out.to_csv(out_dir / "MixTCRpred_input.csv", index=False)
    write_rowmap(out_dir, "MixTCRpred", pid_ser, kept_indices=keep_idx)

        # plan.csv：runner 需要 columns: row_index, model
    # 这里的 model 直接用 ckpt_path（因为 cmd_tpl 里是 --model {model}）
    plan_df = pd.DataFrame({
      "row_index": keep_idx,
      "model": keep_model_name,    # ✅ 对：这里是 A0201_<PEP>，给 MixTCRpred.py --model 用
      "model_key": keep_model_key,
      "ckpt_path": keep_ckpt,      # 仍保留，方便你排查 ckpt 是否存在
      "pos": list(range(len(keep_idx))),
    })
    plan_df.to_csv(out_dir / "MixTCRpred_plan.csv", index=False)

    # og.csv：每条 pair 都保留
    og_df = pd.DataFrame(mixtcr_trace)
    og_df.to_csv(out_dir / "MixTCRpred_og.csv", index=False)

    # logs（append）
    stdout_log = out_dir / "MixTCRpred.stdout.log"
    with stdout_log.open("a", encoding="utf-8") as f:
        f.write("================================================================================\n")
        f.write("[MixTCRpred][INPUT-GEN] Trace why rows are skipped (not written to MixTCRpred_input.csv)\n")
        f.write(f"[MixTCRpred][INPUT-GEN] total_rows={len(df)}\n")
        f.write("[MixTCRpred][INPUT-GEN] peptide_len_allowed=[8,11]\n")
        f.write(f"[MixTCRpred][INPUT-GEN] kept_rows={len(keep_idx)} skipped_rows={len(df)-len(keep_idx)}\n")
        for line in model_diag:
            f.write(line + "\n")
        f.write(f"[MixTCRpred][INPUT-GEN] normalized_models_dir={mixt_models_dir}\n")
        f.write(f"[MixTCRpred][INPUT-GEN] info_models_used={info_path_used}\n")
        if models_dir_exists:
            n_ckpt = len(list(Path(mixt_models_dir).glob("model_*.ckpt")))
            f.write(f"[MixTCRpred][INPUT-GEN] found_ckpt_files={n_ckpt} (pattern=model_*.ckpt)\n")
        else:
            f.write("[MixTCRpred][INPUT-GEN] models_dir_not_found -> ckpt matching will fail\n")
        for r in mixtcr_trace:
            if r["status"] != "written":
                f.write(
                    f"[MixTCRpred][SKIP] row_index={r['row_index']} peptide={r['Peptide']} mhc={r['MHC']} "
                    f"A3={r['cdr3_TRA']} B3={r['cdr3_TRB']} reason={r['filter_reason']}\n"
                )
        f.write(f"[MixTCRpred][INPUT-GEN] wrote_input_csv={str((out_dir / 'MixTCRpred_input.csv').resolve())}\n")
        f.write(f"[MixTCRpred][INPUT-GEN] wrote_plan_csv={str((out_dir / 'MixTCRpred_plan.csv').resolve())}\n")
        f.write(f"[MixTCRpred][INPUT-GEN] wrote_og_csv={str((out_dir / 'MixTCRpred_og.csv').resolve())}\n")

    reason_log = out_dir / "MixTCRpred.filter_reason.log"
    with reason_log.open("a", encoding="utf-8") as f:
        f.write("================================================================================\n")
        f.write("[MixTCRpred][FILTER] row-level reasons\n")
        for r in mixtcr_trace:
            if r["status"] != "written":
                f.write(f"row_index={r['row_index']}\t{r['filter_reason']}\t{r['model_key']}\t{r['expected_ckpt']}\n")

    # -------------------------
    # failed rows
    # -------------------------
    failed_path = out_dir / "failed_rows.csv"
    if failed_records:
        pd.DataFrame(failed_records).to_csv(failed_path, index=False)
        print(f"[WARN] Some rows failed: {len(failed_records)}. See: {failed_path}", file=sys.stderr)
    else:
        pd.DataFrame(columns=list(df.columns) + ["row_index", "failed_reason"]).to_csv(failed_path, index=False)

    # example check（可选，保持原样）
    if args.check_examples:
        assert_columns_match(ergo,       ex_dir / "ERGO-II_example.csv",       "ERGO-II_input.csv")
        assert_columns_match(panpep,     ex_dir / "Panpep_example.csv",        "Panpep_input.csv")
        assert_columns_match(piste,      ex_dir / "PISTE_example.csv",         "PISTE_input.csv")
        assert_columns_match(pmtnet,     ex_dir / "pMTnet_example.csv",        "pMTnet_input.csv")
        assert_columns_match(teinet,     ex_dir / "TEINet_example.csv",        "TEINet_input.csv")
        assert_columns_match(bertrand,   ex_dir / "bertrand_example.csv",      "bertrand_input.csv")
        assert_columns_match(epitcr,     ex_dir / "epiTCR_example.csv",        "epiTCR_input.csv")
        assert_columns_match(nettcr,     ex_dir / "NetTCR_example.csv",        "NetTCR_input.csv")
        assert_columns_match(teim,       ex_dir / "TEIM_example.csv",          "TEIM_input.csv")
        assert_columns_match(mixtcr_out, ex_dir / "MixTCRpred_example.csv",    "MixTCRpred_input.csv")

    print(f"[OK] Wrote outputs to: {out_dir.resolve()}")
    print(f"[OK] MixTCRpred normalized_models_dir: {mixt_models_dir}")
    print(f"[OK] MixTCRpred info_models_used: {info_path_used}")
    print(f"[OK] MixTCRpred input: {(out_dir / 'MixTCRpred_input.csv').resolve()}")
    print(f"[OK] MixTCRpred plan : {(out_dir / 'MixTCRpred_plan.csv').resolve()}")
    print(f"[OK] MixTCRpred og   : {(out_dir / 'MixTCRpred_og.csv').resolve()}")

if __name__ == "__main__":
    main()
