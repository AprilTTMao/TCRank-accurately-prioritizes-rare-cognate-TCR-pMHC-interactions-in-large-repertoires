# backend/job_runner.py
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from ranking_helper import build_ranking_payload
from output_matching import attach_ids

RUN_DIR_RE = re.compile(r"^job_[a-f0-9\-]{8,}$")


@dataclass
class JobPaths:
    job_dir: Path
    logs_dir: Path
    input_dir: Path
    output_dir: Path
    tmp_dir: Path  # per-job tmp files (master_input, etc.)

    @staticmethod
    def for_job_dir(job_dir: Path) -> "JobPaths":
        logs_dir = job_dir / "logs"
        input_dir = job_dir / "inputfiles"
        output_dir = job_dir / "outputs"
        tmp_dir = job_dir / "tmp"
        for p in (logs_dir, input_dir, output_dir, tmp_dir):
            p.mkdir(parents=True, exist_ok=True)
        return JobPaths(job_dir=job_dir, logs_dir=logs_dir, input_dir=input_dir, output_dir=output_dir, tmp_dir=tmp_dir)


# =========================
# JSON helpers (atomic)
# =========================
def _atomic_write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def update_status(job_dir: Path, **fields: Any) -> None:
    """
    Update jobs/<job_id>/status.json
    """
    st_path = job_dir / "status.json"
    st = _read_json(st_path, default={}) or {}
    st.update(fields)
    st["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    _atomic_write_json(st_path, st)


def append_worker_log(paths: JobPaths, msg: str) -> None:
    try:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        log_path = paths.logs_dir / "worker.log"
        log_path.open("a", encoding="utf-8", errors="ignore").write(f"[{ts}] {msg}\n")
        lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        update_status(paths.job_dir, runtime_log="\n".join(lines[-80:]))
    except Exception:
        pass


def update_method_status(job_dir: Path, methods: List[str], method: str, status: str, reason: str = "") -> None:
    st_path = job_dir / "status.json"
    st = _read_json(st_path, default={}) or {}
    current = st.get("method_status")
    if not isinstance(current, dict):
        current = {m: {"status": "queued", "reason": ""} for m in methods}
    for m in methods:
        current.setdefault(m, {"status": "queued", "reason": ""})
    current[method] = {"status": status, "reason": reason}
    update_status(job_dir, method_status=current, current_method=method)


# =========================
# Config + command helpers
# =========================
def load_cfg(cfg_path: Path) -> Dict[str, Any]:
    if not cfg_path.exists():
        raise FileNotFoundError(f"tools_config.json not found: {cfg_path}")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    if "methods" not in cfg or not isinstance(cfg["methods"], list):
        raise ValueError("tools_config.json must contain: methods (list)")
    m = {}
    for it in cfg["methods"]:
        if isinstance(it, dict) and str(it.get("name", "")).strip():
            m[str(it["name"]).strip()] = it
    cfg["_methods_map"] = m
    return cfg


def _render(tpl: str, ctx: Dict[str, str]) -> str:
    out = tpl
    for k, v in ctx.items():
        out = out.replace("{" + k + "}", v)
    return out


def _run_bash(cmd: str, timeout_s: int = 1800) -> Tuple[int, str, str]:
    p = subprocess.run(["bash", "-lc", cmd], capture_output=True, text=True, timeout=timeout_s)
    return p.returncode, p.stdout or "", p.stderr or ""


def _method_worker_count(total: int) -> int:
    if total <= 1:
        return max(1, total)
    raw = os.environ.get("TCRANK_WEB_METHOD_WORKERS", "4")
    try:
        requested = int(raw)
    except Exception:
        requested = 4
    requested = max(1, requested)
    return min(total, requested)


def _safe_write_text(path: Path, text: str) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text or "", encoding="utf-8", errors="ignore")
    except Exception:
        pass


def _dump_run_logs(log_dir: Path, prefix: str, cmd: str, rc: int, so: str, se: str, reason: str = "") -> None:
    _safe_write_text(log_dir / f"{prefix}.cmd.txt", (cmd or "") + "\n")
    _safe_write_text(log_dir / f"{prefix}.stdout.log", so or "")
    _safe_write_text(log_dir / f"{prefix}.stderr.log", se or "")
    if reason:
        _safe_write_text(log_dir / f"{prefix}.reason.txt", reason + "\n")


# =========================
# Key building / matching (same logic as your current backend)
# =========================
def _norm(x: Any) -> str:
    return "" if x is None else str(x).strip()


def _key_lvl1(beta: str, alpha: str, mhc: str, pep: str) -> str:
    return f"{beta}|{alpha}||{mhc}||{pep}"


def _key_lvl2(beta: str, mhc: str, pep: str) -> str:
    return f"{beta}||{mhc}||{pep}"


def _key_lvl3(beta: str, pep: str) -> str:
    return f"{beta}||{pep}"


def _key_lvl4(beta: str, mhc: str) -> str:
    return f"{beta}||{mhc}"


def _input_key_chain(beta: str, alpha: str, mhc: str, pep: str) -> List[str]:
    beta = _norm(beta)
    alpha = _norm(alpha)
    mhc = _norm(mhc)
    pep = _norm(pep)
    keys: List[str] = []
    if beta and alpha and mhc and pep:
        keys.append(_key_lvl1(beta, alpha, mhc, pep))
    if beta and mhc and pep:
        keys.append(_key_lvl2(beta, mhc, pep))
    if beta and mhc:
        keys.append(_key_lvl4(beta, mhc))
    if beta and pep:
        keys.append(_key_lvl3(beta, pep))
    return keys


def _input_key_chain_vregion(
    beta_cdr123: str,
    alpha_cdr123: str,
    beta_cdr3: str,
    alpha_cdr3: str,
    mhc: str,
    pep: str,
) -> List[str]:
    b123 = _norm(beta_cdr123)
    a123 = _norm(alpha_cdr123)
    b3 = _norm(beta_cdr3)
    a3 = _norm(alpha_cdr3)
    mhc = _norm(mhc)
    pep = _norm(pep)

    keys: List[str] = []
    if b123 and a123 and pep and mhc:
        keys.append(_key_lvl1(b123, a123, mhc, pep))
    if b3 and a3 and pep and mhc:
        keys.append(_key_lvl1(b3, a3, mhc, pep))
    if b123 and mhc and pep:
        keys.append(_key_lvl2(b123, mhc, pep))
    if b3 and mhc and pep:
        keys.append(_key_lvl2(b3, mhc, pep))
    if b3 and mhc:
        keys.append(_key_lvl4(b3, mhc))
    if b3 and pep:
        keys.append(_key_lvl3(b3, pep))

    seen = set()
    out: List[str] = []
    for k in keys:
        if k not in seen:
            seen.add(k)
            out.append(k)
    return out


def _output_key_chain(beta: str, alpha: str, mhc: str, pep: str) -> List[str]:
    beta = _norm(beta)
    alpha = _norm(alpha)
    mhc = _norm(mhc)
    pep = _norm(pep)
    keys: List[str] = []
    if beta and alpha and mhc and pep:
        keys.append(_key_lvl1(beta, alpha, mhc, pep))
    if beta and mhc and pep:
        keys.append(_key_lvl2(beta, mhc, pep))
    if beta and pep:
        keys.append(_key_lvl3(beta, pep))
    return keys


def _is_positive(val: float, thr: float, positive_if: str) -> bool:
    op = (positive_if or "gt").strip().lower()
    if op == "gt":
        return val > thr
    if op == "ge":
        return val >= thr
    if op == "lt":
        return val < thr
    if op == "le":
        return val <= thr
    return val > thr


# =========================
# Main job runner
# =========================
def run_job(
    *,
    backend_dir: Path,
    cfg_path: Path,
    runners_dir: Path,
    inputfiles_script: Path,
    job_dir: Path,
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Execute one prediction job.

    - Reads payload (mode, selected_methods, beta/alpha/peptide/mhc, optional csv_path)
    - Generates per-method input files under jobs/<id>/inputfiles/
    - Runs selected methods, writes outputs under jobs/<id>/outputs/
    - Writes result.json and returns the result dict
    """
    paths = JobPaths.for_job_dir(job_dir)
    mode = str(payload.get("mode", "")).strip().lower()
    methods: List[str] = payload.get("methods", []) or []

    if mode not in {"cdr3", "vregion"}:
        raise ValueError("mode must be cdr3 or vregion")
    if not methods:
        raise ValueError("methods is empty")

    update_status(
        job_dir,
        status="running",
        stage="init",
        progress=0.01,
        method_status={m: {"status": "queued", "reason": ""} for m in methods},
        runtime_log="",
    )
    append_worker_log(paths, f"Job start. mode={mode}, methods={methods}")

    cfg = load_cfg(cfg_path)
    methods_map: Dict[str, Dict[str, Any]] = cfg["_methods_map"]

    # IMPORTANT: per-job in/out dir (avoid multi-user conflicts)
    in_dir = str(paths.input_dir.resolve())
    out_dir = str(paths.output_dir.resolve())

    base_dir_raw = str(cfg.get("base_dir", "")).strip()
    if base_dir_raw:
        base_dir = str((backend_dir / Path(base_dir_raw)).resolve()) if not Path(base_dir_raw).is_absolute() else base_dir_raw
    else:
        # default: project root (one level above backend)
        base_dir = str((backend_dir.parent).resolve())

    # per-job run dir is job_dir itself
    ctx = {"base_dir": base_dir, "in_dir": in_dir, "out_dir": out_dir, "run_dir": str(job_dir)}

    # -------------------------
    # 1) Build master input
    # -------------------------
    update_status(job_dir, stage="build_master_input", progress=0.05)
    csv_path = payload.get("csv_path")
    if csv_path:
        df = pd.read_csv(str(csv_path))
    else:
        df = pd.DataFrame([{
            "TCR_alpha": str(payload.get("alpha") or "").strip(),
            "TCR_beta": str(payload.get("beta") or "").strip(),
            "Peptide": str(payload.get("peptide") or "").strip(),
            "MHC": str(payload.get("mhc") or "").strip(),
        }])

    cols_lower = {c.lower(): c for c in df.columns}

    def pick(*names):
        for n in names:
            if n.lower() in cols_lower:
                return cols_lower[n.lower()]
        return None

    ca = pick("tcr_alpha", "alpha", "alpha_cdr3", "alpha_v")
    cb = pick("tcr_beta", "beta", "beta_cdr3", "beta_v")
    cp = pick("peptide")
    cm = pick("mhc", "hla", "mhc/hla", "mhc_hla")

    if cp is None or cm is None or cb is None:
        raise ValueError(f"CSV must include peptide + mhc/hla + beta. Found columns: {list(df.columns)}")
    if mode == "vregion" and ca is None:
        raise ValueError("V region mode requires alpha sequence column (TCR_alpha/alpha/alpha_v).")

    master = pd.DataFrame({
        "TCR_alpha": df[ca].fillna("").astype(str).str.strip() if ca else "",
        "TCR_beta": df[cb].fillna("").astype(str).str.strip(),
        "Peptide": df[cp].fillna("").astype(str).str.strip(),
        "MHC": df[cm].fillna("").astype(str).str.strip(),
    })

    master = master[(master["TCR_beta"] != "") & (master["Peptide"] != "") & (master["MHC"] != "")]
    if mode == "vregion":
        master = master[(master["TCR_alpha"] != "")]
    if len(master) == 0:
        raise ValueError("No valid input rows after parsing.")

    master = master.reset_index(drop=True)
    master.insert(0, "pair_id", master.index.astype(int))

    master_csv = paths.tmp_dir / "master_input.csv"
    master.to_csv(master_csv, index=False)
    append_worker_log(paths, f"master_input rows={len(master)} -> {master_csv}")

    # -------------------------
    # 2) Generate per-method inputfiles via Platform_2_inputfiles.py
    # -------------------------
    update_status(job_dir, stage="generate_method_inputs", progress=0.10)

    if not inputfiles_script.exists():
        raise FileNotFoundError(f"Platform_2_inputfiles.py not found: {inputfiles_script}")

    # MixTCRpred special args (same logic as your current backend)
    mixt_cfg = methods_map.get("MixTCRpred", {}) or {}
    mixt_models_dir = _render(str(mixt_cfg.get("models_dir", "")).strip(), ctx)
    mixt_info_csv = _render(
        str(mixt_cfg.get("43model_info_models_csv", "") or mixt_cfg.get("info_models_csv", "")).strip(),
        ctx,
    )

    try:
        md = Path(mixt_models_dir) if mixt_models_dir else None
        if md:
            if md.name == "models" and (md.parent / "pretrained_models").exists():
                md = md.parent / "pretrained_models"
            elif md.name == "MixTCRpred-main" and (md / "pretrained_models").exists():
                md = md / "pretrained_models"
            mixt_models_dir = str(md)

        if mixt_models_dir:
            mdp = Path(mixt_models_dir)
            cand_43 = mdp / "43model--info_models.csv"
            cand_default = mdp / "info_models.csv"
            if cand_43.exists():
                mixt_info_csv = str(cand_43)
            elif (not mixt_info_csv) and cand_default.exists():
                mixt_info_csv = str(cand_default)
            elif mixt_info_csv:
                ic = Path(mixt_info_csv)
                if (not ic.exists()) and cand_default.exists():
                    mixt_info_csv = str(cand_default)
    except Exception:
        pass

    gen_env = "TCR-PEP_Platform"
    gen_cmd = (
        f'conda run -n "{gen_env}" --no-capture-output '
        f'python "{inputfiles_script}" '
        f'--in_csv "{master_csv}" '
        f'--out_dir "{paths.input_dir}" '
        f'--check_examples 0 '
        f'--mode "{mode}" '
        f'--mixt_models_dir "{mixt_models_dir}" '
        f'--mixt_info_csv "{mixt_info_csv}" '
    )

    rc, so, se = _run_bash(gen_cmd, timeout_s=1200)
    _dump_run_logs(paths.logs_dir, "_inputgen", gen_cmd, rc, so, se, "" if rc == 0 else f"Input generation failed (rc={rc})")
    if rc != 0:
        raise RuntimeError((se or so or "").strip()[-8000:] or "Platform_2_inputfiles failed")
    append_worker_log(paths, "Inputfiles generated.")

    # -------------------------
    # 3) Prepare matching keys and all_rows
    # -------------------------
    update_status(job_dir, stage="prepare_join_keys", progress=0.20)

    items: List[Dict[str, Any]] = []
    cdr_master_df: Optional[pd.DataFrame] = None
    if mode == "vregion":
        cdr_master_path = paths.input_dir / "CDR_master.csv"
        if cdr_master_path.exists():
            try:
                cdr_master_df = pd.read_csv(cdr_master_path)
            except Exception:
                cdr_master_df = None
        if cdr_master_df is None or len(cdr_master_df) != len(master):
            cdr_master_df = None

    for i, r in enumerate(master.itertuples(index=False)):
        pid = int(getattr(r, "pair_id", i))
        beta_raw = _norm(getattr(r, "TCR_beta"))
        alpha_raw = _norm(getattr(r, "TCR_alpha"))
        pep = _norm(getattr(r, "Peptide"))
        mhc_v = _norm(getattr(r, "MHC"))

        if mode == "vregion" and cdr_master_df is not None:
            cm = cdr_master_df.iloc[i]
            items.append({
                "pair_id": pid,
                "beta_cdr1": _norm(cm.get("beta_CDR1", "")),
                "beta_cdr2": _norm(cm.get("beta_CDR2", "")),
                "beta_cdr3": _norm(cm.get("beta_CDR3", "")),
                "alpha_cdr1": _norm(cm.get("alpha_CDR1", "")),
                "alpha_cdr2": _norm(cm.get("alpha_CDR2", "")),
                "alpha_cdr3": _norm(cm.get("alpha_CDR3", "")),
                "beta_cdr123": _norm(cm.get("beta_CDR123", "")),
                "alpha_cdr123": _norm(cm.get("alpha_CDR123", "")),
                "peptide": pep,
                "mhc": mhc_v,
                "beta": beta_raw,
                "alpha": alpha_raw,
            })
        else:
            items.append({
                "pair_id": pid,
                "beta": beta_raw,
                "alpha": alpha_raw,
                "peptide": pep,
                "mhc": mhc_v,
            })

    if mode == "vregion":
        input_key_chains: List[List[str]] = [
            _input_key_chain_vregion(
                x.get("beta_cdr123", ""), x.get("alpha_cdr123", ""),
                x.get("beta_cdr3", ""), x.get("alpha_cdr3", ""),
                x.get("mhc", ""), x.get("peptide", "")
            )
            for x in items
        ]
    else:
        input_key_chains = [
            _input_key_chain(x.get("beta", ""), x.get("alpha", ""), x.get("mhc", ""), x.get("peptide", ""))
            for x in items
        ]

    all_rows: List[Dict[str, Any]] = []
    pid2pos: Dict[int, int] = {}

    for x in items:
        pid = int(x.get("pair_id", len(all_rows)))
        pid2pos[pid] = len(all_rows)

        if mode == "vregion":
            b1 = _norm(x.get("beta_cdr1", ""))
            b2 = _norm(x.get("beta_cdr2", ""))
            b3 = _norm(x.get("beta_cdr3", ""))
            a1 = _norm(x.get("alpha_cdr1", ""))
            a2 = _norm(x.get("alpha_cdr2", ""))
            a3 = _norm(x.get("alpha_cdr3", ""))

            tcr_disp = b3 if not a3 else f"{b3}|{a3}"
            all_rows.append({
                "Index": pid + 1,
                "pair_id": pid,
                "betaCDR1": b1,
                "betaCDR2": b2,
                "betaCDR3": b3,
                "alphaCDR1": a1,
                "alphaCDR2": a2,
                "alphaCDR3": a3,
                "Peptide": x["peptide"],
                "MHC": x["mhc"],
                "TCR": tcr_disp,
            })
        else:
            b3 = _norm(x.get("beta", ""))
            a3 = _norm(x.get("alpha", ""))
            tcr_disp = b3 if not a3 else f"{b3}|{a3}"
            all_rows.append({
                "Index": pid + 1,
                "pair_id": pid,
                "betaCDR3": b3,
                "alphaCDR3": a3,
                "Peptide": x["peptide"],
                "MHC": x["mhc"],
                "TCR": tcr_disp,
            })

    update_status(
        job_dir,
        input_mode=mode,
        candidate_pairs=len(all_rows),
        methods_selected=len(methods),
    )

    # -------------------------
    # 4) Run selected methods
    # -------------------------
    failed_methods: Dict[str, str] = {}
    bar_counts: Dict[str, Dict[str, int]] = {}
    unmatched: Dict[str, Dict[str, Any]] = {}
    successful_methods: List[str] = []
    method_score_rows: List[Dict[str, Any]] = []

    def set_progress(done: int, total: int) -> None:
        # allocate 0.20 -> 0.85
        if total <= 0:
            return
        p = 0.20 + 0.65 * (done / total)
        with status_lock:
            update_status(job_dir, stage="running_methods", progress=round(p, 4))

    total = len(methods)
    done = 0

    update_status(job_dir, stage="running_methods", progress=0.20)

    status_lock = threading.Lock()
    def safe_update_method_status(method: str, status: str, reason: str = "") -> None:
        with status_lock:
            update_method_status(job_dir, methods, method, status, reason)

    def safe_append_worker_log(msg: str) -> None:
        with status_lock:
            append_worker_log(paths, msg)

    def method_failure(method: str, status: str, reason: str, cmd: str = "", rc: int = 999, so: str = "", se: str = "") -> Dict[str, Any]:
        _dump_run_logs(paths.logs_dir, method, cmd=cmd, rc=rc, so=so, se=se, reason=reason)
        return {
            "method": method,
            "status": status,
            "reason": reason,
            "bar_count": {"pos": 0, "neg": 0},
            "unmatched": {"missing_in_output": [], "match_level_used": []},
            "method_score_rows": [],
            "all_row_updates": [],
        }

    def run_one_method(m: str) -> Dict[str, Any]:
        safe_append_worker_log(f"Running method: {m}")
        safe_update_method_status(m, "running")

        try:
            if m not in methods_map:
                return method_failure(m, "failed", "Not found in tools_config.json", rc=999)

            mcfg = methods_map[m]
            allow_modes = mcfg.get("allow_modes")
            if isinstance(allow_modes, list) and mode not in [str(x).strip().lower() for x in allow_modes]:
                return method_failure(m, "unavailable", f"Not supported in mode={mode}", rc=998)

            env = str(mcfg.get("env", "")).strip()
            workdir = _render(str(mcfg.get("workdir", "")).strip(), ctx)
            cmd_tpl = str(mcfg.get("cmd", "")).strip()
            input_csv = _render(str(mcfg.get("input_csv", "")).strip(), ctx)
            output_csv = _render(str(mcfg.get("output_path", "")).strip(), ctx)

            if not env or not workdir or not cmd_tpl or not input_csv or not output_csv:
                return method_failure(
                    m,
                    "failed",
                    "Missing env/workdir/cmd/input_csv/output_path in tools_config.json",
                    rc=997,
                )

            Path(output_csv).parent.mkdir(parents=True, exist_ok=True)

            # MixTCRpred grouped runner
            if str(mcfg.get("runner", "")).strip() == "run_mixtcrpred":
                runner = runners_dir / "run_mixtcrpred.py"
                if not runner.exists():
                    return method_failure(m, "failed", "Missing backend/runners/run_mixtcrpred.py", rc=996)

                plan_csv = _render(str(mcfg.get("plan_csv", "")).strip(), ctx)
                run_cmd = (
                    f'conda run -n "TCR-PEP_Platform" --no-capture-output '
                    f'python "{runner}" '
                    f'--env "{env}" '
                    f'--workdir "{workdir}" '
                    f'--cmd_tpl {json.dumps(cmd_tpl)} '
                    f'--input_csv "{input_csv}" '
                    f'--plan_csv "{plan_csv}" '
                    f'--output_csv "{output_csv}"'
                )
                rc, so, se = _run_bash(run_cmd, timeout_s=2400)
                if rc != 0:
                    reason = (se or so or "").strip()[-8000:] or f"MixTCRpred failed (rc={rc})"
                    return method_failure(m, "failed", reason, cmd=run_cmd, rc=rc, so=so, se=se)
                _dump_run_logs(paths.logs_dir, m, cmd=run_cmd, rc=rc, so=so, se=se, reason="")
            else:
                cmd = _render(cmd_tpl, {**ctx, "input_csv": input_csv, "output_csv": output_csv})
                full = f'cd "{workdir}" && conda run -n "{env}" --no-capture-output bash -c {json.dumps(cmd)}'
                rc, so, se = _run_bash(full, timeout_s=2400)
                if rc != 0:
                    reason = (se or so or "").strip()[-8000:] or f"Non-zero exit (rc={rc})"
                    return method_failure(m, "failed", reason, cmd=full, rc=rc, so=so, se=se)
                _dump_run_logs(paths.logs_dir, m, cmd=full, rc=rc, so=so, se=se, reason="")

            if (not Path(output_csv).exists()) or Path(output_csv).stat().st_size == 0:
                return method_failure(m, "failed", "Output CSV not created (or empty).", rc=995)

            # Parse output + match back
            try:
                out_df = pd.read_csv(output_csv)
            except Exception as e:
                return method_failure(m, "failed", f"Failed to read output CSV: {e}", rc=994)

            om = mcfg.get("output_match") or {}
            pr = mcfg.get("pos_rule") or {}

            beta_col = _norm(om.get("beta_col", ""))
            alpha_col = _norm(om.get("alpha_col", ""))
            pep_col = _norm(om.get("peptide_col", ""))
            mhc_col = _norm(om.get("mhc_col", ""))

            score_col = _norm(pr.get("column", ""))
            thr = float(pr.get("threshold", 0.5))
            positive_if = _norm(pr.get("positive_if", "gt")).lower()
            score_direction = "low" if positive_if == "lt" else "high"

            def getcol(df_: pd.DataFrame, name: str) -> Optional[str]:
                if not name:
                    return None
                if name in df_.columns:
                    return name
                mm = {c.lower(): c for c in df_.columns}
                return mm.get(name.lower())

            bc = getcol(out_df, beta_col)
            ac = getcol(out_df, alpha_col) if alpha_col else None
            pc = getcol(out_df, pep_col) if pep_col else None
            mc = getcol(out_df, mhc_col) if mhc_col else None
            sc = getcol(out_df, score_col)

            if bc is None or sc is None or (pc is None and mc is None):
                reason = f"Output columns not found. Need beta+score and peptide or mhc. Got: {list(out_df.columns)}"
                return method_failure(m, "failed", reason, rc=993)

            rowmap_path = Path(input_csv).parent / f"{m}_rowmap.csv"
            out_with_pid = out_df.copy()

            if "pair_id" in out_with_pid.columns:
                pass
            elif rowmap_path.exists():
                rm_df = pd.read_csv(rowmap_path)
                submitted = pd.read_csv(input_csv, sep=";" if m == "ImRex" else ",")
                mapped = attach_ids(out_with_pid.to_dict("records"), submitted.to_dict("records"),
                                    rm_df["pair_id"].tolist(), om, m)
                out_with_pid = pd.DataFrame(mapped, columns=[*out_df.columns, "pair_id"])

            pos = neg = 0
            missing: List[str] = []
            used_level: List[str] = []
            local_method_score_rows: List[Dict[str, Any]] = []
            all_row_updates: List[Tuple[int, str]] = []

            if "pair_id" in out_with_pid.columns:
                pid_best: Dict[int, float] = {}
                pair_scores = pd.DataFrame({
                    "pair_id": pd.to_numeric(out_with_pid["pair_id"], errors="coerce"),
                    "score": pd.to_numeric(out_with_pid[sc], errors="coerce"),
                }).dropna()
                if len(pair_scores) > 0:
                    pair_scores["pair_id"] = pair_scores["pair_id"].astype(int)
                    best_scores = (
                        pair_scores.sort_values(["pair_id", "score"], ascending=[True, score_direction == "low"], kind="mergesort")
                        .drop_duplicates("pair_id", keep="first")
                    )
                    pid_best = dict(zip(best_scores["pair_id"].astype(int), best_scores["score"].astype(float)))

                for pid, row_pos in pid2pos.items():
                    row = all_rows[row_pos]
                    if pid not in pid_best:
                        all_row_updates.append((row_pos, ""))
                        missing.append(str(pid))
                        local_method_score_rows.append({
                            "pair_id": pid,
                            "group_id": _norm(row.get("Peptide")) + "|" + _norm(row.get("MHC")),
                            "method": m,
                            "raw_score": None,
                            "oriented_score": None,
                            "available": 0,
                            "missing_reason": "missing_output_or_score",
                            "score_column": score_col,
                            "score_direction": score_direction,
                        })
                        continue
                    v = float(pid_best[pid])
                    oriented = v if score_direction == "high" else -v
                    is_pos = _is_positive(v, thr, positive_if)
                    if is_pos:
                        pos += 1
                        all_row_updates.append((row_pos, f"pos({v:.3f})"))
                    else:
                        neg += 1
                        all_row_updates.append((row_pos, f"neg({v:.3f})"))
                    local_method_score_rows.append({
                        "pair_id": pid,
                        "group_id": _norm(row.get("Peptide")) + "|" + _norm(row.get("MHC")),
                        "method": m,
                        "raw_score": v,
                        "oriented_score": oriented,
                        "available": 1,
                        "missing_reason": "",
                        "score_column": score_col,
                        "score_direction": score_direction,
                    })
                    used_level.append("pair_id")
            else:
                out_map: Dict[str, float] = {}
                for _, rr in out_df.iterrows():
                    b = _norm(rr.get(bc, ""))
                    a = _norm(rr.get(ac, "")) if ac else ""
                    pep = _norm(rr.get(pc, "")) if pc else ""
                    mhc_v = _norm(rr.get(mc, "")) if mc else ""
                    try:
                        s = float(rr.get(sc))
                    except Exception:
                        continue
                    if not b:
                        continue
                    for kk in _output_key_chain(b, a, mhc_v, pep):
                        if not kk:
                            continue
                        if (kk not in out_map) or (s > out_map[kk] if score_direction == "high" else s < out_map[kk]):
                            out_map[kk] = s

                for i2, chain in enumerate(input_key_chains):
                    row = all_rows[i2]
                    v = None
                    hit_key = None
                    for kk in chain:
                        if kk in out_map:
                            v = out_map[kk]
                            hit_key = kk
                            break
                    if v is None:
                        missing.append(chain[0] if chain else f"row{i2}")
                        all_row_updates.append((i2, ""))
                        local_method_score_rows.append({
                            "pair_id": int(row.get("pair_id", i2)),
                            "group_id": _norm(row.get("Peptide")) + "|" + _norm(row.get("MHC")),
                            "method": m,
                            "raw_score": None,
                            "oriented_score": None,
                            "available": 0,
                            "missing_reason": "missing_output_or_score",
                            "score_column": score_col,
                            "score_direction": score_direction,
                        })
                        continue
                    oriented = float(v) if score_direction == "high" else -float(v)
                    is_pos = _is_positive(float(v), thr, positive_if)
                    if is_pos:
                        pos += 1
                        all_row_updates.append((i2, f"pos({float(v):.3f})"))
                    else:
                        neg += 1
                        all_row_updates.append((i2, f"neg({float(v):.3f})"))
                    local_method_score_rows.append({
                        "pair_id": int(row.get("pair_id", i2)),
                        "group_id": _norm(row.get("Peptide")) + "|" + _norm(row.get("MHC")),
                        "method": m,
                        "raw_score": float(v),
                        "oriented_score": oriented,
                        "available": 1,
                        "missing_reason": "",
                        "score_column": score_col,
                        "score_direction": score_direction,
                    })
                    used_level.append(hit_key or "")

            return {
                "method": m,
                "status": "completed",
                "reason": "",
                "bar_count": {"pos": pos, "neg": neg},
                "unmatched": {
                    "missing_in_output": missing,
                    "match_level_used": used_level[:50],
                },
                "method_score_rows": local_method_score_rows,
                "all_row_updates": all_row_updates,
            }
        except Exception as e:
            return method_failure(m, "failed", f"Unexpected method error: {e}", rc=992)

    max_workers = _method_worker_count(total)
    append_worker_log(paths, f"Method execution concurrency={max_workers}")

    method_results: Dict[str, Dict[str, Any]] = {}
    if max_workers <= 1:
        for m in methods:
            method_results[m] = run_one_method(m)
            done += 1
            set_progress(done, total)
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(run_one_method, m): m for m in methods}
            for fut in as_completed(futures):
                m = futures[fut]
                try:
                    method_results[m] = fut.result()
                except Exception as e:
                    method_results[m] = method_failure(m, "failed", f"Unexpected method error: {e}", rc=991)
                done += 1
                set_progress(done, total)

    for m in methods:
        res = method_results.get(m)
        if not res:
            reason = "Method did not return a result."
            failed_methods[m] = reason
            safe_update_method_status(m, "failed", reason)
            continue
        if res.get("status") == "completed":
            for row_pos, value in res.get("all_row_updates", []):
                all_rows[int(row_pos)][m] = value
            bar_counts[m] = res.get("bar_count", {"pos": 0, "neg": 0})
            unmatched[m] = res.get("unmatched", {"missing_in_output": [], "match_level_used": []})
            method_score_rows.extend(res.get("method_score_rows", []))
            successful_methods.append(m)
            safe_update_method_status(m, "completed")
        else:
            reason = str(res.get("reason") or "Method failed.")
            failed_methods[m] = reason
            safe_update_method_status(m, str(res.get("status") or "failed"), reason)

    _safe_write_text(paths.logs_dir / "_summary.failed_methods.json", json.dumps(failed_methods, indent=2, ensure_ascii=False))

    # -------------------------
    # 5) Legacy positive_items for old method-level compatibility.
    # The primary user-facing result is ranking_payload/final_score.
    # -------------------------
    update_status(job_dir, stage="build_positive_items", progress=0.90)

    positive_items: List[Dict[str, Any]] = []
    for row in all_rows:
        n_succ = 0
        n_pos = 0
        for mm in successful_methods:
            v = row.get(mm, "")
            if isinstance(v, str) and v:
                n_succ += 1
                if v.startswith("pos("):
                    n_pos += 1
        if n_succ == 0:
            continue
        if (n_pos / n_succ) <= 0.5:
            continue
        ratio_pct = int(round(100.0 * n_pos / n_succ))
        ratio_text = f"{n_pos}/{n_succ} ({ratio_pct}%)"

        if mode == "cdr3":
            positive_items.append({
                "Index": row.get("Index"),
                "betaCDR3": row.get("betaCDR3", ""),
                "alphaCDR3": row.get("alphaCDR3", ""),
                "Peptide": row.get("Peptide", ""),
                "MHC": row.get("MHC", ""),
                "Positive Ratio": ratio_text,
            })
        else:
            positive_items.append({
                "Index": row.get("Index"),
                "betaCDR1": row.get("betaCDR1", ""),
                "betaCDR2": row.get("betaCDR2", ""),
                "betaCDR3": row.get("betaCDR3", ""),
                "alphaCDR1": row.get("alphaCDR1", ""),
                "alphaCDR2": row.get("alphaCDR2", ""),
                "alphaCDR3": row.get("alphaCDR3", ""),
                "Peptide": row.get("Peptide", ""),
                "MHC": row.get("MHC", ""),
                "Positive Ratio": ratio_text,
            })

    update_status(job_dir, stage="ranking_aggregation", progress=0.96)
    ranking_payload = build_ranking_payload(
        mode=mode,
        all_rows=all_rows,
        selected_methods=methods,
        successful_methods=successful_methods,
        failed_methods=failed_methods,
        method_score_rows=method_score_rows,
    )

    result = {
        "mode": mode,
        "n_input_items": len(all_rows),
        "success_method_count": len(successful_methods),
        "successful_method_count": len(successful_methods),
        "successful_methods": successful_methods,
        "failed_methods": failed_methods,
        "bar_counts": bar_counts,
        "all_rows": all_rows,
        "positive_items": positive_items,
        "unmatched": unmatched,
        "run_dir": str(job_dir),
        **ranking_payload,
    }

    update_status(job_dir, status="success", stage="done", progress=1.0, current_method="")
    append_worker_log(paths, "Job success.")
    _atomic_write_json(job_dir / "result.json", result)
    return result
