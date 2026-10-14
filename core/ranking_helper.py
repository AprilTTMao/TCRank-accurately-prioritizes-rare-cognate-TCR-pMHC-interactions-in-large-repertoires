from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


BACKEND_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = BACKEND_DIR / "ranker_artifacts"
MIN_NULL_SIZE = 5


def _read_json(path: Path, default: Any = None) -> Any:
    try:
        if not path.exists():
            return default
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _as_float(value: Any) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        x = float(value)
        if not math.isfinite(x):
            return None
        return x
    except Exception:
        return None


def _norm(value: Any) -> str:
    return str(value or "").strip()


def _valid_null(values: Sequence[float]) -> bool:
    if values is None or len(values) < MIN_NULL_SIZE:
        return False
    mean = sum(values) / len(values)
    var = sum((x - mean) ** 2 for x in values) / len(values)
    return math.isfinite(var) and var > 0


def _reciprocal_rank_feature(rank: Optional[int]) -> float:
    if rank is None:
        return 0.0
    rank = float(rank)
    return 1.0 / rank if rank > 0 else 0.0


def _method_contract() -> List[Dict[str, Any]]:
    return _read_json(ARTIFACT_DIR / "method_score_contract.json", default=[]) or []


def _load_weights(mode: str) -> Dict[str, Any]:
    # The alpha-beta ranker is used when alpha-chain information is present.
    # Both CDR3 and V-region jobs can still fall back to beta-only if needed.
    path = ARTIFACT_DIR / ("ranker_weights_alpha_beta.json" if mode == "vregion" else "ranker_weights_beta_only.json")
    weights = _read_json(path, default=None)
    if not weights:
        weights = _read_json(ARTIFACT_DIR / "ranker_weights_beta_only.json", default=None)
    if not weights:
        return {"feature_columns": ["model_agreement"], "weights": {"model_agreement": 0.0}}
    return weights


def _score_direction_by_method(contract: Iterable[Dict[str, Any]]) -> Dict[str, str]:
    out = {}
    for item in contract:
        method = _norm(item.get("method"))
        if method:
            out[method] = _norm(item.get("direction") or "high").lower()
    return out


def _group_id(row: Dict[str, Any]) -> str:
    peptide = _norm(row.get("Peptide") or row.get("peptide"))
    mhc = _norm(row.get("MHC") or row.get("mhc"))
    return f"{peptide}|{mhc}"


def _identity_fields(row: Dict[str, Any], mode: str) -> Dict[str, Any]:
    base = {
        "pair_id": row.get("pair_id"),
        "Index": row.get("Index"),
        "Peptide": row.get("Peptide", ""),
        "MHC": row.get("MHC", ""),
        "TCR": row.get("TCR", ""),
    }
    if mode == "vregion":
        for key in ["betaCDR1", "betaCDR2", "betaCDR3", "alphaCDR1", "alphaCDR2", "alphaCDR3"]:
            base[key] = row.get(key, "")
    else:
        base["betaCDR3"] = row.get("betaCDR3", "")
        base["alphaCDR3"] = row.get("alphaCDR3", "")
    return base


def _contribution_json(feature_values: Dict[str, float], weights: Dict[str, Any]) -> str:
    contrib = {}
    for col in weights.get("feature_columns", []):
        w = float((weights.get("weights") or {}).get(col, 0.0) or 0.0)
        val = float(feature_values.get(col, 0.0) or 0.0)
        contrib[col] = {"value": val, "weight": w, "contribution": val * w}
    return json.dumps(contrib, ensure_ascii=False)


def _component_summary(feature_values: Dict[str, float], weights: Dict[str, Any]) -> Dict[str, float]:
    empirical = 0.0
    reciprocal = 0.0
    agreement = 0.0
    other = 0.0
    for col in weights.get("feature_columns", []):
        w = float((weights.get("weights") or {}).get(col, 0.0) or 0.0)
        val = float(feature_values.get(col, 0.0) or 0.0)
        contribution = val * w
        if col.startswith("evidence_"):
            empirical += contribution
        elif col.startswith("reciprocal_rank_"):
            reciprocal += contribution
        elif col == "model_agreement":
            agreement += contribution
        else:
            other += contribution
    return {
        "empirical_evidence": empirical,
        "reciprocal_rank_evidence": reciprocal,
        "model_agreement": agreement,
        "other_calibrated_terms": other,
    }


def _score_with_weights(feature_values: Dict[str, float], weights: Dict[str, Any]) -> float:
    score = 0.0
    for col in weights.get("feature_columns", []):
        score += float(feature_values.get(col, 0.0) or 0.0) * float((weights.get("weights") or {}).get(col, 0.0) or 0.0)
    return score


def build_ranking_payload(
    *,
    mode: str,
    all_rows: List[Dict[str, Any]],
    selected_methods: List[str],
    successful_methods: List[str],
    failed_methods: Dict[str, str],
    method_score_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    contract = _method_contract()
    contract_methods = [_norm(x.get("method")) for x in contract if _norm(x.get("method"))]
    method_order = [m for m in contract_methods if m in set(selected_methods or contract_methods)]
    for m in selected_methods:
        if m not in method_order:
            method_order.append(m)

    direction_by_method = _score_direction_by_method(contract)
    global_nulls = _read_json(ARTIFACT_DIR / "global_nulls_train.json", default={}) or {}
    weights = _load_weights(mode)

    rows_by_pid = {int(r.get("pair_id", i)): r for i, r in enumerate(all_rows)}
    groups: Dict[str, List[int]] = {}
    for pid, row in rows_by_pid.items():
        groups.setdefault(_group_id(row), []).append(pid)

    score_lookup: Dict[Tuple[int, str], Dict[str, Any]] = {}
    for item in method_score_rows:
        try:
            pid = int(item.get("pair_id"))
        except Exception:
            continue
        method = _norm(item.get("method"))
        if not method:
            continue
        score_lookup[(pid, method)] = item

    feature_by_pid: Dict[int, Dict[str, float]] = {
        pid: {"model_agreement": 0.0} for pid in rows_by_pid
    }
    detail_rows: Dict[Tuple[int, str], Dict[str, Any]] = {}
    agreement_hits: Dict[int, List[bool]] = {pid: [] for pid in rows_by_pid}

    for method in method_order:
        for pid in rows_by_pid:
            feature_by_pid[pid][f"evidence_{method}"] = 0.0
            feature_by_pid[pid][f"reciprocal_rank_{method}"] = 0.0

    for gid, pids in groups.items():
        for method in method_order:
            avail = []
            for pid in pids:
                seed = score_lookup.get((pid, method), {})
                raw = _as_float(seed.get("raw_score"))
                oriented = _as_float(seed.get("oriented_score"))
                if oriented is None and raw is not None:
                    oriented = raw if direction_by_method.get(method, "high") == "high" else -raw
                available = int(seed.get("available") or 0) == 1 and oriented is not None
                if available:
                    avail.append((pid, oriented, raw))

            ranked = sorted(avail, key=lambda x: x[1], reverse=True)
            rank_by_pid = {pid: i + 1 for i, (pid, _, _) in enumerate(ranked)}
            rank_group_size = len(ranked)
            null_scores = [float(x) for x in (global_nulls.get(method) or []) if _as_float(x) is not None]
            has_null = _valid_null(null_scores)

            for pid in pids:
                row_seed = score_lookup.get((pid, method), {})
                raw = _as_float(row_seed.get("raw_score"))
                oriented = _as_float(row_seed.get("oriented_score"))
                if oriented is None and raw is not None:
                    oriented = raw if direction_by_method.get(method, "high") == "high" else -raw
                available = int(row_seed.get("available") or 0) == 1 and oriented is not None

                rank = rank_by_pid.get(pid)
                reciprocal = _reciprocal_rank_feature(rank)
                evidence = 0.0
                empirical_p = None
                if available and has_null:
                    empirical_p = (1.0 + sum(1 for x in null_scores if x >= float(oriented))) / (1.0 + len(null_scores))
                    evidence = -math.log10(max(empirical_p, 1e-12))
                    agreement_hits[pid].append(evidence >= 1.0)

                feature_by_pid[pid][f"evidence_{method}"] = evidence
                feature_by_pid[pid][f"reciprocal_rank_{method}"] = reciprocal
                failed_reason = _norm(failed_methods.get(method))
                if available:
                    availability = "completed"
                elif "not supported" in failed_reason.lower():
                    availability = "unavailable"
                elif failed_reason:
                    availability = "failed"
                else:
                    availability = "NA"

                detail_rows[(pid, method)] = {
                    "pair_id": pid,
                    "group_id": gid,
                    "method": method,
                    "availability": availability,
                    "raw_score": raw,
                    "oriented_score": oriented,
                    "empirical_p": empirical_p,
                    "evidence": evidence,
                    "rank_within_pmhc": rank,
                    "contribution": 0.0,
                    "missing_reason": row_seed.get("missing_reason") or failed_methods.get(method, ""),
                }

    for pid, hits in agreement_hits.items():
        feature_by_pid[pid]["model_agreement"] = float(sum(hits) / len(hits)) if hits else 0.0

    ranking_rows = []
    for pid, row in rows_by_pid.items():
        features = feature_by_pid.get(pid, {})
        final_score = _score_with_weights(features, weights)
        components = _component_summary(features, weights)
        available_count = sum(
            1 for m in method_order
            if detail_rows.get((pid, m), {}).get("availability") == "completed"
        )

        for m in method_order:
            d = detail_rows.get((pid, m))
            if not d:
                continue
            contrib = 0.0
            for feature_name in (f"evidence_{m}", f"reciprocal_rank_{m}"):
                contrib += float(features.get(feature_name, 0.0) or 0.0) * float((weights.get("weights") or {}).get(feature_name, 0.0) or 0.0)
            d["contribution"] = contrib

        ranking_rows.append({
            **_identity_fields(row, mode),
            "group_id": _group_id(row),
            "ranker_name": weights.get("ranker_name", "ranker"),
            "final_score": final_score,
            "available_method_count": available_count,
            "selected_method_count": len(method_order),
            "method_contribution_json": _contribution_json(features, weights),
            "score_components": components,
        })

    ranking_rows.sort(key=lambda r: float(r.get("final_score") or 0.0), reverse=True)
    for i, row in enumerate(ranking_rows, start=1):
        row["overall_rank"] = i

    by_group: Dict[str, List[Dict[str, Any]]] = {}
    for row in ranking_rows:
        by_group.setdefault(row["group_id"], []).append(row)
    for group_rows in by_group.values():
        group_rows.sort(key=lambda r: float(r.get("final_score") or 0.0), reverse=True)
        size = len(group_rows)
        for i, row in enumerate(group_rows, start=1):
            row["pmhc_rank"] = i
            row["pmhc_group_size"] = size

    pmhc_group_summary = []
    for gid, group_rows in sorted(by_group.items()):
        scores = [float(r.get("final_score") or 0.0) for r in group_rows]
        first = group_rows[0] if group_rows else {}
        pmhc_group_summary.append({
            "group_id": gid,
            "Peptide": first.get("Peptide", ""),
            "MHC": first.get("MHC", ""),
            "pairs": len(group_rows),
            "score_min": min(scores) if scores else None,
            "score_max": max(scores) if scores else None,
            "score_range": f"{min(scores):.3f}-{max(scores):.3f}" if scores else "",
        })

    method_level_rows = []
    for row in ranking_rows:
        pid = int(row["pair_id"])
        for method in method_order:
            detail = detail_rows.get((pid, method))
            if detail:
                method_level_rows.append({
                    **detail,
                    "overall_rank": row.get("overall_rank"),
                    "pmhc_rank": row.get("pmhc_rank"),
                    "Peptide": row.get("Peptide", ""),
                    "MHC": row.get("MHC", ""),
                    "betaCDR1": row.get("betaCDR1", ""),
                    "betaCDR2": row.get("betaCDR2", ""),
                    "betaCDR3": row.get("betaCDR3", ""),
                    "alphaCDR1": row.get("alphaCDR1", ""),
                    "alphaCDR2": row.get("alphaCDR2", ""),
                    "alphaCDR3": row.get("alphaCDR3", ""),
                })

    score_values = [float(r.get("final_score") or 0.0) for r in ranking_rows]
    failed_count = sum(1 for m in selected_methods if m in failed_methods and "not supported" not in _norm(failed_methods.get(m)).lower())
    unavailable_count = sum(1 for m in selected_methods if "not supported" in _norm(failed_methods.get(m)).lower())
    ranking_summary = {
        "input_mode": mode,
        "candidate_pairs": len(ranking_rows),
        "pmhc_groups": len(pmhc_group_summary),
        "methods_completed": len(successful_methods),
        "methods_selected": len(selected_methods),
        "failed_methods": failed_count,
        "unavailable_methods": unavailable_count,
        "score_min": min(score_values) if score_values else None,
        "score_max": max(score_values) if score_values else None,
        "score_range": f"{min(score_values):.3f}-{max(score_values):.3f}" if score_values else "",
        "ranker_name": weights.get("ranker_name", "ranker"),
    }

    return {
        "ranking_rows": ranking_rows,
        "ranking_summary": ranking_summary,
        "pmhc_group_summary": pmhc_group_summary,
        "method_level_rows": method_level_rows,
        "method_order": method_order,
    }
