"""Match predictions by their submitted receptor/target fields, not row order."""
import math


def text(value):
    if value is None or isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip()


def attach_ids(outputs, inputs, pair_ids, output_columns, method):
    if len(inputs) != len(pair_ids):
        raise ValueError("Predictor input and row map have different lengths.")
    if not outputs:
        return []
    output_names = {name.lower(): name for name in outputs[0]}
    input_names = {name.lower(): name for name in inputs[0]} if inputs else {}
    aliases = {"cdr3": "tcr", "peptide": "epitope"} if method == "ATM-TCR" else {}
    pairs = []
    for role in ("beta_col", "alpha_col", "peptide_col", "mhc_col"):
        name = str(output_columns.get(role, "")).lower()
        actual = output_names.get(name)
        expected = input_names.get(name) or input_names.get(aliases.get(name, ""))
        if actual and expected:
            pairs.append((expected, actual))
        elif role in {"beta_col", "peptide_col"} and name:
            raise ValueError("Cannot map predictor output field: " + name)
    for name in ("a1", "a2", "b1", "b2", "trav", "traj", "trbv", "trbj"):
        if name in input_names and name in output_names:
            pairs.append((input_names[name], output_names[name]))
    if len(pairs) < 2:
        raise ValueError("Insufficient receptor/target fields to map predictions safely.")
    lookup = {}
    for row, pair_id in zip(inputs, pair_ids):
        key = tuple(text(row.get(expected)) for expected, _ in pairs)
        lookup.setdefault(key, []).append(int(pair_id))
    mapped = []
    for row in outputs:
        key = tuple(text(row.get(actual)) for _, actual in pairs)
        if key not in lookup:
            raise ValueError("A predictor output cannot be matched to its submitted receptor-target pair.")
        for pair_id in lookup[key]:
            mapped.append(dict(row, pair_id=pair_id))
    return mapped
