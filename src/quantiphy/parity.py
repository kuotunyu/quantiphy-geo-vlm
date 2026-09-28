"""Score one prediction file with both our scorer and the official evaluator, and diff them."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .data import CATEGORIES, SITE_TO_OFFICIAL, VAL_CSV, load_validation, read_predictions
from .mra import score
from .official import run_official

AGG_TOL = 1e-12  # aggregates may differ only by float summation order


def compare_with_official(pred_csv: Path | str, gt_csv: Path | str = VAL_CSV) -> dict:
    """Return both results plus the max absolute differences.

    Our scorer runs with missing="drop" here, because that is what evaluator.py does
    (it only scores ids present in the prediction file).
    """
    gt = load_validation(Path(gt_csv))
    preds = read_predictions(pred_csv)
    ours = score(gt, preds, missing="drop")
    off_row, off_items, _ = run_official(pred_csv, gt_csv)

    # Item level: compare the per-question score for every id the evaluator saw.
    off = off_items.dropna(subset=["qid"]).copy()
    off["qid"] = off["qid"].astype(int)
    off = off.set_index("qid")["mra"].astype(float)
    mine = ours.items.set_index("qid")["mra"].astype(float)
    mine = mine.reindex(off.index)
    items_identical = bool(np.array_equal(off.to_numpy(), mine.to_numpy(), equal_nan=True))
    n_items_diff = int((~((off.to_numpy() == mine.to_numpy()) | (np.isnan(off.to_numpy()) & np.isnan(mine.to_numpy())))).sum())

    official = {c: float(off_row[f"mra_{SITE_TO_OFFICIAL[c]}"]) for c in CATEGORIES}
    official_macro = float(off_row["mra_average"])
    cat_diffs = {c: abs(official[c] - ours.per_category[c]) for c in CATEGORIES}
    macro_diff = abs(official_macro - ours.macro)
    n_rows = len(read_predictions(pred_csv))
    inv_ours = ours.n_invalid_predictions / n_rows * 100 if n_rows else 0.0
    return {
        "pred_csv": str(pred_csv),
        "ours": {"macro": ours.macro, **ours.per_category},
        "official": {"macro": official_macro, **official},
        "official_invalid_percentage": float(off_row["invalid_percentage"]),
        "ours_invalid_percentage": inv_ours,
        "items_compared": int(len(off)),
        "items_identical": items_identical,
        "n_items_different": n_items_diff,
        "max_abs_diff_category": max(cat_diffs.values()),
        "abs_diff_macro": macro_diff,
        "match": items_identical
        and max(cat_diffs.values()) <= AGG_TOL
        and macro_diff <= AGG_TOL
        and abs(inv_ours - float(off_row["invalid_percentage"])) <= AGG_TOL,
    }
