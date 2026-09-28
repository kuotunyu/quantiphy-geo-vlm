"""Mean Relative Accuracy (MRA), re-implemented to match external/QuantiPhy/evaluator.py.

Official rule (evaluator.py, starter kit commit 4f9323c):
    pred  = abs(to_numeric(parsed_value, errors="coerce"))
    hit_t = abs(pred - gt) / gt < (1 - t)   for t in {0.1, ..., 0.9, 0.95}
    item  = sum(hit_t) / 10                  (NaN if gt is NaN or gt == 0 -> item excluded)
    MRA_c = mean(item) over items of category c   (NaN items skipped)
    score = (MRA_S2 + MRA_D2 + MRA_S3 + MRA_D3) / 4

Quirks reproduced on purpose (defaults = official behaviour):
  * predictions are abs()'d, so the sign of a prediction never matters;
  * non-numeric / blank predictions -> NaN -> item scores 0 (still counted);
  * the relative error divides by the *signed* gt. A negative gt makes every finite
    prediction "correct" at all thresholds. The website formula |y_hat - y| / |y|
    corresponds to abs_pred=False, gt_abs=True. (All validation answers are positive,
    so for positive predictions the two variants agree on validation.)
  * the evaluator only scores rows present in the prediction file (missing="drop");
    our default missing="zero" is stricter and counts a missing prediction as 0.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from .data import CATEGORIES

THRESHOLDS: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)


def coerce_predictions(values) -> np.ndarray:
    """to_numeric(errors='coerce') then abs(), exactly like the official evaluator."""
    return pd.to_numeric(pd.Series(values, dtype=object), errors="coerce").abs().to_numpy(dtype=float)


def item_scores(pred, gt, *, abs_pred: bool = True, gt_abs: bool = False) -> np.ndarray:
    """Per-question MRA in [0, 1] (multiples of 0.1), NaN where gt is NaN or 0."""
    p = coerce_predictions(pred) if abs_pred else pd.to_numeric(pd.Series(pred, dtype=object), errors="coerce").to_numpy(dtype=float)
    g = np.asarray(gt, dtype=float)
    if p.shape != g.shape:
        raise ValueError(f"shape mismatch: pred {p.shape} vs gt {g.shape}")
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.abs(p - g) / (np.abs(g) if gt_abs else g)
        hits = np.zeros(g.shape, dtype=np.int64)
        for t in THRESHOLDS:
            hits += rel < (1 - t)  # NaN compares False, as in the official code
    scores = hits / 10
    scores[np.isnan(g) | (g == 0)] = np.nan
    return scores


def _mean(values: np.ndarray) -> float:
    v = values[~np.isnan(values)]
    return math.fsum(v) / len(v) if len(v) else float("nan")


@dataclass
class MRAResult:
    macro: float
    per_category: dict[str, float]
    n_per_category: dict[str, int]
    n_scored: int
    n_missing_predictions: int
    n_extra_predictions: int
    n_invalid_predictions: int  # blank / non-numeric / zero (same definition as evaluator.py)
    items: pd.DataFrame = field(repr=False)

    def as_dict(self) -> dict:
        return {
            "macro_mra": self.macro,
            "per_category_mra": self.per_category,
            "n_per_category": self.n_per_category,
            "n_scored": self.n_scored,
            "n_missing_predictions": self.n_missing_predictions,
            "n_extra_predictions": self.n_extra_predictions,
            "n_invalid_predictions": self.n_invalid_predictions,
        }


def score(
    gt: pd.DataFrame,
    predictions: pd.Series,
    *,
    missing: Literal["zero", "drop"] = "zero",
    abs_pred: bool = True,
    gt_abs: bool = False,
) -> MRAResult:
    """Score predictions (Series indexed by qid) against a GT table.

    `gt` needs columns: qid, category ("2S"/"2D"/"3S"/"3D"), answer.
    """
    if missing not in ("zero", "drop"):
        raise ValueError(missing)
    if not predictions.index.is_unique:
        raise ValueError("duplicate prediction ids")
    gt = gt[["qid", "category", "answer"]].copy()
    present = gt["qid"].isin(predictions.index).to_numpy()
    n_missing = int((~present).sum())
    n_extra = int((~predictions.index.isin(gt["qid"])).sum())
    if missing == "drop":
        gt = gt[present].reset_index(drop=True)
        present = np.ones(len(gt), dtype=bool)
    raw = [predictions.loc[q] if ok else np.nan for q, ok in zip(gt["qid"], present)]
    gt["parsed_value"] = raw
    gt["pred"] = coerce_predictions(raw)
    gt["mra"] = item_scores(raw, gt["answer"].to_numpy(), abs_pred=abs_pred, gt_abs=gt_abs)

    raw_series = pd.Series(raw, dtype=object)
    numeric = pd.to_numeric(raw_series, errors="coerce")
    n_invalid = int(raw_series.isna().sum() + (raw_series.notna() & numeric.isna()).sum() + (numeric == 0).sum())

    per_cat: dict[str, float] = {}
    n_cat: dict[str, int] = {}
    for c in CATEGORIES:
        v = gt.loc[gt["category"] == c, "mra"].to_numpy(dtype=float)
        per_cat[c] = _mean(v)
        n_cat[c] = int((~np.isnan(v)).sum())
    macro = sum(per_cat[c] for c in CATEGORIES) / len(CATEGORIES)
    return MRAResult(
        macro=macro,
        per_category=per_cat,
        n_per_category=n_cat,
        n_scored=int(gt["mra"].notna().sum()),
        n_missing_predictions=n_missing,
        n_extra_predictions=n_extra,
        n_invalid_predictions=n_invalid,
        items=gt,
    )
