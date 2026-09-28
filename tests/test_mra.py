"""Our MRA scorer vs. the unmodified official evaluator (external/QuantiPhy/evaluator.py).

Every parity test writes one prediction CSV and feeds the *same file* to both
scorers. Pass criteria (see quantiphy.parity):
  * per-question scores bit-identical;
  * per-category and macro MRA within 1e-12 (only float summation order differs);
  * invalid_percentage identical.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from quantiphy.data import VAL_CSV, load_validation, read_predictions, write_predictions
from quantiphy.mra import THRESHOLDS, item_scores, score
from quantiphy.official import EVALUATOR
from quantiphy.parity import compare_with_official

STARTER_GPT51 = EVALUATOR.parent / "model_outputs" / "gpt-5.1.csv"


@pytest.fixture(scope="module")
def val():
    return load_validation()


def _check(pred_csv, gt_csv=VAL_CSV):
    r = compare_with_official(pred_csv, gt_csv)
    assert r["items_identical"], r
    assert r["max_abs_diff_category"] <= 1e-12, r
    assert r["abs_diff_macro"] <= 1e-12, r
    assert r["match"], r
    return r


# ---------------------------------------------------------------- hand-computed cases


def test_thresholds_are_official():
    assert THRESHOLDS == (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)


@pytest.mark.parametrize(
    "pred, gt, expected",
    [
        (10.0, 10.0, 1.0),  # exact
        (10.4, 10.0, 1.0),  # rel 0.04 < 0.05
        (10.6, 10.0, 0.9),  # rel 0.06: fails only theta=0.95
        (15.0, 10.0, 0.4),  # rel 0.5: passes 0.1..0.4; 0.5 fails (strict <)
        (-15.0, 10.0, 0.4),  # sign of prediction is ignored (abs)
        (0.0, 10.0, 0.0),  # zero prediction: rel 1.0
        (1000.0, 10.0, 0.0),  # wrong unit (x100)
        (0.1, 10.0, 0.0),  # wrong unit (/100)
        ("abc", 10.0, 0.0),  # non-numeric -> NaN -> 0
        (None, 10.0, 0.0),  # blank -> 0
        (float("inf"), 10.0, 0.0),
        ("1e1", 10.0, 1.0),  # scientific notation string parses
    ],
)
def test_item_score_hand_computed(pred, gt, expected):
    assert item_scores([pred], [gt])[0] == expected


def test_item_score_excluded_gt():
    s = item_scores([1.0, 1.0], [0.0, float("nan")])
    assert np.isnan(s).all()


def test_negative_gt_official_quirk_vs_website_formula():
    # Official code divides by the signed gt: any finite prediction is "correct".
    assert item_scores([123.0], [-2.0])[0] == 1.0
    # Website formula |y_hat - y| / |y| (no abs on the prediction): the sign must be right.
    assert item_scores([123.0], [-2.0], abs_pred=False, gt_abs=True)[0] == 0.0
    assert item_scores([2.0], [-2.0], abs_pred=False, gt_abs=True)[0] == 0.0
    assert item_scores([-2.0], [-2.0], abs_pred=False, gt_abs=True)[0] == 1.0


def test_macro_is_mean_of_four_categories():
    gt = pd.DataFrame(
        {"qid": [1, 2, 3, 4, 5], "category": ["2S", "2S", "2D", "3S", "3D"], "answer": [10.0] * 5}
    )
    preds = pd.Series([10.0, 15.0, 10.0, 0.0, 10.6], index=[1, 2, 3, 4, 5])
    r = score(gt, preds)
    assert r.per_category == {"2S": 0.7, "2D": 1.0, "3S": 0.0, "3D": 0.9}
    assert math.isclose(r.macro, (0.7 + 1.0 + 0.0 + 0.9) / 4, abs_tol=1e-15)


def test_missing_prediction_zero_vs_drop():
    gt = pd.DataFrame({"qid": [1, 2, 3, 4], "category": ["2S", "2D", "3S", "3D"], "answer": [1.0] * 4})
    preds = pd.Series([1.0, 1.0, 1.0], index=[1, 2, 3])
    assert score(gt, preds, missing="zero").macro == 0.75
    r = score(gt, preds, missing="drop")
    assert math.isnan(r.per_category["3D"]) and math.isnan(r.macro)


# ---------------------------------------------------------------- parity with evaluator.py


def test_parity_starter_kit_gpt51_outputs():
    r = _check(STARTER_GPT51)
    # value committed by the organizers in external/QuantiPhy/mra_results/all_model_results.csv
    assert r["official"]["macro"] == 0.4856119486941172


def test_parity_perfect_predictions(tmp_path, val):
    p = write_predictions(tmp_path / "gt.csv", val, val["answer"])
    r = _check(p)
    assert r["official"]["macro"] == 1.0


@pytest.mark.parametrize("seed", range(25))
def test_parity_random_predictions(tmp_path, val, seed):
    rng = np.random.default_rng(seed)
    y = val["answer"].to_numpy()
    # multiplicative noise spanning every threshold band, random signs
    pred = y * np.exp(rng.normal(0, 0.6, size=len(y))) * rng.choice([1, -1], size=len(y), p=[0.8, 0.2])
    p = write_predictions(tmp_path / f"rand{seed}.csv", val, pred)
    _check(p)


def test_parity_threshold_boundaries(tmp_path, val):
    y = val["answer"].to_numpy()
    rng = np.random.default_rng(0)
    t = rng.choice(np.array(THRESHOLDS), size=len(y))
    side = rng.choice([-1.0, 1.0], size=len(y))
    pred = y * (1 + side * (1 - t))  # relative error lands exactly on 1 - theta
    p = write_predictions(tmp_path / "edge.csv", val, pred)
    _check(p)


def test_parity_invalid_zero_negative_values(tmp_path, val):
    y = val["answer"].to_numpy().astype(object)
    vals = list(y)
    specials = [0, 0.0, -1.0, "abc", "5 m", "", "inf", "-inf", "1e3", -0.0, 1e300, "nan"]
    for i, s in enumerate(specials):
        vals[i * 7] = s
    for i in range(100, 130):
        vals[i] = -float(y[i])  # negative of the right answer -> abs -> full marks
    p = write_predictions(tmp_path / "special.csv", val, vals)
    r = _check(p)
    assert r["official_invalid_percentage"] > 0


def test_parity_wrong_units(tmp_path, val):
    y = val["answer"].to_numpy()
    pred = y.copy()
    pred[::2] = pred[::2] * 100  # e.g. answered in cm when meters were asked
    pred[1::4] = pred[1::4] / 1000  # e.g. answered in km
    p = write_predictions(tmp_path / "units.csv", val, pred)
    r = _check(p)
    assert r["official"]["macro"] < 0.5


def test_parity_missing_rows(tmp_path, val):
    rng = np.random.default_rng(1)
    keep = np.sort(rng.choice(len(val), size=120, replace=False))
    sub = val.iloc[keep]
    pred = sub["answer"].to_numpy() * 1.07
    p = write_predictions(tmp_path / "partial.csv", sub, pred)
    _check(p)  # evaluator.py silently skips missing ids; our missing="drop" does the same
    strict = score(val, read_predictions(p), missing="zero")
    assert strict.n_missing_predictions == len(val) - 120
    assert strict.macro < 0.95


def test_parity_modified_gt_zero_negative_nan(tmp_path, val):
    """GT edge cases that do not occur in the real validation set."""
    raw = pd.read_csv(VAL_CSV)
    raw.loc[0, "ground_truth_posterior"] = 0.0  # excluded by both
    raw.loc[1, "ground_truth_posterior"] = -3.0  # official quirk: any finite pred counts as correct
    raw.loc[2, "ground_truth_posterior"] = np.nan  # excluded by both
    gt_csv = tmp_path / "gt_modified.csv"
    raw.to_csv(gt_csv, index=False)
    rng = np.random.default_rng(2)
    pred = val["answer"].to_numpy() * np.exp(rng.normal(0, 0.5, size=len(val)))
    p = write_predictions(tmp_path / "pred.csv", val, pred)
    _check(p, gt_csv)
    ours = score(load_validation(gt_csv), read_predictions(p))
    items = ours.items.set_index("qid")["mra"]
    assert np.isnan(items.iloc[0]) and np.isnan(items.iloc[2])
    assert items.iloc[1] == 1.0
