from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantiphy.bootstrap import holm_adjust, paired_bootstrap, paired_video_bootstrap, subset_bootstrap
from quantiphy.data import load_validation, read_predictions
from quantiphy.mra import score
from quantiphy.official import EVALUATOR


def _toy(n_per=30, seed=0):
    rng = np.random.default_rng(seed)
    cats = np.repeat(["2S", "2D", "3S", "3D"], n_per)
    a = rng.integers(0, 11, size=len(cats)) / 10
    return a, cats


def test_identical_systems_give_zero_interval():
    a, cats = _toy()
    r = paired_bootstrap(a, a.copy(), cats)
    assert r["diff_a_minus_b"] == 0.0
    assert r["ci_low"] == 0.0 and r["ci_high"] == 0.0
    assert r["n_boot"] == 10_000


def test_constant_shift_is_recovered_exactly():
    a, cats = _toy()
    b = a - 0.1
    r = paired_bootstrap(a, b, cats)
    assert r["diff_a_minus_b"] == pytest.approx(0.1)
    assert r["ci_low"] == pytest.approx(0.1) and r["ci_high"] == pytest.approx(0.1)


def test_deterministic_given_seed():
    a, cats = _toy(seed=1)
    b, _ = _toy(seed=2)
    r1 = paired_bootstrap(a, b, cats, seed=7)
    r2 = paired_bootstrap(a, b, cats, seed=7)
    r3 = paired_bootstrap(a, b, cats, seed=8)
    assert r1 == r2
    assert r1["ci_low"] != r3["ci_low"]


def test_observed_diff_matches_macro_scorer():
    val = load_validation()
    gpt = score(val, read_predictions(EVALUATOR.parent / "model_outputs" / "gpt-5.1.csv"))
    rng = np.random.default_rng(3)
    noisy = val["answer"].to_numpy() * np.exp(rng.normal(0, 0.8, size=len(val)))
    import pandas as pd

    other = score(val, pd.Series(noisy, index=val["qid"].to_numpy()))
    r = paired_bootstrap(gpt.items["mra"], other.items["mra"], gpt.items["category"])
    assert r["macro_mra_a"] == pytest.approx(gpt.macro, abs=1e-12)
    assert r["macro_mra_b"] == pytest.approx(other.macro, abs=1e-12)
    assert r["ci_low"] <= r["diff_a_minus_b"] <= r["ci_high"]


# ---------------------------------------------------------------- video-level (cluster)


def _toy_videos(n_videos_per_cat=6, q_per_video=5, seed=0):
    """Toy data with a strong per-video effect: every question of a video shares its diff."""
    rng = np.random.default_rng(seed)
    cats, vids, a, b = [], [], [], []
    for c in ("2S", "2D", "3S", "3D"):
        for v in range(n_videos_per_cat):
            base = rng.integers(2, 9) / 10
            shift = rng.normal(0, 0.2)
            for _ in range(q_per_video):
                cats.append(c)
                vids.append(f"{c}_v{v}")
                a.append(base)
                b.append(float(np.clip(base - shift, 0, 1)))
    return np.array(a), np.array(b), np.array(cats), np.array(vids)


def test_video_identical_systems_zero_interval_and_p_one():
    a, _, cats, vids = _toy_videos()
    r = paired_video_bootstrap(a, a.copy(), cats, vids)
    assert r["diff_a_minus_b"] == 0.0
    assert r["ci_low"] == 0.0 and r["ci_high"] == 0.0
    assert r["p_two_sided"] == 1.0
    assert r["n_boot"] == 10_000
    assert r["n_videos_per_category"] == {"2S": 6, "2D": 6, "3S": 6, "3D": 6}


def test_video_constant_shift_is_recovered_exactly():
    a, _, cats, vids = _toy_videos()
    r = paired_video_bootstrap(a, a - 0.1, cats, vids)
    assert r["diff_a_minus_b"] == pytest.approx(0.1)
    assert r["ci_low"] == pytest.approx(0.1) and r["ci_high"] == pytest.approx(0.1)
    assert r["p_two_sided"] == pytest.approx(2 / 10_001)


def test_video_bootstrap_deterministic_given_seed():
    a, b, cats, vids = _toy_videos(seed=3)
    assert paired_video_bootstrap(a, b, cats, vids, seed=7) == paired_video_bootstrap(a, b, cats, vids, seed=7)
    assert paired_video_bootstrap(a, b, cats, vids, seed=7)["ci_low"] != paired_video_bootstrap(
        a, b, cats, vids, seed=8)["ci_low"]


def test_one_question_per_video_equals_question_bootstrap():
    # Sorted video ids follow row order, so both procedures consume the RNG identically.
    a, cats = _toy(n_per=20, seed=4)
    b, _ = _toy(n_per=20, seed=5)
    vids = np.array([f"v{i:04d}" for i in range(len(a))])
    rq = paired_bootstrap(a, b, cats, seed=11)
    rv = paired_video_bootstrap(a, b, cats, vids, seed=11)
    assert rv["ci_low"] == pytest.approx(rq["ci_low"], abs=1e-12)
    assert rv["ci_high"] == pytest.approx(rq["ci_high"], abs=1e-12)
    assert rv["p_two_sided"] == pytest.approx(rq["p_two_sided"], abs=1e-12)


def test_video_bootstrap_is_wider_when_questions_cluster_by_video():
    a, b, cats, vids = _toy_videos(q_per_video=8, seed=6)
    rq = paired_bootstrap(a, b, cats)
    rv = paired_video_bootstrap(a, b, cats, vids)
    assert rv["diff_a_minus_b"] == pytest.approx(rq["diff_a_minus_b"])
    assert (rv["ci_high"] - rv["ci_low"]) > 1.5 * (rq["ci_high"] - rq["ci_low"])


def test_video_bootstrap_rejects_shape_mismatch():
    a, b, cats, vids = _toy_videos()
    with pytest.raises(ValueError):
        paired_video_bootstrap(a, b, cats, vids[:-1])


def test_video_observed_diff_matches_macro_scorer_on_validation():
    val = load_validation()
    gpt = score(val, read_predictions(EVALUATOR.parent / "model_outputs" / "gpt-5.1.csv"))
    prior = score(val, pd.Series(val["prior_value"].to_numpy(), index=val["qid"].to_numpy()))
    vids = val.set_index("qid").loc[gpt.items["qid"], "video_id"].to_numpy()
    r = paired_video_bootstrap(gpt.items["mra"], prior.items["mra"], gpt.items["category"], vids)
    assert r["macro_mra_a"] == pytest.approx(gpt.macro, abs=1e-12)
    assert r["macro_mra_b"] == pytest.approx(prior.macro, abs=1e-12)
    assert sum(r["n_videos_per_category"].values()) == 24
    assert r["ci_low"] <= r["diff_a_minus_b"] <= r["ci_high"]


# ---------------------------------------------------------------- subsets and Holm


def test_subset_bootstrap_video_equals_question_when_one_question_per_video():
    rng = np.random.default_rng(9)
    a = rng.integers(0, 11, size=40) / 10
    b = rng.integers(0, 11, size=40) / 10
    vids = np.array([f"v{i:04d}" for i in range(40)])
    rq = subset_bootstrap(a, b, seed=2)
    rv = subset_bootstrap(a, b, vids, seed=2)
    assert rq["resample_unit"] == "question" and rv["resample_unit"] == "video"
    assert rv["ci_low"] == pytest.approx(rq["ci_low"]) and rv["ci_high"] == pytest.approx(rq["ci_high"])
    assert rq["diff_a_minus_b"] == pytest.approx(float((a - b).mean()))


def test_subset_bootstrap_uses_item_weighted_mean_per_video():
    # Video x has 3 questions (diff 0.3), video y has 1 (diff -0.1): item mean = 0.2.
    a = np.array([0.5, 0.5, 0.5, 0.2])
    b = np.array([0.2, 0.2, 0.2, 0.3])
    r = subset_bootstrap(a, b, np.array(["x", "x", "x", "y"]))
    assert r["diff_a_minus_b"] == pytest.approx(0.2)
    assert r["n_resampled_units"] == 2
    assert r["ci_low"] == pytest.approx(-0.1) and r["ci_high"] == pytest.approx(0.3)


def test_holm_adjust_known_values():
    adj = holm_adjust({"a": 0.01, "b": 0.04, "c": 0.03, "d": 0.2})
    assert adj["a"] == pytest.approx(0.04)
    assert adj["c"] == pytest.approx(0.09)
    assert adj["b"] == pytest.approx(0.09)  # 2 * 0.04 = 0.08, raised to keep monotone
    assert adj["d"] == pytest.approx(0.2)
    assert holm_adjust({"x": 0.6, "y": 0.7}) == {"x": 1.0, "y": 1.0}
