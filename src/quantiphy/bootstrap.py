"""Paired bootstraps for the difference in macro MRA between two prediction sets.

Two resampling units, both stratified by category (each of 2S/2D/3S/3D keeps its own
size, which matches how the macro score is built and guarantees no category is ever
empty) and both paired (the two systems are scored on the *same* resample):

* ``paired_video_bootstrap`` (PRIMARY): resample videos within each category and keep
  every question of each sampled video. Questions from the same video are correlated,
  so this is the honest interval.
* ``paired_bootstrap`` (secondary): resample single questions within each category.
  With only 24 validation videos this interval is optimistic (too narrow).

``subset_bootstrap`` handles unstratified subsets (e.g. only speed questions) and
``holm_adjust`` applies a Holm-Bonferroni correction over a family of comparisons.
"""

from __future__ import annotations

import numpy as np

from .data import CATEGORIES


def paired_bootstrap(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    categories: np.ndarray,
    *,
    n_boot: int = 10_000,
    seed: int = 20260923,
    alpha: float = 0.05,
) -> dict:
    """scores_*: per-question MRA (0..1, NaN = not scored); categories: '2S'/'2D'/'3S'/'3D'."""
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    cats = np.asarray(categories)
    if not (a.shape == b.shape == cats.shape):
        raise ValueError("scores_a, scores_b and categories must have the same shape")
    if n_boot < 1:
        raise ValueError("n_boot must be >= 1")
    keep = ~(np.isnan(a) | np.isnan(b))
    rng = np.random.default_rng(seed)

    obs_a, obs_b, boot_diff = [], [], np.zeros(n_boot)
    n_per_cat = {}
    for c in CATEGORIES:
        m = keep & (cats == c)
        ac, bc = a[m], b[m]
        n = len(ac)
        n_per_cat[c] = int(n)
        if n == 0:
            raise ValueError(f"category {c} has no scored questions")
        obs_a.append(ac.mean())
        obs_b.append(bc.mean())
        idx = rng.integers(0, n, size=(n_boot, n))
        boot_diff += (ac - bc)[idx].mean(axis=1)
    boot_diff /= len(CATEGORIES)

    macro_a = float(np.mean(obs_a))
    macro_b = float(np.mean(obs_b))
    lo, hi = np.quantile(boot_diff, [alpha / 2, 1 - alpha / 2])
    return {
        "macro_mra_a": macro_a,
        "macro_mra_b": macro_b,
        "diff_a_minus_b": macro_a - macro_b,
        "ci_level": 1 - alpha,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "boot_mean_diff": float(boot_diff.mean()),
        "frac_boot_diff_le_0": float((boot_diff <= 0).mean()),
        "p_two_sided": _two_sided_p(boot_diff),
        "n_boot": int(n_boot),
        "seed": int(seed),
        "resample_unit": "question, stratified by category",
        "n_per_category": n_per_cat,
    }


def _two_sided_p(boot_diff: np.ndarray) -> float:
    """Two-sided bootstrap p-value for H0: diff = 0 (percentile method, +1 smoothing)."""
    n = len(boot_diff)
    le = (np.sum(boot_diff <= 0) + 1) / (n + 1)
    ge = (np.sum(boot_diff >= 0) + 1) / (n + 1)
    return float(min(1.0, 2 * min(le, ge)))


def _cluster_sums(diff: np.ndarray, clusters: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-cluster sum of item differences and item counts (clusters in sorted order)."""
    _, inv = np.unique(clusters, return_inverse=True)
    sums = np.bincount(inv, weights=diff)
    counts = np.bincount(inv).astype(float)
    return sums, counts


def paired_video_bootstrap(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    categories: np.ndarray,
    videos: np.ndarray,
    *,
    n_boot: int = 10_000,
    seed: int = 20260923,
    alpha: float = 0.05,
) -> dict:
    """Paired *cluster* bootstrap: resample videos (with replacement) within each category.

    Every sampled video contributes all of its scored questions, and both systems are
    evaluated on the same resampled videos (paired). The category mean in a replicate
    is (sum of item differences over sampled videos) / (number of items in them), i.e.
    the same item-averaged MRA the official scorer computes; the macro difference is
    the unweighted mean over the four categories.

    This respects the correlation between questions of the same video, so it is the
    primary interval in our reports. With only ~6 videos per validation category the
    percentile interval is itself crude; treat it as a rough uncertainty band.
    """
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    cats = np.asarray(categories)
    vids = np.asarray(videos)
    if not (a.shape == b.shape == cats.shape == vids.shape):
        raise ValueError("scores_a, scores_b, categories and videos must have the same shape")
    if n_boot < 1:
        raise ValueError("n_boot must be >= 1")
    keep = ~(np.isnan(a) | np.isnan(b))
    rng = np.random.default_rng(seed)

    obs_a, obs_b, boot_diff = [], [], np.zeros(n_boot)
    n_per_cat, v_per_cat = {}, {}
    for c in CATEGORIES:
        m = keep & (cats == c)
        if not m.any():
            raise ValueError(f"category {c} has no scored questions")
        ac, bc = a[m], b[m]
        n_per_cat[c] = int(m.sum())
        obs_a.append(ac.mean())
        obs_b.append(bc.mean())
        sums, counts = _cluster_sums(ac - bc, vids[m])
        k = len(sums)
        v_per_cat[c] = int(k)
        idx = rng.integers(0, k, size=(n_boot, k))
        boot_diff += sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    boot_diff /= len(CATEGORIES)

    macro_a = float(np.mean(obs_a))
    macro_b = float(np.mean(obs_b))
    lo, hi = np.quantile(boot_diff, [alpha / 2, 1 - alpha / 2])
    return {
        "macro_mra_a": macro_a,
        "macro_mra_b": macro_b,
        "diff_a_minus_b": macro_a - macro_b,
        "ci_level": 1 - alpha,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "boot_mean_diff": float(boot_diff.mean()),
        "frac_boot_diff_le_0": float((boot_diff <= 0).mean()),
        "p_two_sided": _two_sided_p(boot_diff),
        "n_boot": int(n_boot),
        "seed": int(seed),
        "resample_unit": "video, stratified by category (all questions of a sampled video kept)",
        "n_per_category": n_per_cat,
        "n_videos_per_category": v_per_cat,
    }


def subset_bootstrap(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    videos: np.ndarray | None = None,
    *,
    n_boot: int = 10_000,
    seed: int = 20260923,
    alpha: float = 0.05,
) -> dict:
    """Paired bootstrap of the plain item mean difference on a subset of questions.

    No category stratification (a subset such as "speed questions" is not balanced over
    categories). If `videos` is given, videos are resampled (cluster bootstrap);
    otherwise single questions are resampled.
    """
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("scores_a and scores_b must have the same shape")
    keep = ~(np.isnan(a) | np.isnan(b))
    if not keep.any():
        raise ValueError("no scored questions")
    d = (a - b)[keep]
    rng = np.random.default_rng(seed)
    if videos is None:
        idx = rng.integers(0, len(d), size=(n_boot, len(d)))
        boot = d[idx].mean(axis=1)
        unit, n_clusters = "question", int(len(d))
    else:
        vids = np.asarray(videos)
        if vids.shape != a.shape:
            raise ValueError("videos must have the same shape as the scores")
        sums, counts = _cluster_sums(d, vids[keep])
        idx = rng.integers(0, len(sums), size=(n_boot, len(sums)))
        boot = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
        unit, n_clusters = "video", int(len(sums))
    lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    return {
        "mean_a": float(a[keep].mean()),
        "mean_b": float(b[keep].mean()),
        "diff_a_minus_b": float(d.mean()),
        "ci_level": 1 - alpha,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "p_two_sided": _two_sided_p(boot),
        "n_boot": int(n_boot),
        "seed": int(seed),
        "resample_unit": unit,
        "n_questions": int(keep.sum()),
        "n_resampled_units": n_clusters,
    }


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni step-down adjusted p-values (family-wise error control)."""
    names = sorted(p_values, key=lambda k: p_values[k])
    m = len(names)
    out: dict[str, float] = {}
    running = 0.0
    for i, k in enumerate(names):
        running = max(running, min(1.0, (m - i) * p_values[k]))
        out[k] = running
    return out
