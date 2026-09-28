"""Stage 2 of hybrid_v2 (CPU): arbitrate between geometry_v1 and vlm_baseline.

    uv run python -m methods.hybrid_v2.hybrid --split val
    uv run python -m methods.hybrid_v2.hybrid --split test

Rule (thresholds in config.json; label-free, fixed before the validation set is scored):
  1. geometry fell back (could not measure)            -> VLM answer        (as hybrid_v1)
  2. geometry measured, and every track it used passed  -> geometry answer   (as hybrid_v1)
     the VLM box check (mean p_yes >= p_yes_threshold)
  3. geometry measured, but at least one used track     -> VLM answer, unless the VLM itself
     failed the box check                                   fell back to a constant
     (keep_geometry_if_vlm_fallback), then geometry

Needs geometry_v1 and vlm_baseline runs for the split, plus `methods.hybrid_v2.verify`.
Outputs:
  val : results/predictions/val_hybrid_v2.csv, results/hybrid_v2_val_summary.json
  test: submissions/hybrid_v2_test.csv, results/hybrid_v2_test_summary.json
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd

from quantiphy.data import ROOT, load_template, load_test, load_validation, write_predictions

from .tracks import used_tracks
from .verify import RUNS, VERSION, load_config


def track_verdicts(split: str, df: pd.DataFrame) -> pd.DataFrame:
    """One row per (question, used track): mean p_yes from the verifier cache."""
    tracks = used_tracks(df, split)
    cache: dict[str, dict] = {}
    rows = []
    for qid, refs in tracks.items():
        for ref in refs:
            if ref.video_id not in cache:
                p = RUNS / split / "verify" / f"{ref.video_id}.json"
                if not p.exists():
                    raise SystemExit(f"missing {p}; run `python -m methods.hybrid_v2.verify --split {split}` first")
                cache[ref.video_id] = json.loads(p.read_text(encoding="utf-8"))["tracks"]
            v = cache[ref.video_id][ref.key]
            rows.append({"qid": qid, "role": ref.role, "query": ref.query, "key": ref.key,
                         "mean_p_yes": v["mean_p_yes"]})
    return pd.DataFrame(rows, columns=["qid", "role", "query", "key", "mean_p_yes"])


def decide(geo: pd.DataFrame, vlm: pd.DataFrame, verdicts: pd.DataFrame, rule: dict) -> pd.DataFrame:
    """Pure arbitration step (unit-tested).

    geo: index qid, columns category, used_fallback, prediction
    vlm: index qid, columns prediction, used_fallback
    verdicts: rows (qid, role, mean_p_yes); mean_p_yes None/NaN = could not be checked
    """
    thr = float(rule["p_yes_threshold"])
    v = verdicts.copy()
    v["rejected"] = v["mean_p_yes"].astype(float) < thr  # NaN compares False: unchecked counts as passed
    per_q = v.groupby("qid").agg(
        n_tracks=("role", "size"),
        n_rejected=("rejected", "sum"),
        target_rejected=("rejected", lambda s: bool(s[v.loc[s.index, "role"] == "target"].any())),
        prior_rejected=("rejected", lambda s: bool(s[v.loc[s.index, "role"] == "prior"].any())),
        min_p_yes=("mean_p_yes", "min"),
    )
    df = geo[["category", "used_fallback", "prediction"]].rename(
        columns={"used_fallback": "geo_fallback", "prediction": "geo_pred"}
    ).join(vlm[["prediction", "used_fallback"]].rename(columns={"prediction": "vlm_pred", "used_fallback": "vlm_fallback"}))
    df = df.join(per_q)
    df["n_tracks"] = df["n_tracks"].fillna(0).astype(int)
    df["n_rejected"] = df["n_rejected"].fillna(0).astype(int)
    df[["target_rejected", "prior_rejected"]] = df[["target_rejected", "prior_rejected"]].fillna(False).astype(bool)

    box_rejected = ~df.geo_fallback & (df.n_rejected > 0)
    keep_geo = box_rejected & df.vlm_fallback & bool(rule["keep_geometry_if_vlm_fallback"])
    df["source"] = np.select(
        [df.geo_fallback, box_rejected & ~keep_geo],
        ["vlm:geometry_fell_back", "vlm:box_rejected"],
        default="geometry",
    )
    df.loc[keep_geo, "source"] = "geometry:box_rejected_but_vlm_fell_back"
    df["prediction"] = np.where(df.source.str.startswith("vlm"), df.vlm_pred, df.geo_pred)
    return df


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["val", "test"], required=True)
    args = ap.parse_args()
    cfg = load_config()
    df_q = load_validation().drop(columns=["answer"]) if args.split == "val" else load_test()
    geo = pd.read_csv(ROOT / "runs" / "geometry_v1" / args.split / "items.csv").set_index("qid")
    vlm = pd.read_csv(ROOT / "runs" / "vlm_baseline" / args.split / "items.csv").set_index("qid")
    if set(geo.index) != set(vlm.index) or set(geo.index) != set(df_q.qid):
        raise SystemExit("geometry, VLM and question table cover different questions")
    verdicts = track_verdicts(args.split, df_q)
    df = decide(geo, vlm, verdicts, cfg["rule"])
    out_dir = RUNS / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "items.csv")
    verdicts.to_csv(out_dir / "track_verdicts.csv", index=False)

    measured = ~df.geo_fallback
    tv = verdicts.drop_duplicates(["key", "qid"])
    summ = {
        "version": VERSION,
        "config": cfg,
        "n": int(len(df)),
        "source_counts": df.source.value_counts().to_dict(),
        "source_counts_by_category": {c: g.source.value_counts().to_dict() for c, g in df.groupby("category")},
        "changed_vs_hybrid_v1": int((df.source == "vlm:box_rejected").sum()),
        "changed_vs_hybrid_v1_by_category": df[df.source == "vlm:box_rejected"].category.value_counts().to_dict(),
        "measured_questions": int(measured.sum()),
        "measured_with_rejected_target": int((measured & df.target_rejected).sum()),
        "measured_with_rejected_prior": int((measured & df.prior_rejected).sum()),
        "measured_with_no_checkable_track": int((measured & (df.n_tracks == 0)).sum()),
        "question_track_pairs": int(len(tv)),
        "question_track_pairs_unchecked": int(tv.mean_p_yes.isna().sum()),
    }
    if args.split == "val":
        val = load_validation()
        p = ROOT / "results" / "predictions" / f"val_{VERSION}.csv"
        write_predictions(p, val, df.loc[val.qid, "prediction"].to_numpy())
        summ["predictions"] = p.relative_to(ROOT).as_posix()
    else:
        tmpl = load_template()
        tmpl["parsed_value"] = df.loc[tmpl["id"], "prediction"].to_numpy()
        p = ROOT / "submissions" / f"{VERSION}_test.csv"
        p.parent.mkdir(parents=True, exist_ok=True)
        tmpl.to_csv(p, index=False)
        summ["submission"] = p.relative_to(ROOT).as_posix()
    out = ROOT / "results" / f"{VERSION}_{args.split}_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summ, indent=2), encoding="utf-8")
    print(json.dumps(summ, indent=1))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
