"""hybrid_v4 = hybrid_v2 with its VLM answers taken from Qwen3-VL-32B (see README.md).

    uv run python -m methods.vlm_large.hybrid --split val
    uv run python -m methods.vlm_large.hybrid --split test

Rule: every question hybrid_v2 answered with the 8B VLM (source "vlm:*") gets the 32B
answer instead, unless the 32B reply had no usable number (then the hybrid_v2 answer
stays). Every other question keeps its hybrid_v2 answer. Nothing is tuned.
Outputs:
  val : results/predictions/val_hybrid_v4.csv, results/predictions/val_vlm_large.csv
        (32B direct answers on all 159 questions, exploratory), results/hybrid_v4_val_summary.json
  test: submissions/hybrid_v4_test.csv, results/hybrid_v4_test_summary.json
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd

from quantiphy.data import ROOT, load_template, load_test, load_validation, write_predictions

VERSION = "hybrid_v4"


def combine(v2: pd.DataFrame, big: pd.DataFrame) -> pd.DataFrame:
    """v2: index qid, columns category, source, prediction; big: index qid, prediction, used_fallback."""
    df = v2[["category", "source", "prediction"]].rename(columns={"source": "v2_source", "prediction": "v2_pred"})
    df = df.join(big[["prediction", "used_fallback"]].rename(columns={"prediction": "big_pred", "used_fallback": "big_fallback"}))
    wants = df.v2_source.str.startswith("vlm")
    missing = wants & df.big_pred.isna()
    if missing.any():
        raise SystemExit(f"{int(missing.sum())} VLM-answered questions have no 32B answer; run methods.vlm_large.run first")
    use = wants & ~df.big_fallback.fillna(True).astype(bool)
    df["source"] = np.where(use, "vlm32", np.where(wants, "v2:vlm32_fell_back", "v2"))
    df["prediction"] = np.where(use, df.big_pred, df.v2_pred)
    return df


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["val", "test"], required=True)
    args = ap.parse_args()
    v2 = pd.read_csv(ROOT / "runs" / "hybrid_v2" / args.split / "items.csv").set_index("qid")
    big = pd.read_csv(ROOT / "runs" / "vlm_large" / args.split / "items.csv").set_index("qid")
    df = combine(v2, big)
    (ROOT / "runs" / VERSION / args.split).mkdir(parents=True, exist_ok=True)
    df.to_csv(ROOT / "runs" / VERSION / args.split / "items.csv")
    changed = df.source == "vlm32"
    summ = {
        "version": VERSION, "n": int(len(df)),
        "source_counts": df.source.value_counts().to_dict(),
        "changed_vs_hybrid_v2": int((changed & (df.prediction != df.v2_pred)).sum()),
        "replaced_by_32b": int(changed.sum()),
        "replaced_by_32b_by_category": df[changed].category.value_counts().to_dict(),
        "vlm_answered_where_32b_fell_back": int((df.source == "v2:vlm32_fell_back").sum()),
    }
    if args.split == "val":
        val = load_validation()
        p = ROOT / "results" / "predictions" / f"val_{VERSION}.csv"
        write_predictions(p, val, df.loc[val.qid, "prediction"].to_numpy())
        p32 = ROOT / "results" / "predictions" / "val_vlm_large.csv"
        write_predictions(p32, val, big.loc[val.qid, "prediction"].to_numpy())
        summ["predictions"] = p.relative_to(ROOT).as_posix()
        summ["vlm_large_direct_predictions"] = p32.relative_to(ROOT).as_posix()
    else:
        load_test()  # asserts template/parquet alignment
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
