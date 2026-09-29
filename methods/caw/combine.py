"""caw / caw_geospeed / caw_authors: Code-as-World-VL-9B answers (open weights -> Track B).

    uv run python -m methods.caw.combine --split test
    uv run python -m methods.caw.combine --split val
    uv run python -m methods.caw.combine --split test --accept-failed-videos <video_id> ...

CPU only, main env. Inputs: runs/caw/<split>/items.csv (methods.caw.run), the hybrid_v4 answers
(submissions/hybrid_v4_test.csv, results/predictions/val_hybrid_v4.csv) and hybrid_v2's
per-question source (runs/hybrid_v2/<split>/items.csv). Three frozen variants; nothing is tuned:
  caw           every question takes the CaW answer when usable (parse_answer_sci read a finite,
                non-zero number, i.e. methods.caw.run used_fallback == False; the value is |number|),
                else its hybrid_v4 answer.
  caw_geospeed  as caw, except speed questions (methods.geometry.parse.parse_question kind "speed")
                whose hybrid_v2 source is exactly "geometry" (measured with all boxes accepted) take
                that geometry answer (hybrid_v2 and hybrid_v4 agree on those questions; asserted).
  caw_authors   the authors' own answer: |first number of the reply| (their parsed_value, no unit
                conversion), a blank cell when the reply has no number; no fallback. The official
                evaluator takes |value| and scores a blank cell 0. This is the port-fidelity output
                (the paper's 55.4 for the 9B comes from this parser), not an upload candidate: it is
                written for validation only (and as a column of combined.csv on both splits).
Any CaW record with an error (a video the run gave up on) makes this refuse to run, unless every
such video is named with --accept-failed-videos (their questions then take hybrid_v4 in caw and
caw_geospeed, and a blank cell in caw_authors).
Outputs:
  val : results/predictions/val_caw.csv, val_caw_geospeed.csv, val_caw_authors.csv,
        results/caw_combine_val_summary.json
  test: submissions/caw_test.csv, caw_geospeed_test.csv, results/caw_combine_test_summary.json
  both: runs/caw/<split>/combined.csv (per question: every answer and where it came from)
"""

from __future__ import annotations

import argparse
import json
import math
import sys

import numpy as np
import pandas as pd

from quantiphy.data import ROOT, load_template, load_test, load_validation, read_predictions, write_predictions

from methods.geometry.parse import parse_question

VARIANTS = ("caw", "caw_geospeed")  # upload candidates (test submissions)
FIDELITY = "caw_authors"  # validation only
V4_FILES = {"test": ROOT / "submissions" / "hybrid_v4_test.csv", "val": ROOT / "results" / "predictions" / "val_hybrid_v4.csv"}


def check_errors(caw: pd.DataFrame, accepted: list[str] | None) -> list[str]:
    """Videos whose CaW records carry an error. Refuses unless `accepted` names exactly those videos."""
    err = caw.error.fillna("").astype(str) if "error" in caw else pd.Series("", index=caw.index)
    failed = sorted(caw.loc[err != "", "video_id"].astype(str).unique().tolist())
    accepted = sorted(set(accepted or []))
    if failed != accepted:
        raise SystemExit(
            f"CaW records with errors in videos {failed} (questions: {int((err != '').sum())}); accepted: {accepted}. "
            "Rerun methods.caw.run if the error can go away, or name exactly these videos with "
            "--accept-failed-videos to give their questions the fallback answers.")
    return failed


def combine(meta: pd.DataFrame, caw: pd.DataFrame, v4: pd.Series, v2: pd.DataFrame) -> pd.DataFrame:
    """meta: loader rows (qid, category, question); caw: index qid, used_fallback, prediction,
    parsed_value_authors; v4: hybrid_v4 answer by qid; v2: index qid, source, prediction.
    Returns one row per question."""
    df = meta[["qid", "category", "question"]].set_index("qid")
    missing = df.index.difference(caw.index)
    if len(missing):
        raise SystemExit(f"{len(missing)} questions have no CaW record; run methods.caw.run first")
    df["kind"] = [parse_question(q).get("kind") for q in df.question]
    df["v2_source"] = v2.source.reindex(df.index)
    df["v2_pred"] = pd.to_numeric(v2.prediction.reindex(df.index), errors="coerce")
    df["v4_pred"] = pd.to_numeric(v4.reindex(df.index), errors="coerce")
    df["caw_pred"] = pd.to_numeric(caw.prediction.reindex(df.index), errors="coerce")
    df["caw_usable"] = ~caw.used_fallback.reindex(df.index).astype(bool)
    if df.v4_pred.isna().any() or df.v2_source.isna().any():
        raise SystemExit("hybrid_v4 / hybrid_v2 do not cover every question")
    bad = df.caw_usable & ~(np.isfinite(df.caw_pred) & (df.caw_pred > 0))
    if bad.any():
        raise SystemExit(f"{int(bad.sum())} 'usable' CaW answers are not finite and positive")
    df["caw_final"] = np.where(df.caw_usable, df.caw_pred, df.v4_pred)
    df["caw_src"] = np.where(df.caw_usable, "caw", "hybrid_v4:caw_unusable")
    geo_speed = (df.kind == "speed") & (df.v2_source == "geometry")
    if not np.allclose(df.v2_pred[geo_speed], df.v4_pred[geo_speed], rtol=1e-12, atol=0):
        raise SystemExit("hybrid_v2 and hybrid_v4 disagree on geometry-answered speed questions")
    df["caw_geospeed_final"] = np.where(geo_speed, df.v2_pred, df.caw_final)
    df["caw_geospeed_src"] = np.where(geo_speed, "geometry:speed", df.caw_src)
    authors = pd.to_numeric(caw.parsed_value_authors.reindex(df.index), errors="coerce")
    df["caw_authors_final"] = authors.abs()  # NaN (a blank cell) when their parser found no number
    df["caw_authors_src"] = np.where(authors.isna(), "blank:no_number", "caw_authors")
    df["caw_authors_signed_negative"] = authors < 0  # their own _mra uses the signed value
    return df


def summary(df: pd.DataFrame) -> dict:
    out = {"n": int(len(df)), "caw_usable": int(df.caw_usable.sum()),
           "caw_usable_by_category": df[df.caw_usable].category.value_counts().to_dict()}
    for v in VARIANTS:
        changed = df[f"{v}_final"] != df.v4_pred
        out[v] = {
            "source_counts": df[f"{v}_src"].value_counts().to_dict(),
            "source_counts_by_category": {c: g[f"{v}_src"].value_counts().to_dict() for c, g in df.groupby("category")},
            "changed_vs_hybrid_v4": int(changed.sum()),
        }
    a = df[f"{FIDELITY}_final"]
    out[FIDELITY] = {
        "blank": int(a.isna().sum()), "blank_by_category": df[a.isna()].category.value_counts().to_dict(),
        "zero": int((a == 0).sum()), "negative_before_abs": int(df.caw_authors_signed_negative.sum()),
    }
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["val", "test"], required=True)
    ap.add_argument("--accept-failed-videos", nargs="+", default=None, metavar="VIDEO_ID",
                    help="videos whose CaW records carry an error; their questions take the fallback answers")
    args = ap.parse_args(argv)
    meta = load_validation().drop(columns=["answer"]) if args.split == "val" else load_test()
    caw = pd.read_csv(ROOT / "runs" / "caw" / args.split / "items.csv").set_index("qid")
    failed_videos = check_errors(caw, args.accept_failed_videos)
    v4 = read_predictions(V4_FILES[args.split])
    v2 = pd.read_csv(ROOT / "runs" / "hybrid_v2" / args.split / "items.csv").set_index("qid")
    df = combine(meta, caw, v4, v2)
    df.to_csv(ROOT / "runs" / "caw" / args.split / "combined.csv")
    run_summary = ROOT / "results" / f"caw_{args.split}_summary.json"
    run_code = json.loads(run_summary.read_text(encoding="utf-8")).get("code") if run_summary.exists() else None
    summ = summary(df) | {"accepted_failed_videos": failed_videos, "run_code": run_code,
                          "inputs": {"caw": f"runs/caw/{args.split}/items.csv",
                                     "hybrid_v4": V4_FILES[args.split].relative_to(ROOT).as_posix(),
                                     "hybrid_v2": f"runs/hybrid_v2/{args.split}/items.csv"}}
    if args.split == "val":
        for v in VARIANTS + (FIDELITY,):  # meta has no answer column; write_predictions writes id + metadata + parsed_value
            p = ROOT / "results" / "predictions" / f"val_{v}.csv"
            write_predictions(p, meta, df.loc[meta.qid, f"{v}_final"].to_numpy())  # NaN -> blank cell
            summ[v]["predictions"] = p.relative_to(ROOT).as_posix()
    else:
        for v in VARIANTS:
            tmpl = load_template()
            tmpl["parsed_value"] = df.loc[tmpl["id"], f"{v}_final"].to_numpy()
            if not all(math.isfinite(x) and x > 0 for x in tmpl.parsed_value):
                raise SystemExit(f"{v}: non-finite or non-positive answers")
            p = ROOT / "submissions" / f"{v}_test.csv"
            tmpl.to_csv(p, index=False)
            summ[v]["submission"] = p.relative_to(ROOT).as_posix()
    out = ROOT / "results" / f"caw_combine_{args.split}_summary.json"
    out.write_text(json.dumps(summ, indent=2), encoding="utf-8")
    print(json.dumps(summ, indent=1))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
