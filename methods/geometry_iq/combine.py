"""caw_geospeed_iq / caw_iq: the Track B files in which every answer uses only its own question
(README "Per-question version"). CPU only, main env.

    uv run python -m methods.geometry_iq.combine --split val  [--insurance-only]
    uv run python -m methods.geometry_iq.combine --split test [--insurance-only]

Per question, from its own records only:
  own_fallback     same-dimension value in the question's own prior (methods.geometry.solver.fallback_value),
                   else the dimension's fixed validation-prior constant (geometry_iq.run.ref_constants),
                   in the question's unit. Used only where the CaW reply is unusable.
  caw_iq           (insurance; no GPU) the CaW answer when usable (as methods.caw.combine: used_fallback False,
                   finite, > 0), else own_fallback. Differs from caw only by not using hybrid_v4.
  caw_geospeed_iq  as caw_iq, except questions whose own text is kind "speed" and whose geometry_iq source
                   is "geometry" take the geometry_iq answer (the caw_geospeed rule on the per-question
                   pipeline).
CaW answers are reused as they are (runs/caw/<split>/items.csv, methods.caw.run; per question).
Outputs:
  val : results/predictions/val_caw_iq.csv, val_caw_geospeed_iq.csv
  test: submissions/caw_iq_test.csv, submissions/caw_geospeed_iq_test.csv (official template)
  both: runs/geometry_iq/<split>/combined.csv, results/caw_iq_combine_<split>_summary.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys

import numpy as np
import pandas as pd

from quantiphy.data import ROOT, load_template, load_test, load_validation, read_predictions, write_predictions

from methods.geometry.parse import KIND_DIM, parse_prior, parse_question, si_to_unit
from methods.geometry.solver import fallback_value

from .run import RUNS, code_sha256, ref_constants

OLD = {"test": ROOT / "submissions" / "caw_geospeed_test.csv", "val": ROOT / "results" / "predictions" / "val_caw_geospeed.csv"}


def own_fallback(row, ref: dict) -> tuple[float, str]:
    qs = parse_question(row.question)
    v_si, kind = fallback_value(qs, parse_prior(row.prior), ref)
    return si_to_unit(v_si, row.target_unit, KIND_DIM.get(qs.get("kind"), "L")), kind


def combine(meta: pd.DataFrame, caw: pd.DataFrame, geo: pd.DataFrame | None, ref: dict) -> pd.DataFrame:
    """meta: question rows; caw: index qid (used_fallback, prediction); geo: index qid (source, prediction)
    for the speed questions, or None (insurance only). Every output row depends only on its own inputs."""
    rows = []
    for r in meta.itertuples():
        if r.qid not in caw.index:
            raise SystemExit(f"qid {r.qid} has no CaW record")
        c = caw.loc[r.qid]
        cp = float(c.prediction)
        usable = (not bool(c.used_fallback)) and math.isfinite(cp) and cp > 0
        fb, fb_kind = own_fallback(r, ref)
        caw_iq = cp if usable else fb
        out = {"qid": int(r.qid), "category": r.category, "kind": parse_question(r.question).get("kind"),
               "caw_pred": cp, "caw_usable": usable, "own_fallback": fb, "own_fallback_kind": fb_kind,
               "caw_iq_final": caw_iq, "caw_iq_src": "caw" if usable else f"own_fallback:{fb_kind}"}
        if geo is not None:
            take = False
            if out["kind"] == "speed":
                if r.qid not in geo.index:
                    raise SystemExit(f"speed question {r.qid} has no geometry_iq record")
                g = geo.loc[r.qid]
                out.update({"geo_iq_source": g.source, "geo_iq_pred": float(g.prediction)})
                take = g.source == "geometry"
            out["caw_geospeed_iq_final"] = out["geo_iq_pred"] if take else caw_iq
            out["caw_geospeed_iq_src"] = "geometry_iq:speed" if take else out["caw_iq_src"]
        rows.append(out)
    return pd.DataFrame(rows).set_index("qid")


def check_template(tmpl: pd.DataFrame, n_expected: int = 3289) -> None:
    ref = load_template()
    if len(tmpl) != n_expected or list(tmpl["id"]) != list(ref["id"]) or list(tmpl.columns) != list(ref.columns):
        raise SystemExit("submission does not match the official template (rows / ids / columns)")
    v = pd.to_numeric(tmpl.parsed_value, errors="coerce")
    if not (np.isfinite(v) & (v > 0)).all():
        raise SystemExit("non-finite or non-positive answers")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["val", "test"], required=True)
    ap.add_argument("--insurance-only", action="store_true", help="write caw_iq only (no geometry_iq needed)")
    args = ap.parse_args(argv)
    meta = load_validation().drop(columns=["answer"]) if args.split == "val" else load_test()
    caw = pd.read_csv(ROOT / "runs" / "caw" / args.split / "items.csv").set_index("qid")
    if "error" in caw and caw.error.fillna("").astype(str).ne("").any():
        raise SystemExit("CaW records with errors; see methods.caw.combine")
    ref = ref_constants()
    geo = None
    variants = ["caw_iq"]
    if not args.insurance_only:
        geo = pd.read_csv(RUNS / args.split / "items.csv").set_index("qid")
        variants.append("caw_geospeed_iq")
    df = combine(meta, caw, geo, ref)
    (RUNS / args.split).mkdir(parents=True, exist_ok=True)
    df.to_csv(RUNS / args.split / ("combined_insurance.csv" if args.insurance_only else "combined.csv"))
    old = read_predictions(OLD[args.split]).reindex(df.index).astype(float)
    old_src = pd.read_csv(ROOT / "runs" / "caw" / args.split / "combined.csv").set_index("qid").caw_geospeed_src.reindex(df.index)
    summ = {"n": int(len(df)), "reference_constants_si": ref, "code_sha256": code_sha256(),
            "caw_usable": int(df.caw_usable.sum())}
    old_geo = old_src == "geometry:speed"
    for v in variants:
        changed = ~np.isclose(df[f"{v}_final"].astype(float), old, rtol=1e-9, atol=0)
        summ[v] = {
            "source_counts": df[f"{v}_src"].value_counts().to_dict(),
            "source_counts_by_category": {c: g[f"{v}_src"].value_counts().to_dict() for c, g in df.groupby("category")},
            "changed_vs_caw_geospeed": int(changed.sum()),
            "changed_vs_caw_geospeed_among_its_geometry_speed": int((changed & old_geo).sum()),
            "changed_vs_caw_geospeed_elsewhere": int((changed & ~old_geo).sum()),
        }
        if v == "caw_geospeed_iq":
            new_geo = df[f"{v}_src"] == "geometry_iq:speed"
            summ[v].update({
                "old_geometry_now_caw": int((old_geo & (df[f"{v}_src"] == "caw")).sum()),
                "old_geometry_now_own_fallback": int((old_geo & df[f"{v}_src"].str.startswith("own_fallback")).sum()),
                "old_geometry_still_geometry": int((old_geo & new_geo).sum()),
                "old_geometry_still_geometry_value_changed": int((old_geo & new_geo & changed).sum()),
                "new_geometry_not_in_old": int((new_geo & ~old_geo).sum()),
            })
        if args.split == "val":
            p = ROOT / "results" / "predictions" / f"val_{v}.csv"
            write_predictions(p, meta, df.loc[meta.qid, f"{v}_final"].to_numpy())
            summ[v]["predictions"] = p.relative_to(ROOT).as_posix()
        else:
            tmpl = load_template()
            tmpl["parsed_value"] = df.loc[tmpl["id"], f"{v}_final"].to_numpy()
            check_template(tmpl)
            p = ROOT / "submissions" / f"{v}_test.csv"
            tmpl.to_csv(p, index=False)
            summ[v]["submission"] = p.relative_to(ROOT).as_posix()
            summ[v]["sha256"] = hashlib.sha256(p.read_bytes()).hexdigest()
    if geo is not None:
        gs = ROOT / "results" / f"geometry_iq_{args.split}_summary.json"
        summ["geometry_iq_code_sha256"] = json.loads(gs.read_text(encoding="utf-8")).get("code_sha256") if gs.exists() else None
    name = "caw_iq_insurance" if args.insurance_only else "caw_iq"
    out = ROOT / "results" / f"{name}_combine_{args.split}_summary.json"
    out.write_text(json.dumps(summ, indent=2), encoding="utf-8")
    print(json.dumps(summ, indent=1))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
