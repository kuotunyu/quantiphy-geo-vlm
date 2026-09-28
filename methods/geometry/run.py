"""One-command runner for the geometry method v1.

    uv run python -m methods.geometry.run --split val
    uv run python -m methods.geometry.run --split test

Stage 1 (GPU, cached): OWLv2 detections per video -> runs/geometry_v1/<split>/det/
Stage 2 (CPU): per-question solve -> runs/geometry_v1/<split>/items/<qid>.json
Outputs:
  val : results/predictions/val_geometry_v1.csv, results/geometry_v1_val_summary.json
  test: submissions/geometry_v1_test.csv (official template, only parsed_value filled),
        results/geometry_v1_test_summary.json
Predictions are produced 100% by this program; validation answers are only read
by the scoring step at the very end (never by the solver).
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from quantiphy.data import ROOT, load_template, load_test, load_validation, write_predictions

from . import detect as D
from .parse import parse_depth, parse_prior, parse_question
from .solver import VideoContext, reference_constants, solve

VERSION = "geometry_v1"
RUNS = ROOT / "runs" / VERSION
MIN_FREE_GPU_GB = 8.0


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def build_plan(df: pd.DataFrame) -> tuple[dict, dict, dict, dict]:
    """Parse everything; collect the detector queries each video needs."""
    qspecs, pspecs, dspecs = {}, {}, {}
    video_queries: dict[str, list[str]] = collections.defaultdict(list)
    video_objs: dict[str, list[dict]] = collections.defaultdict(list)
    for r in df.itertuples():
        qs = parse_question(r.question)
        ps = parse_prior(r.prior)
        ds = parse_depth(r.depth_info)
        qspecs[r.qid], pspecs[r.qid], dspecs[r.qid] = qs, ps, ds
        for o in qs.get("objects", []):
            if o["query"] and o["query"] not in video_queries[r.video_id]:
                video_queries[r.video_id].append(o["query"])
            if o["query"] and all(o["query"] != x["query"] or o.get("spatial") != x.get("spatial") for x in video_objs[r.video_id]):
                video_objs[r.video_id].append(o)
        for p in ps:
            for o in p.get("objects", []):
                if o["query"] and o["query"] not in video_queries[r.video_id]:
                    video_queries[r.video_id].append(o["query"])
    return qspecs, pspecs, dspecs, {"queries": dict(video_queries), "objects": dict(video_objs)}


def gpu_check() -> None:
    import torch

    if not torch.cuda.is_available():
        sys.exit("CUDA not available")
    free, total = torch.cuda.mem_get_info()
    if free / 2**30 < MIN_FREE_GPU_GB:
        sys.exit(f"refusing to start: only {free / 2**30:.1f} GB free GPU memory (another GPU job running?)")


def stage_detect(df: pd.DataFrame, plan: dict, det_dir: Path, force: bool = False) -> dict:
    todo = []
    for vid, g in df.groupby("video_id", sort=True):
        queries = plan["queries"].get(vid, [])
        out = det_dir / f"{vid}.npz"
        meta_p = out.with_suffix(".json")
        if not force and meta_p.exists():
            meta = json.loads(meta_p.read_text(encoding="utf-8"))
            if meta.get("queries") == queries or not meta.get("ok"):
                continue
        todo.append((vid, g.iloc[0].video_path, queries, out))
    print(f"[detect] {len(todo)} videos to process ({df.video_id.nunique()} total)")
    if not todo:
        return {"n_videos_run": 0}
    gpu_check()
    det = D.OwlDetector()
    t0 = time.time()
    for i, (vid, path, queries, out) in enumerate(todo):
        if not queries:
            queries = ["object"]
        try:
            m = D.run_video(det, path, queries, out)
        except Exception as e:  # keep going; the solver will fall back for this video
            m = {"ok": False, "reason": f"error: {e!r}"}
            out.parent.mkdir(parents=True, exist_ok=True)
            out.with_suffix(".json").write_text(json.dumps(m), encoding="utf-8")
        if (i + 1) % 10 == 0 or i + 1 == len(todo):
            el = time.time() - t0
            print(f"[detect] {i + 1}/{len(todo)} videos, {el / 60:.1f} min elapsed, "
                  f"eta {el / (i + 1) * (len(todo) - i - 1) / 60:.1f} min", flush=True)
    return {"n_videos_run": len(todo), "seconds": round(time.time() - t0, 1)}


def stage_solve(df: pd.DataFrame, plan: dict, qspecs, pspecs, dspecs, det_dir: Path, items_dir: Path, ref: dict) -> pd.DataFrame:
    items_dir.mkdir(parents=True, exist_ok=True)
    ctxs: dict[str, VideoContext] = {}
    rows = []
    for r in df.itertuples():
        if r.video_id not in ctxs:
            p = det_dir / f"{r.video_id}.npz"
            if p.with_suffix(".json").exists():
                meta, dets = D.load_video_dets(p)
            else:
                meta, dets = {"ok": False}, None
            ctxs[r.video_id] = VideoContext(meta, dets, r.fps)
        grav = plan["objects"].get(r.video_id, [])
        rec = solve(r, qspecs[r.qid], pspecs[r.qid], dspecs[r.qid], ctxs[r.video_id], grav, ref)
        rec.update({"video_id": r.video_id, "question": r.question, "prior": r.prior, "target_unit": r.target_unit})
        (items_dir / f"{r.qid}.json").write_text(json.dumps(rec, indent=1, default=_json_default), encoding="utf-8")
        rows.append({
            "qid": r.qid, "category": r.category, "video_id": r.video_id, "kind": rec["question_spec"].get("kind"),
            "method": rec["method"], "used_fallback": rec["used_fallback"], "reason": rec["reason"],
            "flags": ";".join(sorted(set(rec["flags"]))), "prediction": rec["prediction"],
        })
    return pd.DataFrame(rows)


def summarize(items: pd.DataFrame) -> dict:
    out = {
        "n": int(len(items)),
        "fallback_rate": float(items.used_fallback.mean()),
        "fallback_rate_by_category": items.groupby("category").used_fallback.mean().round(4).to_dict(),
        "method_counts": items.method.value_counts().to_dict(),
        "failure_reasons": items.reason.dropna().value_counts().to_dict(),
        "failure_reasons_by_category": {
            c: g.reason.dropna().value_counts().to_dict() for c, g in items.groupby("category")
        },
        "flag_counts": collections.Counter(
            f for fl in items["flags"].fillna("") for f in fl.split(";") if f
        ),
    }
    out["flag_counts"] = dict(out["flag_counts"])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["val", "test"], required=True)
    ap.add_argument("--stage", choices=["all", "detect", "solve"], default="all")
    ap.add_argument("--force-detect", action="store_true")
    args = ap.parse_args()

    df = load_validation() if args.split == "val" else load_test()
    if args.split == "val":
        df = df.drop(columns=["answer"])  # the solver never sees answers
    test_df = load_test()
    ref = reference_constants(test_df)
    qspecs, pspecs, dspecs, plan = build_plan(df)
    det_dir = RUNS / args.split / "det"
    items_dir = RUNS / args.split / "items"
    det_info = {}
    if args.stage in ("all", "detect"):
        det_info = stage_detect(df, plan, det_dir, force=args.force_detect)
    if args.stage == "detect":
        return 0

    items = stage_solve(df, plan, qspecs, pspecs, dspecs, det_dir, items_dir, ref)
    items.to_csv(RUNS / args.split / "items.csv", index=False)
    summ = summarize(items)
    summ.update({"version": VERSION, "detector": D.MODEL_ID, "reference_constants_si": ref, "detect_run": det_info})

    if args.split == "val":
        pred_path = ROOT / "results" / "predictions" / f"val_{VERSION}.csv"
        val = load_validation()
        preds = items.set_index("qid").loc[val.qid, "prediction"].to_numpy()
        write_predictions(pred_path, val, preds)
        summ["predictions"] = pred_path.relative_to(ROOT).as_posix()
        # fallback-only ablation: same pipeline output but every question answered by its fallback
        fb = []
        for q in val.qid:
            rec = json.loads((items_dir / f"{q}.json").read_text(encoding="utf-8"))
            from .parse import KIND_DIM, si_to_unit

            fb.append(si_to_unit(rec["fallback_si"], rec["target_unit"], KIND_DIM.get(rec["question_spec"].get("kind"), "L")))
        fb_path = ROOT / "results" / "predictions" / f"val_{VERSION}_fallback_only.csv"
        write_predictions(fb_path, val, fb)
        summ["fallback_only_predictions"] = fb_path.relative_to(ROOT).as_posix()
        out = ROOT / "results" / f"{VERSION}_val_summary.json"
    else:
        tmpl = load_template()
        pred = items.set_index("qid").loc[tmpl["id"], "prediction"].to_numpy()
        tmpl["parsed_value"] = pred
        sub = ROOT / "submissions" / f"{VERSION}_test.csv"
        sub.parent.mkdir(parents=True, exist_ok=True)
        tmpl.to_csv(sub, index=False)
        summ["submission"] = sub.relative_to(ROOT).as_posix()
        out = ROOT / "results" / f"{VERSION}_test_summary.json"
    out.write_text(json.dumps(summ, indent=2, default=_json_default), encoding="utf-8")
    print(json.dumps({k: summ[k] for k in ("n", "fallback_rate", "fallback_rate_by_category", "method_counts", "failure_reasons")}, indent=1, default=_json_default))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
