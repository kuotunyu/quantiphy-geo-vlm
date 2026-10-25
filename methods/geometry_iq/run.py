"""geometry_iq: the geometry speed answers of caw_geospeed, rebuilt so that every question is answered
from its own inputs only (README "Per-question version"; the Oct 15 rule "answer each question
independently, without using other questions or their associated answers or priors").

    uv run python -m methods.geometry_iq.run --split val  --prereg README.md
    uv run python -m methods.geometry_iq.run --split test --prereg README.md
    (--stage detect|solve|verify|decide runs one stage; default all)

Only questions whose own text parses as kind "speed" (methods.geometry.parse.parse_question) are
processed: those are the only questions where caw_geospeed used geometry. Differences from the
frozen geometry_v1 + hybrid_v2 path (methods/geometry, methods/hybrid_v2 are imported, not changed):
  1. detection (GPU, OWLv2, same weights/frames/settings): one separate detector call per question,
     whose text queries are only the object names in that question and in its own prior
     (geometry_v1: one call per video with the names of every question of the video);
  2. the cross-query box suppression (solver.VideoContext._exclusive_scores) therefore only sees
     that question's own queries; free-fall candidates are only that question's own objects;
  3. plausibility / fallback reference constants: median |prior| per dimension over the
     VALIDATION priors (geometry_v1 used the test priors), fixed for both splits;
  4. box check (GPU, Qwen3-VL-8B, same prompt/frames/threshold as hybrid_v2): every track a
     question used is checked with that question's own wording, per (question, track)
     (hybrid_v2 checked each (video, track) once, with the first question that used it);
  5. per question: source "geometry" iff geometry measured (no fallback) and no used track has
     mean p_yes < 0.5 (unchecked counts as passed) -- the hybrid_v2 rule that made v2_source ==
     "geometry". Video frames are decoded per question as well; nothing is shared between questions
     except the model weights.
Outputs under runs/geometry_iq/<split>/: det/<qid>.npz|json, items/<qid>.json, verify/<qid>.json,
items.csv; results/geometry_iq_<split>_summary.json. No answers are read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from quantiphy.data import ROOT, load_test, load_validation

from methods.geometry import detect as D
from methods.geometry.parse import parse_depth, parse_prior, parse_question
from methods.geometry.solver import VideoContext, reference_constants, solve
from methods.hybrid_v2 import tracks as T
from methods.hybrid_v2.verify import load_config, read_det_frames, render

VERSION = "geometry_iq"
RUNS = ROOT / "runs" / VERSION
MIN_FREE_GPU_GB = {"detect": 8.0, "verify": 19.0}
CODE_FILES = [
    "methods/geometry_iq/run.py", "methods/geometry/detect.py", "methods/geometry/measure.py",
    "methods/geometry/parse.py", "methods/geometry/solver.py", "methods/hybrid_v2/tracks.py",
    "methods/hybrid_v2/verify.py", "methods/hybrid_v2/config.json", "methods/vlm_baseline/run.py",
    "src/quantiphy/data.py",
]


def code_sha256() -> str:
    h = hashlib.sha256()
    for f in CODE_FILES:
        h.update(f.encode())
        h.update((ROOT / f).read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()


def _json_default(o):
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


# ---------------------------------------------------------------------------
# per-question plan: a pure function of one question row
# ---------------------------------------------------------------------------


def question_plan(row) -> dict:
    """Parsed specs, detector queries and free-fall candidates of ONE question (its own text only).

    Same order and de-duplication as geometry.run.build_plan, restricted to this question."""
    qs = parse_question(row.question)
    ps = parse_prior(row.prior)
    ds = parse_depth(row.depth_info)
    queries: list[str] = []
    gravity: list[dict] = []
    for o in qs.get("objects", []):
        if o["query"] and o["query"] not in queries:
            queries.append(o["query"])
        if o["query"] and all(o["query"] != x["query"] or o.get("spatial") != x.get("spatial") for x in gravity):
            gravity.append(o)
    for p in ps:
        for o in p.get("objects", []):
            if o["query"] and o["query"] not in queries:
                queries.append(o["query"])
    return {"qspec": qs, "pspecs": ps, "dspec": ds, "queries": queries, "gravity": gravity}


def is_speed(row) -> bool:
    return parse_question(row.question).get("kind") == "speed"


def select(df: pd.DataFrame) -> pd.DataFrame:
    return df[[is_speed(r) for r in df.itertuples()]]


def ref_constants() -> dict:
    """Fixed reference constants: median |prior| per dimension over the validation priors."""
    return reference_constants(load_validation())


# ---------------------------------------------------------------------------
# GPU guards
# ---------------------------------------------------------------------------


def gpu_guard(stage: str) -> None:
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,name", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout
    me = str(os.getpid())  # a stage after the detector shares this process's CUDA context
    others = [ln for ln in out.splitlines() if "python" in ln.lower() and ln.split(",")[0].strip() != me]
    if others:
        sys.exit(f"refusing to start {stage}: other python GPU processes are running:\n" + "\n".join(others))
    import torch

    if not torch.cuda.is_available():
        sys.exit("CUDA not available")
    free, _ = torch.cuda.mem_get_info()
    if free / 2**30 < MIN_FREE_GPU_GB[stage]:
        sys.exit(f"refusing to start {stage}: only {free / 2**30:.1f} GB free GPU memory")


# ---------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------


def stage_detect(df: pd.DataFrame, det_dir: Path) -> dict:
    todo = []
    for r in df.itertuples():
        queries = question_plan(r)["queries"] or ["object"]
        meta_p = det_dir / f"{r.qid}.json"
        if meta_p.exists():
            meta = json.loads(meta_p.read_text(encoding="utf-8"))
            if meta.get("queries") == queries or not meta.get("ok"):
                continue
        todo.append((r.qid, r.video_path, queries))
    print(f"[detect] {len(todo)} questions to process ({len(df)} selected)", flush=True)
    if not todo:
        return {"n_run": 0}
    gpu_guard("detect")
    t_load = time.time()
    det = D.OwlDetector()
    load_s = time.time() - t_load
    t0 = time.time()
    for i, (qid, path, queries) in enumerate(todo):
        out = det_dir / f"{qid}.npz"
        try:
            D.run_video(det, path, queries, out)  # one call = this question's queries only
        except Exception as e:  # the solver falls back for this question
            out.parent.mkdir(parents=True, exist_ok=True)
            out.with_suffix(".json").write_text(json.dumps({"ok": False, "reason": f"error: {e!r}"}), encoding="utf-8")
        if (i + 1) % 25 == 0 or i + 1 == len(todo):
            el = time.time() - t0
            print(f"[detect] {i + 1}/{len(todo)}, {el / 60:.1f} min, eta {el / (i + 1) * (len(todo) - i - 1) / 60:.1f} min",
                  flush=True)
    return {"n_run": len(todo), "model_load_s": round(load_s, 1), "seconds": round(time.time() - t0, 1)}


def solve_question(row, det_dir: Path, ref: dict) -> dict:
    """geometry_v1's solver on this question's own detections, candidates and fixed constants."""
    plan = question_plan(row)
    p = det_dir / f"{row.qid}.npz"
    if p.with_suffix(".json").exists():
        meta, dets = D.load_video_dets(p)
    else:
        meta, dets = {"ok": False}, None
    ctx = VideoContext(meta, dets, row.fps)
    rec = solve(row, plan["qspec"], plan["pspecs"], plan["dspec"], ctx, plan["gravity"], ref)
    rec.update({"video_id": row.video_id, "question": row.question, "prior": row.prior,
                "target_unit": row.target_unit, "queries": plan["queries"]})
    return rec


def stage_solve(df: pd.DataFrame, det_dir: Path, items_dir: Path, ref: dict) -> None:
    items_dir.mkdir(parents=True, exist_ok=True)
    for r in df.itertuples():
        rec = solve_question(r, det_dir, ref)
        (items_dir / f"{r.qid}.json").write_text(json.dumps(rec, indent=1, default=_json_default), encoding="utf-8")


def used_track_refs(row, rec: dict, det_dir: Path) -> list[T.TrackRef]:
    """Tracks this question's answer used (hybrid_v2.tracks rules), rebuilt from its own detections
    and checked against the solver's summaries. Contexts: the question text for targets, the
    question's own prior text for priors. De-duplicated by track key within the question."""
    if rec["used_fallback"]:
        return []
    plan = question_plan(row)
    meta, dets = D.load_video_dets(det_dir / f"{row.qid}.npz")
    ctx = VideoContext(meta, dets, row.fps)
    refs: list[T.TrackRef] = []
    for k, (obj, inst) in enumerate(T.target_objects(plan["qspec"])):
        ref = T._to_ref(ctx, row.video_id, obj, inst, "target", row.question)
        T._check(ref, rec["target_tracks"][k], row.qid)
        refs.append(ref)
    for i in T.used_prior_indices(rec):
        p = plan["pspecs"][i]
        if p.get("gravity"):
            continue
        ref = T._to_ref(ctx, row.video_id, p["objects"][0], 0, "prior", p["text"])
        T._check(ref, rec["prior_tracks"][i], row.qid)
        refs.append(ref)
    out, seen = [], set()
    for ref in refs:
        if ref.key not in seen:
            seen.add(ref.key)
            out.append(ref)
    return out


def stage_verify(df: pd.DataFrame, det_dir: Path, items_dir: Path, ver_dir: Path) -> dict:
    cfg = load_config()["verifier"]
    ver_dir.mkdir(parents=True, exist_ok=True)
    todo = []
    for r in df.itertuples():
        if (ver_dir / f"{r.qid}.json").exists():
            continue
        rec = json.loads((items_dir / f"{r.qid}.json").read_text(encoding="utf-8"))
        todo.append((r, used_track_refs(r, rec, det_dir)))
    n_tracks = sum(len(x) for _, x in todo)
    print(f"[verify] {len(todo)} questions / {n_tracks} tracks to check", flush=True)
    if not todo:
        return {"n_run": 0}
    ver = None
    if n_tracks:
        gpu_guard("verify")
        from methods.hybrid_v2.verify import Verifier

        t_load = time.time()
        ver = Verifier()
        load_s = time.time() - t_load
    t0, done = time.time(), 0
    for qi, (r, refs) in enumerate(todo):
        out = {"qid": int(r.qid), "video_id": r.video_id, "tracks": {}}
        if refs:
            picks = {ref.key: T.frames_to_check(ref, cfg["frame_quantiles"]) for ref in refs}
            wanted = {int(ref.frame_idx[p]) for ref in refs for p in picks[ref.key]}
            frames = read_det_frames(r.video_path, wanted)  # decoded for this question only
            meta = json.loads((det_dir / f"{r.qid}.json").read_text(encoding="utf-8"))
            h, w = int(meta["height"]), int(meta["width"])
            for ref in refs:
                per = []
                for p in picks[ref.key]:
                    fi = int(ref.frame_idx[p])
                    fr = frames.get(fi)
                    if fr is None or fr.shape[:2] != (h, w):
                        per.append({"frame_idx": fi, "error": "frame_missing_or_size_mismatch"})
                        continue
                    full, crop = render(fr, ref.box[p], cfg["crop_side"], cfg["crop_margin"])
                    py, mass = ver.p_yes(full, crop, ref.query, ref.context)
                    per.append({"frame_idx": fi, "box": [round(float(x), 1) for x in ref.box[p]],
                                "det_score": round(float(ref.score[p]), 4), "p_yes": py, "yes_no_mass": mass})
                ok = [x["p_yes"] for x in per if "p_yes" in x]
                out["tracks"][ref.key] = {"query": ref.query, "spatial": ref.spatial, "instance": ref.instance,
                                          "role": ref.role, "context": ref.context,
                                          "n_track_frames": int(len(ref.frame_idx)), "frames": per,
                                          "mean_p_yes": float(np.mean(ok)) if ok else None}
                done += 1
            ver.torch.cuda.empty_cache()
        (ver_dir / f"{r.qid}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
        if (qi + 1) % 25 == 0 or qi + 1 == len(todo):
            el = time.time() - t0
            print(f"[verify] {qi + 1}/{len(todo)} questions, {done}/{n_tracks} tracks, {el / 60:.1f} min, "
                  f"eta {el / max(done, 1) * (n_tracks - done) / 60:.1f} min", flush=True)
    res = {"n_run": len(todo), "n_tracks": n_tracks, "seconds": round(time.time() - t0, 1)}
    if ver is not None:
        res["model_load_s"] = round(load_s, 1)
    return res


def decide_question(rec: dict, ver: dict, threshold: float) -> dict:
    """The hybrid_v2 rule that produced v2_source == "geometry", on one question's own records."""
    p = [t["mean_p_yes"] for t in ver.get("tracks", {}).values()]
    n_rej = sum(1 for x in p if x is not None and x < threshold)  # unchecked (None) counts as passed
    if rec["used_fallback"]:
        source = "not_geometry:fell_back"
    elif n_rej:
        source = "not_geometry:box_rejected"
    else:
        source = "geometry"
    return {"n_tracks": len(p), "n_rejected": n_rej, "min_p_yes": min([x for x in p if x is not None], default=None),
            "source": source}


def stage_decide(df: pd.DataFrame, items_dir: Path, ver_dir: Path) -> pd.DataFrame:
    thr = float(load_config()["rule"]["p_yes_threshold"])
    rows = []
    for r in df.itertuples():
        rec = json.loads((items_dir / f"{r.qid}.json").read_text(encoding="utf-8"))
        ver = json.loads((ver_dir / f"{r.qid}.json").read_text(encoding="utf-8"))
        d = decide_question(rec, ver, thr)
        rows.append({"qid": int(r.qid), "category": r.category, "video_id": r.video_id,
                     "kind": rec["question_spec"].get("kind"), "method": rec["method"],
                     "used_fallback": rec["used_fallback"], "reason": rec["reason"],
                     "prediction": rec["prediction"], **d})
    return pd.DataFrame(rows)


def check_prereg(prereg: str) -> dict:
    p = ROOT / prereg
    dirty = subprocess.run(["git", "status", "--porcelain", "--", prereg, "methods/geometry_iq", "methods/geometry",
                            "methods/hybrid_v2", "methods/vlm_baseline", "src/quantiphy"],
                           cwd=ROOT, capture_output=True, text=True).stdout.strip()
    if not p.exists() or dirty:
        sys.exit(f"plan / code not committed (freeze first):\n{dirty or prereg + ' missing'}")
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    return {"prereg": prereg, "prereg_sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "commit": head}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["val", "test"], required=True)
    ap.add_argument("--stage", choices=["all", "detect", "solve", "verify", "decide"], default="all")
    ap.add_argument("--prereg", required=True, help="committed, unmodified pre-registration (README.md)")
    args = ap.parse_args(argv)
    freeze = check_prereg(args.prereg)
    df = load_validation().drop(columns=["answer"]) if args.split == "val" else load_test()
    df = select(df)
    base = RUNS / args.split
    det_dir, items_dir, ver_dir = base / "det", base / "items", base / "verify"
    ref = ref_constants()
    summ_p = ROOT / "results" / f"{VERSION}_{args.split}_summary.json"
    summ = json.loads(summ_p.read_text(encoding="utf-8")) if summ_p.exists() else {"stages": {}}
    summ.update({"version": VERSION, "n_selected": int(len(df)), "reference_constants_si": ref,
                 "reference_constants_source": "validation priors", "code_sha256": code_sha256(), **freeze})
    stages = ["detect", "solve", "verify", "decide"] if args.stage == "all" else [args.stage]
    for st in stages:
        t0 = time.time()
        if st == "detect":
            info = stage_detect(df, det_dir)
        elif st == "solve":
            stage_solve(df, det_dir, items_dir, ref)
            info = {}
        elif st == "verify":
            info = stage_verify(df, det_dir, items_dir, ver_dir)
        else:
            items = stage_decide(df, items_dir, ver_dir)
            items.to_csv(base / "items.csv", index=False)
            info = {"source_counts": items.source.value_counts().to_dict(),
                    "source_counts_by_category": {c: g.source.value_counts().to_dict() for c, g in items.groupby("category")},
                    "method_counts": items.method.value_counts().to_dict()}
        info["wall_s"] = round(time.time() - t0, 1)
        info["code_sha256"] = summ["code_sha256"]
        summ["stages"][st] = info
        summ_p.write_text(json.dumps(summ, indent=2, default=_json_default), encoding="utf-8")
        print(f"[{st}] done: {json.dumps(info, default=_json_default)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
