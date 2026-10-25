"""GPU spot check for stage 13 (README "Per-question version"): a question's OWLv2 detections do not depend on
which other detector calls ran before it.

    uv run python scripts/check_geometry_iq_detect.py --split test --n 12

For a sample of speed questions whose video has other questions, re-runs the detector for that question
alone, in reverse order, each time right after a call with the OTHER questions' pooled names on the same
video, and compares with runs/geometry_iq/<split>/det/<qid>.npz. Writes nothing under runs/; prints and
saves results/geometry_iq_detect_check_<split>.json.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from quantiphy.data import ROOT, load_test, load_validation

from methods.geometry import detect as D
from methods.geometry_iq import run as R


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["val", "test"], default="test")
    ap.add_argument("--n", type=int, default=12)
    args = ap.parse_args()
    df = load_validation().drop(columns=["answer"]) if args.split == "val" else load_test()
    counts = df.video_id.value_counts()
    sp = R.select(df)
    sp = sp[sp.video_id.map(counts) >= 2]
    sample = sp.sample(min(args.n, len(sp)), random_state=20261024)
    R.gpu_guard("detect")
    det = D.OwlDetector()
    rows = []
    for r in list(sample.itertuples())[::-1]:
        others = df[(df.video_id == r.video_id) & (df.qid != r.qid)]
        pooled = []
        for o in others.itertuples():
            for q in R.question_plan(o)["queries"]:
                if q not in pooled:
                    pooled.append(q)
        idx, imgs, *_ = D.read_frames(r.video_path)
        if pooled:
            det.detect(imgs, pooled)  # an unrelated call first
        q = R.question_plan(r)["queries"] or ["object"]
        b, s = det.detect(imgs, q)
        z = np.load(R.RUNS / args.split / "det" / f"{r.qid}.npz")
        rows.append({"qid": int(r.qid), "queries": q, "other_names": pooled,
                     "boxes_identical": bool(np.array_equal(b, z["boxes"])),
                     "scores_identical": bool(np.array_equal(s, z["scores"])),
                     "max_abs_score_diff": float(np.abs(s - z["scores"]).max()),
                     "max_abs_box_diff_px": float(np.abs(b - z["boxes"]).max())})
        print(rows[-1], flush=True)
    out = ROOT / "results" / f"geometry_iq_detect_check_{args.split}.json"
    out.write_text(json.dumps({"n": len(rows), "all_identical": all(x["boxes_identical"] and x["scores_identical"] for x in rows),
                               "rows": rows}, indent=1), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
