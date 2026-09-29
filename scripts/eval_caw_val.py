"""The single pre-registered validation scoring of the Code-as-World-VL-9B Track B variants.

The pre-registration plan it was run under is an internal document that is not part of this
repository (the file name recorded in the output, docs/PHASE11_PLAN.md, is kept as it was).
Run once, after the plan is committed and after
    uv run --project envs/caw python -m methods.caw.run --split val --prereg <committed plan file>
    uv run python -m methods.caw.combine --split val
then:
    uv run python scripts/eval_caw_val.py

* Refuses to run if results/caw_val.json already exists (score once) or if the scoring or combining
  code (methods/caw, methods/common, methods/geometry, src/quantiphy, this script) or the fallback
  predictions (results/predictions/val_hybrid_v4.csv) have uncommitted changes, or if any prediction
  file does not cover exactly the 159 validation questions.
* Also reads results/hybrid_v4_val.json (hybrid_v4's validation score, written by a scoring script
  that is not part of this repository) and results/caw_val_summary.json /
  results/caw_combine_val_summary.json.
* Gate (pre-registered): caw macro MRA >= 0.5007, the literal threshold (hybrid_v4 is 0.50062) -> the test run may go ahead.
* Reported comparison (not a criterion; the gate is the only rule): caw - hybrid_v4, video-level paired bootstrap (videos resampled
  within category), 10,000 replicates, seed 20260923, as every other comparison in this repo.
* Report only: caw_authors (the authors' parser, no fallback) against the paper's 55.4
  (2S 55.0 / 2D 52.9 / 3S 55.6 / 3D 58.1); caw_geospeed - caw (its speed rule came from validation).
Writes results/caw_val.json (with the commit hash).
"""

from __future__ import annotations

import json
import subprocess
import sys

from quantiphy.bootstrap import paired_bootstrap, paired_video_bootstrap
from quantiphy.data import CATEGORIES, ROOT, load_validation, read_predictions
from quantiphy.mra import score
from quantiphy.parity import compare_with_official

N_BOOT = 10_000
SEED = 20260923
GATE = 0.5007  # strictly above hybrid_v4's validation macro MRA 0.5006 (results/hybrid_v4_val.json)
PAPER = {"macro": 0.554, "2S": 0.550, "2D": 0.529, "3S": 0.556, "3D": 0.581}  # arXiv 2608.27549 Table 1, 9B
P = ROOT / "results" / "predictions"
SYSTEMS = {
    "caw": P / "val_caw.csv",
    "caw_geospeed": P / "val_caw_geospeed.csv",
    "caw_authors": P / "val_caw_authors.csv",
    "hybrid_v4": P / "val_hybrid_v4.csv",
}
KEYS = ("macro_mra_a", "macro_mra_b", "diff_a_minus_b", "ci_low", "ci_high", "p_two_sided")
OUT = ROOT / "results" / "caw_val.json"


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def main() -> int:
    if OUT.exists():
        sys.exit(f"{OUT.relative_to(ROOT)} exists: the caw variants were already scored once on validation")
    dirty = git("status", "--porcelain", "--", "methods/caw", "methods/common", "methods/geometry", "src/quantiphy",
                "scripts/eval_caw_val.py", "results/predictions/val_hybrid_v4.csv")
    if dirty:
        sys.exit(f"uncommitted changes; commit (freeze) before scoring:\n{dirty}")
    val = load_validation()
    # caw_authors keeps blank cells where the authors' parser found no number: those score 0, as in the paper
    res = {k: score(val, read_predictions(p), missing="zero") for k, p in SYSTEMS.items()}
    for k, r in res.items():
        if r.n_missing_predictions or r.n_extra_predictions:
            sys.exit(f"{k}: {r.n_missing_predictions} missing / {r.n_extra_predictions} extra questions; refusing to score")
    ref = json.loads((ROOT / "results" / "hybrid_v4_val.json").read_text(encoding="utf-8"))["systems"]["hybrid_v4"]["macro_mra"]
    if abs(res["hybrid_v4"].macro - ref) > 1e-12:
        sys.exit(f"hybrid_v4 validation macro {res['hybrid_v4'].macro} differs from results/hybrid_v4_val.json {ref}")
    qids = res["hybrid_v4"].items["qid"].to_numpy()
    cats = res["hybrid_v4"].items["category"].to_numpy()
    vids = val.set_index("qid").loc[qids, "video_id"].to_numpy()
    mra = {k: r.items.set_index("qid").loc[qids, "mra"].to_numpy(dtype=float) for k, r in res.items()}
    comps = {}
    for a, b in [("caw", "hybrid_v4"), ("caw_geospeed", "caw"), ("caw_geospeed", "hybrid_v4")]:
        v = paired_video_bootstrap(mra[a], mra[b], cats, vids, n_boot=N_BOOT, seed=SEED)
        q = paired_bootstrap(mra[a], mra[b], cats, n_boot=N_BOOT, seed=SEED)
        comps[f"{a}-{b}"] = {"video_level": {k: v[k] for k in KEYS}, "question_level": {k: q[k] for k in KEYS}}
    gate_pass = bool(res["caw"].macro >= GATE)
    out = {
        "plan": "docs/PHASE11_PLAN.md",
        "commit": git("rev-parse", "HEAD"),
        "gate": {"system": "caw", "threshold": GATE, "macro_mra": res["caw"].macro, "pass": gate_pass},
        "systems": {k: {"predictions": SYSTEMS[k].relative_to(ROOT).as_posix(), "macro_mra": r.macro,
                        "per_category_mra": r.per_category} for k, r in res.items()},
        "caw_matches_official_scorer": compare_with_official(SYSTEMS["caw"])["match"],
        "comparisons": comps,
        "fidelity_report_only": {"caw_authors": {"macro": res["caw_authors"].macro, **res["caw_authors"].per_category},
                                 "paper_9b": PAPER,
                                 "diff_vs_paper": {"macro": res["caw_authors"].macro - PAPER["macro"],
                                                   **{c: res["caw_authors"].per_category[c] - PAPER[c] for c in CATEGORIES}}},
        "inputs": {name: {k: v for k, v in json.loads((ROOT / "results" / f"{name}.json").read_text(encoding="utf-8")).items()
                          if k in ("prereg", "code", "run_code", "accepted_failed_videos")}
                   for name in ("caw_val_summary", "caw_combine_val_summary")},
        "settings": {"n_boot": N_BOOT, "seed": SEED},
    }
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    for k, r in res.items():
        print(f"{k:<14} macro {r.macro * 100:6.2f} | " + " ".join(f"{c} {r.per_category[c] * 100:5.1f}" for c in CATEGORIES))
    for name, c in comps.items():
        v = c["video_level"]
        print(f"{name:<24} {v['diff_a_minus_b'] * 100:+.2f}  video CI [{v['ci_low'] * 100:+.2f}, {v['ci_high'] * 100:+.2f}]")
    print(f"gate (caw >= {GATE}): {'PASS' if gate_pass else 'FAIL'}")
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
