"""The single pre-registered validation scoring of stage 13 (README "Per-question version"): caw_geospeed_iq and caw_iq.

Run once, after the freeze commit and after
    uv run python -m methods.geometry_iq.run --split val --prereg README.md
    uv run python -m methods.geometry_iq.combine --split val
then:
    uv run python scripts/eval_caw_iq_val.py

* Refuses if results/caw_iq_val.json exists (score once), if the plan or code (methods/geometry_iq,
  methods/geometry, methods/hybrid_v2, methods/caw, src/quantiphy, this script) has uncommitted changes,
  or if a prediction file does not cover exactly the 159 validation questions.
* Nothing here decides anything: the plan uploads caw_geospeed_iq to Track B whatever its score.
* Reported: macro / per-category MRA of caw_geospeed_iq, caw_iq, and the old caw_geospeed (58.42) and caw
  (55.18), recomputed and checked against results/caw_val.json; video-level paired bootstrap (10,000, seed
  20260923) for caw_geospeed_iq - caw_geospeed, caw_iq - caw, caw_geospeed_iq - caw_iq.
Writes results/caw_iq_val.json (with the commit hash).
"""

from __future__ import annotations

import json
import subprocess
import sys

from quantiphy.bootstrap import paired_bootstrap, paired_video_bootstrap
from quantiphy.data import CATEGORIES, ROOT, load_validation, read_predictions
from quantiphy.mra import score

N_BOOT = 10_000
SEED = 20260923
P = ROOT / "results" / "predictions"
SYSTEMS = {
    "caw_geospeed_iq": P / "val_caw_geospeed_iq.csv",
    "caw_iq": P / "val_caw_iq.csv",
    "caw_geospeed": P / "val_caw_geospeed.csv",
    "caw": P / "val_caw.csv",
}
KEYS = ("macro_mra_a", "macro_mra_b", "diff_a_minus_b", "ci_low", "ci_high", "p_two_sided")
OUT = ROOT / "results" / "caw_iq_val.json"


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def main() -> int:
    if OUT.exists():
        sys.exit(f"{OUT.relative_to(ROOT)} exists: stage 13 was already scored once on validation")
    dirty = git("status", "--porcelain", "--", "README.md", "methods/geometry_iq", "methods/geometry",
                "methods/hybrid_v2", "methods/caw", "src/quantiphy", "scripts/eval_caw_iq_val.py")
    if dirty:
        sys.exit(f"uncommitted changes; commit (freeze) before scoring:\n{dirty}")
    val = load_validation()
    res = {k: score(val, read_predictions(p), missing="zero") for k, p in SYSTEMS.items()}
    for k, r in res.items():
        if r.n_missing_predictions or r.n_extra_predictions:
            sys.exit(f"{k}: {r.n_missing_predictions} missing / {r.n_extra_predictions} extra questions; refusing to score")
    old = json.loads((ROOT / "results" / "caw_val.json").read_text(encoding="utf-8"))["systems"]
    for k in ("caw_geospeed", "caw"):
        if abs(res[k].macro - old[k]["macro_mra"]) > 1e-12:
            sys.exit(f"{k} recomputed {res[k].macro} differs from results/caw_val.json {old[k]['macro_mra']}")
    qids = res["caw"].items["qid"].to_numpy()
    cats = res["caw"].items["category"].to_numpy()
    vids = val.set_index("qid").loc[qids, "video_id"].to_numpy()
    mra = {k: r.items.set_index("qid").loc[qids, "mra"].to_numpy(dtype=float) for k, r in res.items()}
    comps = {}
    for a, b in [("caw_geospeed_iq", "caw_geospeed"), ("caw_iq", "caw"), ("caw_geospeed_iq", "caw_iq")]:
        v = paired_video_bootstrap(mra[a], mra[b], cats, vids, n_boot=N_BOOT, seed=SEED)
        q = paired_bootstrap(mra[a], mra[b], cats, n_boot=N_BOOT, seed=SEED)
        comps[f"{a}-{b}"] = {"video_level": {k: v[k] for k in KEYS}, "question_level": {k: q[k] for k in KEYS}}
    out = {
        "plan": "README.md",
        "commit": git("rev-parse", "HEAD"),
        "systems": {k: {"predictions": SYSTEMS[k].relative_to(ROOT).as_posix(), "macro_mra": r.macro,
                        "per_category_mra": r.per_category} for k, r in res.items()},
        "comparisons": comps,
        "inputs": {name: {k: v for k, v in json.loads((ROOT / "results" / f"{name}.json").read_text(encoding="utf-8")).items()
                          if k in ("prereg", "prereg_sha256", "commit", "code_sha256", "geometry_iq_code_sha256")}
                   for name in ("geometry_iq_val_summary", "caw_iq_combine_val_summary")},
        "settings": {"n_boot": N_BOOT, "seed": SEED},
    }
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    for k, r in res.items():
        print(f"{k:<16} macro {r.macro * 100:6.2f} | " + " ".join(f"{c} {r.per_category[c] * 100:5.1f}" for c in CATEGORIES))
    for name, c in comps.items():
        v = c["video_level"]
        print(f"{name:<30} {v['diff_a_minus_b'] * 100:+.2f}  video CI [{v['ci_low'] * 100:+.2f}, {v['ci_high'] * 100:+.2f}]")
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
