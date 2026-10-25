"""Independence tests for methods/geometry_iq (README "Per-question version"): a question's answer must not change
when the other questions of the same video are removed or altered.

The tests on real runs (runs/geometry_iq/<split>/...) are skipped until those runs exist; the
plan / combine tests use the real question tables and the cached CaW records."""

from __future__ import annotations

import json
import shutil

import numpy as np
import pandas as pd
import pytest

from quantiphy.data import ROOT, load_test, load_validation

from methods.geometry.parse import parse_prior
from methods.geometry.run import build_plan
from methods.geometry.solver import reference_constants
from methods.geometry_iq import run as R
from methods.geometry_iq.combine import combine

SEED = 20261024


def _multi_question_speed(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Up to n speed questions (spread over videos) whose video has at least 2 questions."""
    counts = df.video_id.value_counts()
    sp = R.select(df)
    sp = sp[sp.video_id.map(counts) >= 2]
    return sp.sample(min(n, len(sp)), random_state=SEED) if len(sp) > n else sp


def _alter_others(df: pd.DataFrame, qid: int) -> pd.DataFrame:
    """Same table, but every OTHER question of qid's video gets a different question and prior."""
    vid = df.loc[df.qid == qid, "video_id"].iloc[0]
    out = df.copy()
    m = (out.video_id == vid) & (out.qid != qid)
    out.loc[m, "question"] = "What is the velocity of the yoga ball at 1.0s in m/s?"
    out.loc[m, "prior"] = "the falling soccer ball accelerates at 9.8 m/s^2"
    return out


@pytest.fixture(scope="module")
def tables():
    return {"val": load_validation().drop(columns=["answer"]), "test": load_test()}


def test_reference_constants_come_from_validation_only():
    assert R.ref_constants() == reference_constants(load_validation())
    # the test priors give different medians, so this really changed (geometry_v1 used the test priors)
    assert R.ref_constants() != reference_constants(load_test())


@pytest.mark.parametrize("split", ["val", "test"])
def test_plan_depends_only_on_own_row(tables, split):
    df = tables[split]
    sample = _multi_question_speed(df, 60)
    assert len(sample) > 0
    _, _, _, old = build_plan(df)
    n_fewer = 0
    for r in sample.itertuples():
        p_full = R.question_plan(r)  # row taken from the full table
        alone = df[df.qid == r.qid]
        altered = _alter_others(df, r.qid)
        p_alone = R.question_plan(next(alone.itertuples()))
        p_alt = R.question_plan(next(altered[altered.qid == r.qid].itertuples()))
        assert p_full["queries"] == p_alone["queries"] == p_alt["queries"]
        assert p_full["gravity"] == p_alone["gravity"] == p_alt["gravity"]
        # own queries: only names from this question and its own prior, a subset of the old per-video list
        own = {o["query"] for o in p_full["qspec"].get("objects", []) if o["query"]}
        own |= {o["query"] for p in parse_prior(r.prior) for o in p.get("objects", []) if o["query"]}
        assert set(p_full["queries"]) == own
        assert set(p_full["queries"]) <= set(old["queries"].get(r.video_id, []))
        n_fewer += len(p_full["queries"]) < len(old["queries"].get(r.video_id, []))
    assert n_fewer > 0  # the old path really did pool names from other questions


def test_select_is_per_row(tables):
    df = tables["test"]
    s1 = set(R.select(df).qid)
    s2 = {q for q in df.qid if len(R.select(df[df.qid == q])) == 1}
    assert s1 == s2


def _runs_exist(split: str) -> bool:
    return (R.RUNS / split / "items.csv").exists()


@pytest.mark.parametrize("split", ["val", "test"])
def test_solve_unchanged_when_other_questions_removed_or_altered(tables, split, tmp_path):
    if not _runs_exist(split):
        pytest.skip(f"runs/geometry_iq/{split} not built yet")
    df = R.select(tables[split])
    sample = _multi_question_speed(tables[split], 40)
    ref = R.ref_constants()
    det = R.RUNS / split / "det"
    for r in sample.itertuples():
        base = json.loads((R.RUNS / split / "items" / f"{r.qid}.json").read_text(encoding="utf-8"))
        # 1. only this question's detection file present (the others removed)
        d1 = tmp_path / f"alone_{r.qid}"
        d1.mkdir()
        for ext in (".npz", ".json"):
            if (det / f"{r.qid}{ext}").exists():
                shutil.copy(det / f"{r.qid}{ext}", d1 / f"{r.qid}{ext}")
        a = R.solve_question(r, d1, ref)
        # 2. the other questions of the video altered: their rows rewritten and their detections garbled
        alt = _alter_others(tables[split], r.qid)
        r2 = next(alt[alt.qid == r.qid].itertuples())
        d2 = tmp_path / f"alt_{r.qid}"
        shutil.copytree(d1, d2)
        for o in df[(df.video_id == r.video_id) & (df.qid != r.qid)].qid:
            np.savez_compressed(d2 / f"{o}.npz", frame_idx=np.arange(3), boxes=np.zeros((3, 1, 8, 4)),
                                scores=np.ones((3, 1, 8)))
            (d2 / f"{o}.json").write_text(json.dumps({"ok": True, "queries": ["x"], "width": 9, "height": 9}))
        b = R.solve_question(r2, d2, ref)
        assert a["prediction"] == b["prediction"] == base["prediction"], r.qid
        assert a["method"] == b["method"] == base["method"]


@pytest.mark.parametrize("split", ["val", "test"])
def test_decision_uses_only_own_records(tables, split):
    if not _runs_exist(split):
        pytest.skip(f"runs/geometry_iq/{split} not built yet")
    items = pd.read_csv(R.RUNS / split / "items.csv").set_index("qid")
    sample = _multi_question_speed(tables[split], 40)
    thr = float(R.load_config()["rule"]["p_yes_threshold"])
    det = R.RUNS / split / "det"
    for r in sample.itertuples():
        rec = json.loads((R.RUNS / split / "items" / f"{r.qid}.json").read_text(encoding="utf-8"))
        ver = json.loads((R.RUNS / split / "verify" / f"{r.qid}.json").read_text(encoding="utf-8"))
        assert R.decide_question(rec, ver, thr)["source"] == items.loc[r.qid, "source"]
        # the tracks checked are exactly the ones rebuilt from this question alone, with its own wording
        refs = R.used_track_refs(r, rec, det)
        assert sorted(ref.key for ref in refs) == sorted(ver["tracks"])
        for ref in refs:
            assert ref.context in (r.question, *[p["text"] for p in parse_prior(r.prior)])
            assert ver["tracks"][ref.key]["context"] == ref.context


@pytest.mark.parametrize("split", ["val", "test"])
def test_combine_row_unchanged_when_other_questions_altered(tables, split):
    meta = tables[split]
    caw = pd.read_csv(ROOT / "runs" / "caw" / split / "items.csv").set_index("qid")
    ref = R.ref_constants()
    sp = R.select(meta)
    rng = np.random.default_rng(SEED)
    geo = pd.DataFrame({"qid": sp.qid, "source": rng.choice(["geometry", "not_geometry:fell_back"], len(sp)),
                        "prediction": rng.uniform(0.1, 10, len(sp))}).set_index("qid")
    full = combine(meta, caw, geo, ref)
    for r in _multi_question_speed(meta, 15).itertuples():
        others = meta[(meta.video_id == r.video_id) & (meta.qid != r.qid)].qid
        alt_meta = _alter_others(meta, r.qid)
        alt_caw = caw.copy()
        alt_caw.loc[others, "prediction"] = 123.0
        alt_caw.loc[others, "used_fallback"] = True
        alt_geo = geo.reindex(geo.index.union(others)).copy()  # the altered others are all speed questions now
        alt_geo.loc[others, "source"] = "geometry"
        alt_geo.loc[others, "prediction"] = 7.0
        # altered others, and this question alone
        alone = combine(meta[meta.qid == r.qid], caw.loc[[r.qid]], geo.loc[[r.qid]], ref)
        alt_meta = pd.concat([alt_meta[alt_meta.qid != r.qid], alt_meta[alt_meta.qid == r.qid]])  # order changed too
        alt = combine(alt_meta[alt_meta.video_id == r.video_id], alt_caw, alt_geo, ref)
        for col in ("caw_iq_final", "caw_geospeed_iq_final", "caw_geospeed_iq_src"):
            assert full.loc[r.qid, col] == alone.loc[r.qid, col] == alt.loc[r.qid, col], (r.qid, col)


def test_combine_rule():
    meta = pd.DataFrame([
        {"qid": 1, "category": "2S", "question": "What is the velocity of the ball at 1.0s in m/s?", "prior": "speed of the ball = 2 m/s", "target_unit": "m/s"},
        {"qid": 2, "category": "2S", "question": "What is the velocity of the ball at 1.0s in m/s?", "prior": "diameter of the ball = 0.2 m", "target_unit": "m/s"},
        {"qid": 3, "category": "2S", "question": "What is the velocity of the ball at 1.0s in m/s?", "prior": "speed of the ball = 2 m/s", "target_unit": "m/s"},
        {"qid": 4, "category": "2S", "question": "What is the diameter of the ball in m?", "prior": "speed of the ball = 2 m/s", "target_unit": "m"},
    ])
    caw = pd.DataFrame({"qid": [1, 2, 3, 4], "prediction": [5.0, 0.0, 6.0, 0.0],
                        "used_fallback": [False, True, False, True]}).set_index("qid")
    geo = pd.DataFrame({"qid": [1, 2, 3], "source": ["geometry", "not_geometry:box_rejected", "not_geometry:fell_back"],
                        "prediction": [9.0, 8.0, 7.0]}).set_index("qid")
    ref = {"L": 0.94, "V": 1.25, "A": 4.0}
    out = combine(meta, caw, geo, ref)
    assert out.caw_geospeed_iq_final.tolist() == [9.0, 1.25, 6.0, 0.94]
    assert out.caw_iq_final.tolist() == [5.0, 1.25, 6.0, 0.94]
    assert out.caw_geospeed_iq_src.tolist() == ["geometry_iq:speed", "own_fallback:reference_constant", "caw",
                                                "own_fallback:reference_constant"]
