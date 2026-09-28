"""Tests for methods/hybrid_v2: arbitration rule, used-track selection, rendering (no GPU)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from methods.hybrid_v2.hybrid import decide
from methods.hybrid_v2.tracks import TrackRef, frames_to_check, target_objects, used_prior_indices
from methods.hybrid_v2.verify import CONFIG, render

RULE = {"p_yes_threshold": 0.5, "keep_geometry_if_vlm_fallback": True}


def _tables(rows):
    """rows: (qid, geo_fallback, geo_pred, vlm_pred, vlm_fallback)."""
    geo = pd.DataFrame([{"qid": q, "category": "2S", "used_fallback": gf, "prediction": gp}
                        for q, gf, gp, _, _ in rows]).set_index("qid")
    vlm = pd.DataFrame([{"qid": q, "prediction": vp, "used_fallback": vf}
                        for q, _, _, vp, vf in rows]).set_index("qid")
    return geo, vlm


def test_decide_rule_branches():
    geo, vlm = _tables([
        (1, True, 1.0, 5.0, False),    # geometry fell back -> VLM (as hybrid_v1)
        (2, False, 2.0, 6.0, False),   # all tracks pass -> geometry
        (3, False, 3.0, 7.0, False),   # target rejected -> VLM
        (4, False, 4.0, 8.0, False),   # prior rejected -> VLM
        (5, False, 5.0, 9.0, True),    # rejected but VLM fell back -> keep geometry
        (6, False, 6.0, 10.0, False),  # no checkable track (e.g. depth_info answer) -> geometry
        (7, False, 7.0, 11.0, False),  # verifier could not read the frames (NaN) -> geometry
    ])
    verdicts = pd.DataFrame([
        {"qid": 2, "role": "target", "mean_p_yes": 0.9}, {"qid": 2, "role": "prior", "mean_p_yes": 0.5},
        {"qid": 3, "role": "target", "mean_p_yes": 0.1}, {"qid": 3, "role": "prior", "mean_p_yes": 0.99},
        {"qid": 4, "role": "target", "mean_p_yes": 0.8}, {"qid": 4, "role": "prior", "mean_p_yes": 0.49},
        {"qid": 5, "role": "target", "mean_p_yes": 0.0},
        {"qid": 7, "role": "target", "mean_p_yes": np.nan},
    ])
    df = decide(geo, vlm, verdicts, RULE)
    assert df.prediction.to_dict() == {1: 5.0, 2: 2.0, 3: 7.0, 4: 8.0, 5: 5.0, 6: 6.0, 7: 7.0}
    assert df.loc[3, "target_rejected"] and not df.loc[3, "prior_rejected"]
    assert df.loc[4, "prior_rejected"] and not df.loc[4, "target_rejected"]
    assert df.loc[5, "source"] == "geometry:box_rejected_but_vlm_fell_back"
    assert df.loc[6, "n_tracks"] == 0 and df.loc[6, "source"] == "geometry"
    assert (df.source == "vlm:box_rejected").sum() == 2


def test_decide_threshold_is_strict_below():
    geo, vlm = _tables([(1, False, 1.0, 2.0, False)])
    at = pd.DataFrame([{"qid": 1, "role": "target", "mean_p_yes": 0.5}])
    assert decide(geo, vlm, at, RULE).loc[1, "prediction"] == 1.0
    below = pd.DataFrame([{"qid": 1, "role": "target", "mean_p_yes": 0.4999}])
    assert decide(geo, vlm, below, RULE).loc[1, "prediction"] == 2.0
    no_keep = dict(RULE, keep_geometry_if_vlm_fallback=False)
    geo, vlm = _tables([(1, False, 1.0, 2.0, True)])
    assert decide(geo, vlm, below, no_keep).loc[1, "prediction"] == 2.0


def test_used_prior_indices_follow_solver_methods():
    pt = [{"scale_m_per_px": 0.1}, {"reason": "not_detected"}, {"scale_m_per_px": 0.2}]
    assert used_prior_indices({"method": "2d_scale", "prior_tracks": pt}) == [0, 2]
    assert used_prior_indices({"method": "2d_scale_in_3d", "prior_tracks": pt}) == [0, 2]
    assert used_prior_indices({"method": "3d_calibrated_focal", "prior_tracks": pt}) == [0]
    assert used_prior_indices({"method": "3d_scene_depth", "focal_source": "prior_depth", "prior_tracks": pt}) == [0]
    assert used_prior_indices({"method": "3d_scene_depth", "focal_source": "default", "prior_tracks": pt}) == []
    assert used_prior_indices({"method": "3d_default_focal", "prior_tracks": pt}) == []
    assert used_prior_indices({"method": "depth_info", "prior_tracks": []}) == []


def test_target_objects():
    car = {"query": "car", "spatial": None, "count": 1}
    sign = {"query": "road sign", "spatial": None, "count": 2}
    assert target_objects({"kind": "size", "objects": [car]}) == [(car, 0)]
    assert target_objects({"kind": "distance", "objects": [sign]}) == [(sign, 0), (sign, 1)]
    assert target_objects({"kind": "distance", "objects": [car, sign]}) == [(car, 0), (sign, 0)]
    assert target_objects({"kind": "camera_dist", "objects": [car]}) == []


def test_frames_to_check():
    def ref(n):
        return TrackRef("v", "ball", None, 0, "target", "", np.arange(n) / 24, np.arange(n), np.zeros((n, 4)), np.ones(n))
    assert frames_to_check(ref(11), [0.2, 0.5, 0.8]) == [2, 5, 8]
    assert frames_to_check(ref(1), [0.2, 0.5, 0.8]) == [0]
    assert frames_to_check(ref(2), [0.2, 0.5, 0.8]) == [0, 1]


def test_render_keeps_box_interior_clean():
    frame = np.full((480, 854, 3), 255, np.uint8)
    box = np.array([400.0, 200.0, 420.0, 220.0])  # small object
    full, crop = render(frame, box, crop_side=448, margin=1.0)
    assert full.size == (854, 480) and max(crop.size) == 448
    f = np.asarray(full)
    assert (f[201:220, 401:420] == 255).all()  # outline drawn outside the box
    assert (f[196, 396:424] == [255, 0, 0]).all()  # top edge: 2 px gap + 2 px line above y1=200


def test_config_has_all_rule_keys():
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert set(cfg["rule"]) == set(RULE)
    assert set(cfg["verifier"]) == {"frame_quantiles", "crop_side", "crop_margin"}
    assert 0 < cfg["rule"]["p_yes_threshold"] < 1
