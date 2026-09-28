"""Tests for methods/geometry: parsing, kinematics and an end-to-end synthetic solve."""

from __future__ import annotations

import types

import numpy as np
import pytest

from methods.geometry import measure as M
from methods.geometry.parse import depth_for, parse_depth, parse_prior, parse_question, si_to_unit, unit_to_si
from methods.geometry.solver import VideoContext, solve

REF = {"L": 1.0, "V": 1.0, "A": 9.8}


# ---------------------------------------------------------------- parsing


@pytest.mark.parametrize(
    "q, kind, extra",
    [
        ("What is the length of the black car in meters?", "size", {"dim": "length", "query": "black car"}),
        ("What is the speed of the ball at time 0.5s in m/s?", "speed", {"t": 0.5, "average": False}),
        ("What is the average speed of the small bird in m/s?", "speed", {"average": True, "query": "small bird"}),
        ("What is the acceleration of the yellow ballon the left at 2.00s in cm/s^2?", "accel",
         {"t": 2.0, "query": "yellow ball", "spatial": "left"}),
        ("What is the displacement of the ping pong ball between 1.0s and 1.4s in cm?", "displacement",
         {"t1": 1.0, "t2": 1.4}),
        ("What is the white ball's average velocity in 1.00s to 2.00s in cm/s?", "speed",
         {"t1": 1.0, "t2": 2.0, "query": "white ball"}),
        ("What is the total distance traveled by the left tennis ball in cm?", "path",
         {"query": "tennis ball", "spatial": "left"}),
        ("What is the orbital diameter of the Europa model?", "orbit", {"query": "europa"}),
        ("What is the eagle’s wingspan (the distance from the tip of one wing to the tip of the other) in meters?",
         "size", {"dim": "width", "query": "eagle"}),
        ("How long is the shark in meters?", "size", {"dim": "length", "query": "shark"}),
        ("What is the length of the white car in meters？", "size", {"query": "white car"}),
    ],
)
def test_parse_question(q, kind, extra):
    s = parse_question(q)
    assert s["ok"], s
    assert s["kind"] == kind
    o = s["objects"][0]
    for k, v in extra.items():
        if k in ("query", "spatial"):
            assert o[k] == v, (k, o)
        elif k in ("t", "t1", "t2"):
            assert s["times"][k] == pytest.approx(v)
        else:
            assert s[k] == v, (k, s)


def test_parse_question_two_objects_and_pair():
    s = parse_question("What is the distance between the soccer ball and the basketball at 2.0s in meters?")
    assert [o["query"] for o in s["objects"]] == ["soccer ball", "basketball"] and s["times"]["t"] == 2.0
    s = parse_question("What is the distance between the two black road signs in meters?")
    assert s["objects"][0]["count"] == 2 and s["objects"][0]["query"] == "black road sign"
    s = parse_question("What is the distance between the ball and the floor at 1.50s in meters?")
    assert not s["ok"] and s["reason"] == "question_reference_unsupported"


def test_parse_prior_variants():
    p = parse_prior("gravity acc = 9.8m/s^2")[0]
    assert p["gravity"] and p["kind"] == "accel" and p["value"] == 9.8
    p = parse_prior("t=1.5, ball acceleration = 3.0 m/s^2")[0]
    assert p["kind"] == "accel" and p["times"] == {"t": 1.5} and p["objects"][0]["query"] == "ball"
    p = parse_prior("acceleration of the typewriter before 0.45s = 9.8m/s^2")[0]
    assert p["times"] == {"t1": 0.0, "t2": 0.45} and p["objects"][0]["query"] == "typewriter"
    p = parse_prior("velocity of the orange ball at 0.5s = 2.21m/s")[0]
    assert p["kind"] == "speed" and not p["average"] and p["times"]["t"] == 0.5
    ps = parse_prior("diameter of the tire of the yellow car = 0.764m\ndiameter of the tire of the blue car = 0.369m")
    assert len(ps) == 2 and ps[1]["value"] == 0.369
    p = parse_prior("lane width = 3.66 m")[0]
    assert p["kind"] == "size" and p["dim"] == "width" and p["objects"][0]["query"] == "lane"
    p = parse_prior("ruler calibre = 1 cm")[0]
    assert p.get("unsupported") == "ruler_tick_prior"


def test_units():
    assert unit_to_si("mm", "L") == (0.001, True)
    assert unit_to_si("cm/s^2", "A") == (0.01, True)
    assert unit_to_si("m/s", "A") == (1.0, False)  # prior written with a wrong unit
    assert si_to_unit(1.0, "cm", "L") == pytest.approx(100.0)
    assert si_to_unit(1.0, None, "L") == 1.0


def test_depth_parsing_and_matching():
    d = parse_depth("t=0s, distance_boat_camera =11.827 m\nt = 1.0s, distance_boat_camera = 13.154 m\n"
                    "distance_pier_camera = 18.920 m\nt=1s, distance_human_camera = 10.97 m")
    assert [e["t"] for e in d] == [0.0, 1.0, None, 1.0]
    assert [e["name"] for e in depth_for("boat", d)] == ["boat", "boat"]
    assert depth_for("walking person", d)[0]["name"] == "human"
    assert depth_for("tree", d) == []


# ---------------------------------------------------------------- kinematics


def test_kinematics_constant_acceleration():
    t = np.arange(0, 2, 1 / 24)
    a, v0 = np.array([0.0, 9.8]), np.array([3.0, -2.0])
    p = v0 * t[:, None] + 0.5 * a * t[:, None] ** 2
    assert np.allclose(M.velocity_at(t, p, 1.0), v0 + a * 1.0, atol=1e-6)
    assert np.allclose(M.accel_at(t, p, 1.0), a, atol=1e-6)
    assert M.window_accel(t, p) == pytest.approx(9.8, rel=1e-6)
    assert M.gravity_accel_px(t, p) == pytest.approx(9.8, rel=1e-6)
    q = np.stack([t * 2.0, np.zeros_like(t)], -1)
    assert M.average_speed(t, q) == pytest.approx(2.0, rel=1e-6)
    assert M.path_length(t, q, 0.5, 1.5) == pytest.approx(2.0, rel=1e-3)


def test_depth_series_linear_and_looming():
    ents = [{"t": 0.0, "d": 10.0}, {"t": 1.0, "d": 12.0}]
    assert np.allclose(M.depth_series(ents, np.array([0.5, 2.0])), [11.0, 14.0])
    # one anchor: an object whose apparent size halves is twice as far away
    t = np.linspace(0, 1, 11)
    d_true = 5.0 + 5.0 * t
    size = 100.0 / d_true
    box = np.stack([np.zeros_like(t), np.zeros_like(t), size, size], -1)
    d = M.depth_series([{"t": 0.0, "d": 5.0}], t, box)
    assert np.allclose(d, d_true, rtol=1e-6)


# ---------------------------------------------------------------- end to end


def _synthetic_ctx(fps=24.0, n=48, speed_px=120.0, diam_px=40.0):
    """One 'ball' moving right at speed_px px/s with a diam_px box, plus a clutter box."""
    t = np.arange(n) / fps
    boxes = np.zeros((n, 1, 8, 4), np.float32)
    scores = np.zeros((n, 1, 8), np.float32)
    x = 100 + speed_px * t
    y = np.full(n, 200.0)
    boxes[:, 0, 0] = np.stack([x - diam_px / 2, y - diam_px / 2, x + diam_px / 2, y + diam_px / 2], -1)
    scores[:, 0, 0] = 0.6
    boxes[:, 0, 1] = [500, 50, 540, 90]
    scores[:, 0, 1] = 0.12
    meta = {"ok": True, "queries": ["ball"], "width": 640, "height": 360}
    dets = {"frame_idx": np.arange(n), "boxes": boxes, "scores": scores}
    return VideoContext(meta, dets, fps)


def _row(category="2S", unit="m/s", qid=1):
    return types.SimpleNamespace(qid=qid, category=category, target_unit=unit)


def test_solve_2d_speed_from_size_prior():
    ctx = _synthetic_ctx()
    q = parse_question("What is the speed of the ball at 1s in m/s?")
    p = parse_prior("diameter of the ball = 0.2 m")
    rec = solve(_row(), q, p, [], ctx, [], REF)
    # 120 px/s * (0.2 m / 40 px) = 0.6 m/s
    assert not rec["used_fallback"] and rec["method"] == "2d_scale"
    assert rec["prediction"] == pytest.approx(0.6, rel=1e-3)


def test_solve_2d_size_from_speed_prior_in_cm():
    ctx = _synthetic_ctx()
    q = parse_question("What is the diameter of the ball in cm?")
    p = parse_prior("speed of the ball = 3 m/s")
    rec = solve(_row(category="2D", unit="cm"), q, p, [], ctx, [], REF)
    # scale = 3 / 120 m/px -> 40 px = 1 m = 100 cm
    assert rec["prediction"] == pytest.approx(100.0, rel=1e-3)


def test_solve_3d_calibrated_focal():
    ctx = _synthetic_ctx()
    depth = parse_depth("distance_ball_camera = 4.0 m")
    q = parse_question("What is the average speed of the ball in m/s?")
    p = parse_prior("diameter of the ball = 0.2 m")
    rec = solve(_row(category="3S"), q, p, depth, ctx, [], REF)
    assert rec["method"] == "3d_calibrated_focal"
    assert 0.5 < rec["prediction"] < 0.7  # ~0.6 m/s, small off-axis effects


def test_solve_fallbacks():
    ctx = _synthetic_ctx()
    p = parse_prior("diameter of the ball = 0.2 m")
    rec = solve(_row(unit="meters"), parse_question("What is the height of the giraffe in meters?"), p, [], ctx, [], REF)
    assert rec["used_fallback"] and rec["reason"] == "target_not_detected"
    assert rec["prediction"] == pytest.approx(0.2)  # same-dimension prior
    rec = solve(_row(unit="m/s"), parse_question("What is the speed of the giraffe in m/s?"), p, [], ctx, [], REF)
    assert rec["prediction"] == pytest.approx(REF["V"])  # reference constant
