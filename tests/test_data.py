from __future__ import annotations

import os

import numpy as np

from quantiphy.data import load_template, load_test, load_validation, parse_prior, parse_target_unit


def test_validation_shape_and_categories():
    v = load_validation()
    assert len(v) == 159
    assert v["qid"].is_unique
    assert v["category"].value_counts().to_dict() == {"3D": 47, "3S": 43, "2D": 37, "2S": 32}
    assert v["answer"].notna().all()
    assert all(os.path.exists(p) for p in v["video_path"])


def test_test_shape_and_template_alignment():
    t = load_test()
    tmpl = load_template()
    assert len(t) == len(tmpl) == 3289
    assert (t["qid"].to_numpy() == np.arange(1, 3290)).all()
    assert t["category"].value_counts().to_dict() == {"2D": 1160, "3D": 972, "2S": 581, "3S": 576}
    assert t["video_id"].nunique() == 568
    assert all(os.path.exists(p) for p in t["video_path"])
    # every test question states its output unit (after the 2026-09-14 template fix)
    assert t["target_unit"].notna().all()
    assert t["prior_value"].notna().all()


def test_parse_prior():
    assert parse_prior("length of boat = 3.62m") == (3.62, "m")
    assert parse_prior("acceleration of the orange car = -2.86m/s^2") == (-2.86, "m/s^2")
    assert parse_prior("pedestrian walking speed ~1.1 m/s") == (1.1, "m/s")
    assert parse_prior("t=0.6, ball acceleration = 4.6m/s^2") == (4.6, "m/s^2")
    v, u = parse_prior("no number here")
    assert np.isnan(v) and u == ""


def test_parse_target_unit():
    assert parse_target_unit("What is the length of the pier in meters?") == "meters"
    assert parse_target_unit("What is the acceleration of the ball at 2.00s in cm/s^2?") == "cm/s^2"
    assert parse_target_unit("What is the speed of the car at 1.0s, in m/s?") == "m/s"
    assert parse_target_unit("What is the orbital diameter of the Io model?") is None
