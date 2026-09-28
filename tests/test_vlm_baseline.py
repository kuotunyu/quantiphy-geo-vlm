"""Tests for methods/vlm_baseline: reply parsing, unit conversion and prompt construction (no GPU)."""

from __future__ import annotations

import math
import types

import pytest

from methods.vlm_baseline.answer import parse_answer, unit_dim
from methods.vlm_baseline.run import build_prompt, context_prefix


@pytest.mark.parametrize(
    "reply, target, value, converted",
    [
        ("2.5 cm", "cm", 2.5, False),
        ("150 cm", "meters", 1.5, True),
        ("3.2 m/s", "cm/s", 320.0, True),
        ("9.8 m/s^2", "cm/s^2", 980.0, True),
        ("9.8 m/s²", "m/s^2", 9.8, False),
        ("0.12 m/s/s", "cm/s^2", 12.0, True),
        ("5 meters per second", "cm/s", 500.0, True),
        ("36 km/h", "m/s", 10.0, True),
        ("1,086.5 meters", "meters", 1086.5, False),
        (".75 m", "cm", 75.0, True),
        ("1.2m", "meters", 1.2, False),
        ("0.5", "m/s", 0.5, False),
        ("4.8 m/s", "meters", 4.8, False),  # dimension mismatch: keep the number
        ("5 mph", "m/s", 5.0, False),  # unknown unit: keep the number
        ("-2.3 m/s2", "m/s^2", -2.3, False),
        ("3 mm", "cm", 0.3, True),
    ],
)
def test_parse_answer(reply, target, value, converted):
    out = parse_answer(reply, target)
    assert out["ok"]
    assert out["value"] == pytest.approx(value)
    assert out["converted"] is converted


@pytest.mark.parametrize("reply", ["I cannot tell", "", None, "unknown m/s"])
def test_parse_failure(reply):
    out = parse_answer(reply, "m")
    assert not out["ok"] and math.isnan(out["value"])


def test_unit_dim():
    assert unit_dim("meters") == "L" and unit_dim("cm") == "L"
    assert unit_dim("m/s") == "V" and unit_dim("km/h") == "V"
    assert unit_dim("cm/s^2") == "A" and unit_dim("m/s/s") == "A"
    assert unit_dim(None) is None and unit_dim("mph") is None


def test_prompt_matches_starter_kit_wording():
    c = context_prefix("length of boat = 3.62m", "t=0s, distance_boat_camera = 11.8 m")
    assert c.startswith("Given that length of boat = 3.62m. Additionally, you have the following information")
    assert c.endswith("11.8 m. ")
    assert context_prefix(float("nan"), float("nan")) == ""
    row = types.SimpleNamespace(fps=24, prior="speed of the boat = 1.5 m/s", depth_info=float("nan"),
                                question="What is the length of the boat in meters?")
    p = build_prompt(row, n_total=60, n_shown=32)
    assert "24 frames per second" in p and "2.50 s (60 frames)" in p and "32 frames" in p
    assert p.rstrip().endswith("No explanation needed.")
