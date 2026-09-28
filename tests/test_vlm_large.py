"""Tests for methods/vlm_large: hybrid_v4 combination rule and model pinning (no GPU)."""

from __future__ import annotations

import pandas as pd
import pytest

from methods.vlm_baseline import run as base
from methods.vlm_large import run as big_run
from methods.vlm_large.hybrid import combine


def _v2():
    return pd.DataFrame([
        {"qid": 1, "category": "2S", "source": "geometry", "prediction": 1.0},
        {"qid": 2, "category": "2D", "source": "vlm:geometry_fell_back", "prediction": 2.0},
        {"qid": 3, "category": "3S", "source": "vlm:box_rejected", "prediction": 3.0},
        {"qid": 4, "category": "3D", "source": "geometry:box_rejected_but_vlm_fell_back", "prediction": 4.0},
    ]).set_index("qid")


def test_combine_replaces_only_vlm_answers():
    big = pd.DataFrame([
        {"qid": 1, "prediction": 10.0, "used_fallback": False},
        {"qid": 2, "prediction": 20.0, "used_fallback": False},
        {"qid": 3, "prediction": 30.0, "used_fallback": True},  # 32B gave no number -> keep v2
        {"qid": 4, "prediction": 40.0, "used_fallback": False},
    ]).set_index("qid")
    df = combine(_v2(), big)
    assert df.prediction.to_dict() == {1: 1.0, 2: 20.0, 3: 3.0, 4: 4.0}
    assert df.source.to_dict() == {1: "v2", 2: "vlm32", 3: "v2:vlm32_fell_back", 4: "v2"}


def test_combine_needs_32b_answer_for_every_vlm_question():
    big = pd.DataFrame([{"qid": 2, "prediction": 20.0, "used_fallback": False}]).set_index("qid")
    with pytest.raises(SystemExit):
        combine(_v2(), big)  # qid 3 is VLM-answered in v2 but has no 32B answer


def test_same_recipe_as_baseline_only_model_differs():
    assert big_run.MODEL_ID == "Qwen/Qwen3-VL-32B-Instruct" and len(big_run.MODEL_REVISION) == 40
    assert big_run.QwenVL32.ask is base.QwenVL.ask  # prompt / frames / decoding inherited, not copied
    assert big_run.base.build_prompt is base.build_prompt and big_run.base.read_video is base.read_video
    assert big_run.QUANT["bnb_4bit_quant_type"] == "nf4" and big_run.QUANT["llm_int8_skip_modules"] == []  # everything 4-bit (fits 24 GB VRAM)
