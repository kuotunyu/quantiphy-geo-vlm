"""Tests for methods/caw (Code-as-World-VL-9B port): prompt vs the authors' code, parsers, token
layout, combine rules, CLI guards, failure caching, code stamps, answer-free input checks. CPU only, no model. Tests that need the caw env (transformers 5.11,
qwen-vl-utils, the checkpoint's tokenizer files in the HF cache) are skipped in the main env; tests
that need the authors' code (external/Code-as-World, git-ignored) are skipped when it is missing."""

from __future__ import annotations

import ast
import importlib.util
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quantiphy.data import ROOT, TEMPLATE_CSV, load_test

from methods.caw import combine as CB
from methods.caw import recipe as R
from methods.caw import run as C

AUTHORS = ROOT / "external" / "Code-as-World" / "code_as_world"
needs_authors = pytest.mark.skipif(not (AUTHORS / "evaluation.py").exists(), reason="external/Code-as-World not present")
needs_caw_env = pytest.mark.skipif(importlib.util.find_spec("qwen_vl_utils") is None, reason="caw env only (envs/caw)")

# Test qid 1 (simulation_0007) as the authors' code renders it (chat template, thinking off), before the video expansion.
# Rendered with the authors' chat template (MirroS-Lab/Code-as-World @1353bf07, Apache License 2.0); see THIRD_PARTY_NOTICES.md.
GOLDEN_QID1 = (
    "<|im_start|>system\nYou are an expert video analyst specializing in physics measurements. Analyze the video "
    "frames carefully and provide ONLY the numerical answer with units. No explanation or reasoning needed. Format "
    "your response as: [value] [unit]. Example: 2.5 cm. Be as accurate as possible with measurements and calculations. "
    "Please give me an estimated answer even if you are not sure.<|im_end|>\n<|im_start|>user\n"
    "<|vision_start|><|video_pad|><|vision_end|> Given that length of boat = 3.62m. What is the width of the wooden "
    "pier in meters?\n\nPlease answer the question with numbers and units ONLY. No explanation needed.<|im_end|>\n"
    "<|im_start|>assistant\n"
)
SIM7_INDICES = [0, 5, 10, 15, 19, 24, 29, 34, 39, 44, 49, 54, 58, 63, 68, 73]  # 16 of 74 frames (decord)


@pytest.fixture(scope="module")
def authors():
    """evaluation.py executed without its vLLM / qwen-vl-utils / transformers imports (never called here)."""
    src = (AUTHORS / "evaluation.py").read_text(encoding="utf-8")
    heavy = ("vllm", "qwen_vl_utils", "transformers")
    body = [n for n in ast.parse(src).body
            if not (isinstance(n, ast.ImportFrom) and n.module and n.module.split(".")[0] in heavy)
            and not isinstance(n, ast.If)]
    ns = {"__file__": str(AUTHORS / "evaluation.py"), "__name__": "caw_authors"}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(AUTHORS / "evaluation.py"), "exec"), ns)
    from jinja2 import Template

    ns["format_template"] = Template(ns["FORMAT_PROMPT"].read_text(encoding="utf-8").strip())
    records, _, _ = ns["_load_csv"](TEMPLATE_CSV, TEMPLATE_CSV.parent, ".mp4")
    ns["test_records"] = {int(r["original_id"]): r for r in records}
    return ns


@pytest.fixture(scope="module")
def test_rows():
    return load_test().set_index("qid", drop=False)


def _render_chat(messages) -> str:
    """The checkpoint chat template rendered the way transformers does (sandbox, trim/lstrip blocks)."""
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    tmpl = env.from_string((AUTHORS / "templates" / "qwen3_5_no_think.jinja").read_text(encoding="utf-8"))
    return tmpl.render(messages=messages, add_generation_prompt=True, enable_thinking=False)


# --- prompt ---------------------------------------------------------------------------------------


@needs_authors
def test_constants_match_the_authors(authors):
    assert R.SYSTEM_PROMPT == authors["SYSTEM_PROMPT"] and R.DEPTH_PREFIX == authors["DEPTH_PREFIX"]
    assert R.FORMAT_PROMPT == (AUTHORS / "templates" / "quantiphy_video.jinja").read_text(encoding="utf-8").strip()
    assert R.NUMBER_PATTERN.pattern == authors["NUMBER_PATTERN"].pattern
    assert R.NFRAMES_INTERVAL_PATTERN.pattern == authors["NFRAMES_INTERVAL_PATTERN"].pattern
    assert R.SAMPLING_CONFIG == authors["SAMPLING_CONFIG"] and R.SEED == authors["SEED"]
    for k in ("MAX_PROMPT_LENGTH", "MAX_RESPONSE_LENGTH", "MAX_MODEL_LEN", "VIDEO_NFRAMES", "VIDEO_TIMESTAMP_FPS",
              "VIDEO_FPS", "MIN_PIXELS", "MAX_PIXELS"):
        assert getattr(R, k) == authors[k], k
    import hashlib

    assert hashlib.sha256((AUTHORS / "templates" / "qwen3_5_no_think.jinja").read_bytes()).hexdigest() == R.CHAT_TEMPLATE_SHA256


@needs_authors
@pytest.mark.parametrize("qid", [1, 1037, 1536])  # 2D; 3D with multi-line depth info; multi-line prior
def test_message_content_equals_the_authors_on_sample_rows(authors, test_rows, qid):
    mine = R.record_from_row(test_rows.loc[qid])
    theirs = authors["test_records"][qid]
    assert R.message_content(mine) == authors["_message_content"](theirs, authors["format_template"])
    assert R.build_messages(mine)[0] == {"role": "system", "content": authors["SYSTEM_PROMPT"]}
    for k in ("question", "ground_truth_prior", "depth_info", "fps", "category"):
        assert mine[k] == theirs[k], k


@needs_authors
def test_message_content_equals_the_authors_on_every_test_row(authors, test_rows):
    fmt = authors["format_template"]
    diff = [q for q, rec in authors["test_records"].items()
            if R.message_content(R.record_from_row(test_rows.loc[q])) != authors["_message_content"](rec, fmt)]
    assert diff == []


def test_content_rules_on_synthetic_records():
    rec = {"ground_truth_prior": "mass = 2 kg", "question": "given that mass = 2 kg, what is the length in m？",
           "depth_info": "t=0s, d = 3 m"}
    assert R.content_from_record(rec) == (
        "Given that mass = 2 kg. " + R.DEPTH_PREFIX + " t=0s, d = 3 m. What is the length in m?")
    assert R.content_from_record({"question": "How long?", "depth_info": float("nan")}) == "How long?"
    c = R.message_content({"ground_truth_prior": "a = 1 m", "question": "Q?"})
    assert c[0] == {"type": "video"} and c[1]["text"].startswith(" Given that a = 1 m. Q?\n\nPlease answer")


@needs_authors
def test_chat_render_of_qid1_is_the_golden_string(test_rows):
    assert _render_chat(R.build_messages(R.record_from_row(test_rows.loc[1]))) == GOLDEN_QID1


@needs_caw_env
def test_processor_prompt_and_vllm_token_layout(test_rows):
    """Real tokenizer/processor from the HF cache: golden text, 129 tokens, 2238 after the vLLM expansion."""
    try:
        prep = C.Prep()
    except OSError as e:  # checkpoint files not in the local cache
        pytest.skip(f"checkpoint not cached: {e}")
    rendered, ids, truncated = prep.prompt(R.record_from_row(test_rows.loc[1]))
    assert rendered == GOLDEN_QID1 and len(ids) == 129 and not truncated
    assert prep.encode("<0.1 seconds>") == [27, 15, 13, 16, 6283, 29]
    ts = R.calculate_timestamps(SIM7_INDICES, 24.0)
    full = R.expand_video_tokens(ids, (8, 32, 32), ts, prep.encode)
    R.check_layout(full, (8, 32, 32))
    assert len(full) == 2238
    text = prep.tokenizer.decode(full).replace("<|video_pad|>", "")
    assert "user\n<0.1 seconds><|vision_start|><|vision_end|><0.5 seconds>" in text
    assert "<2.9 seconds><|vision_start|><|vision_end|> Given that length of boat" in text
    assert text.endswith("<|im_start|>assistant\n")


# --- video metadata / tokens ------------------------------------------------------------------------


def test_timestamps_use_temporal_groups_of_two():
    ts = R.calculate_timestamps(SIM7_INDICES, 24.0)
    assert [R.timestamp_text(t) for t in ts] == [f"<{x} seconds>" for x in ("0.1", "0.5", "0.9", "1.3", "1.7", "2.1", "2.5", "2.9")]
    assert R.calculate_timestamps([0, 1, 2], 10.0) == pytest.approx([0.05, 0.2])  # odd count: last index repeated


def test_expand_video_tokens_layout():
    enc = {"<0.0 seconds>": [900], "<0.5 seconds>": [901, 902]}
    prompt = [1, 2, R.VISION_START_ID, R.VIDEO_TOKEN_ID, R.VISION_END_ID, 3]
    out = R.expand_video_tokens(prompt, (2, 4, 6), [0.0, 0.5], lambda s: enc[s])
    per = [R.VISION_START_ID] + [R.VIDEO_TOKEN_ID] * 6 + [R.VISION_END_ID]
    assert out == [1, 2, 900] + per + [901, 902] + per + [3]
    R.check_layout(out, (2, 4, 6))
    assert R.mm_token_type_ids([5, R.VIDEO_TOKEN_ID, R.IMAGE_TOKEN_ID]) == [0, 2, 1]
    with pytest.raises(ValueError):
        R.expand_video_tokens([1, 2, 3], (2, 4, 6), [0.0, 0.5], lambda s: enc[s])
    with pytest.raises(ValueError):
        R.check_layout(out + [R.VIDEO_TOKEN_ID], (2, 4, 6))
    R.check_layout([7] * (R.MAX_MODEL_LEN - 1 - len(out)) + out, (2, 4, 6))  # vLLM accepts it, 1 answer token left
    with pytest.raises(ValueError):  # vLLM max_model_len 4608: no room for an answer
        R.check_layout([7] * (R.MAX_MODEL_LEN - len(out)) + out, (2, 4, 6))


def test_max_new_tokens_follows_vllm_max_model_len():
    assert R.max_new_tokens(2238) == 512 and R.max_new_tokens(4096) == 512
    assert R.max_new_tokens(4200) == 408 and R.max_new_tokens(R.MAX_MODEL_LEN - 1) == 1


def test_retry_nframes_follows_the_authors():
    assert R.retry_nframes("nframes should in interval [2, 12], but got 16.") == 12
    assert R.retry_nframes("nframes should in interval [2, 15], but got 16.") == 15  # then qwen-vl-utils asks for 16 again
    assert R.retry_nframes("nframes should in interval [2, 20], but got 16.") is None
    assert R.retry_nframes("some other error") is None


def test_video_input_uses_the_table_fps():
    import torch

    v = {"video": torch.zeros(4, 3, 28, 28), "sample_fps": 2.0,
         "metadata": {"fps": 20.6, "frames_indices": torch.tensor([0, 3, 6, 9]), "total_num_frames": 10,
                      "video_backend": "torchvision", "junk": 1}}
    video, meta, kw = C.video_input(v, 24)
    assert kw == {"fps": 24.0, "do_sample_frames": False}
    assert meta == {"fps": 24.0, "frames_indices": [0, 3, 6, 9], "total_num_frames": 10, "video_backend": "torchvision"}
    _, meta2, _ = C.video_input({**v, "metadata": {"frames_indices": [0, 1]}}, "nan")  # bad fps -> 24; wrong length -> range
    assert meta2["fps"] == R.VIDEO_TIMESTAMP_FPS and meta2["frames_indices"] == [0, 1, 2, 3]


# --- parsers ------------------------------------------------------------------------------------------

REPLIES = [  # reply, question unit, authors' parser, repo parse_answer_sci
    ("150 cm", "meters", 150.0, 1.5), ("0.099 m", "cm", 0.099, 9.9), ("2.5cm", "meters", 2.5, 0.025),
    ("36 km/h", "m/s", 36.0, 10.0), ("1,086.5 m", "meters", 1.0, 1086.5), ("2.27×10³ cm/s", "cm/s", 2.27, 2270.0),
    ("A4 paper is 0.297 m", "meters", 4.0, 0.297), ("<answer>3.2 m/s</answer>", "m/s", 3.2, 3.2),
    ("-9.8 m/s^2", "m/s^2", -9.8, -9.8), ("5e-2 m", "meters", 0.05, 0.05), ("approximately 12.5 meters", "meters", 12.5, 12.5),
]


@pytest.mark.parametrize("reply,unit,authors_value,sci_value", REPLIES)
def test_both_parsers(reply, unit, authors_value, sci_value):
    f = C.parse_fields(reply, unit)
    assert f["parsed_value_authors"] == pytest.approx(authors_value)
    assert f["parse_sci"]["value"] == pytest.approx(sci_value)
    assert f["prediction"] == pytest.approx(abs(sci_value)) and f["used_fallback"] is False


@needs_authors
def test_authors_parser_is_theirs(authors):
    for reply, *_ in REPLIES + [("no number", None, None, None), ("", None, None, None), (None, None, None, None)]:
        assert R.parse_prediction(reply) == authors["_parse_prediction"](reply)
    assert R.format_number(0.1 + 0.2) == authors["_format_number"](0.1 + 0.2) == "0.3"
    assert R.format_number(None) == ""


def test_unusable_replies_fall_back():
    for reply in (None, "no idea", "0 m", "0.0"):
        f = C.parse_fields(reply, "meters")
        assert f["used_fallback"] is True and f["prediction"] is None
    assert C.parse_fields(None, "meters")["parsed_value_authors"] is None


# --- combine ------------------------------------------------------------------------------------------


def _frames():
    meta = pd.DataFrame({
        "qid": [1, 2, 3, 4, 5],
        "category": ["2S", "2S", "3D", "2D", "2D"],
        "question": ["What is the length of the boat in meters?", "What is the speed of the car in m/s?",
                     "What is the velocity of the ball at t=1s in m/s?", "What is the speed of the cart in m/s?",
                     "What is the width of the box in cm?"],
    })
    caw = pd.DataFrame({"used_fallback": [False, False, True, False, True], "prediction": [2.0, 3.0, np.nan, 4.0, np.nan],
                        "parsed_value_authors": [200.0, -3.0, 0.0, 4.0, np.nan],
                        "video_id": ["a", "a", "b", "c", "d"], "error": [np.nan] * 5},
                       index=pd.Index([1, 2, 3, 4, 5], name="qid"))
    v4 = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0], index=[1, 2, 3, 4, 5])
    v2 = pd.DataFrame({"source": ["geometry", "geometry", "geometry", "geometry:box_rejected_but_vlm_fell_back", "vlm:box_rejected"],
                       "prediction": [10.0, 20.0, 30.0, 40.0, 5.0]}, index=pd.Index([1, 2, 3, 4, 5], name="qid"))
    return meta, caw, v4, v2


def test_combine_rules():
    df = CB.combine(*_frames())
    assert df.kind.tolist() == ["size", "speed", "speed", "speed", "size"]
    assert df.caw_final.tolist() == [2.0, 3.0, 30.0, 4.0, 50.0]  # CaW when usable, else hybrid_v4
    assert df.caw_src.tolist() == ["caw", "caw", "hybrid_v4:caw_unusable", "caw", "hybrid_v4:caw_unusable"]
    # speed + hybrid_v2 source exactly "geometry" -> geometry answer; the "geometry:*" source is not measured-with-all-boxes
    assert df.caw_geospeed_final.tolist() == [2.0, 20.0, 30.0, 4.0, 50.0]
    assert df.caw_geospeed_src.tolist() == ["caw", "geometry:speed", "geometry:speed", "caw", "hybrid_v4:caw_unusable"]
    # the authors' parser as-is: |first number|, blank (NaN) when there is none, no fallback
    assert df.caw_authors_final.tolist()[:4] == [200.0, 3.0, 0.0, 4.0] and math.isnan(df.caw_authors_final.iloc[4])
    assert df.caw_authors_src.tolist() == ["caw_authors"] * 4 + ["blank:no_number"]
    s = CB.summary(df)
    assert s["caw"]["changed_vs_hybrid_v4"] == 3 and s["caw_geospeed"]["changed_vs_hybrid_v4"] == 2
    assert s["caw_authors"] == {"blank": 1, "blank_by_category": {"2D": 1}, "zero": 1, "negative_before_abs": 1}


def test_combine_refuses_errors_unless_the_videos_are_named():
    _, caw, _, _ = _frames()
    assert CB.check_errors(caw, None) == []
    caw["error"] = caw["error"].astype(object)
    caw.loc[3, "error"] = "video decode failed: too few frames"
    with pytest.raises(SystemExit):
        CB.check_errors(caw, None)
    with pytest.raises(SystemExit):  # every failed video must be named, and nothing else
        CB.check_errors(caw, ["b", "c"])
    assert CB.check_errors(caw, ["b"]) == ["b"]


def test_combine_refuses_missing_or_inconsistent_inputs():
    meta, caw, v4, v2 = _frames()
    with pytest.raises(SystemExit):
        CB.combine(meta, caw.drop(index=5), v4, v2)
    bad = v2.copy()
    bad.loc[2, "prediction"] = 21.0  # hybrid_v2 != hybrid_v4 on a geometry speed question
    with pytest.raises(SystemExit):
        CB.combine(meta, caw, v4, bad)
    zero = caw.copy()
    zero.loc[1, "prediction"] = 0.0
    with pytest.raises(SystemExit):
        CB.combine(meta, zero, v4, v2)


# --- CLI guards / answers ------------------------------------------------------------------------------


@pytest.mark.parametrize("argv", [["--split", "val", "--limit-videos", "1"], ["--split", "val"],
                                  ["--split", "test", "--recheck", "1"]])
def test_cli_guards(argv):
    with pytest.raises(SystemExit) as e:
        C.main(argv)
    assert e.value.code == 2


def test_prereg_must_be_committed(tmp_path):
    with pytest.raises(SystemExit):
        C.prereg_sha256(str(tmp_path / "nope.md"))
    with pytest.raises(SystemExit):  # untracked file inside the repo
        p = ROOT / "runs" / "caw_prereg_probe_test.md"
        p.parent.mkdir(exist_ok=True)
        p.write_text("x", encoding="utf-8")
        try:
            C.prereg_sha256(str(p))
        finally:
            p.unlink()
    assert len(C.prereg_sha256("pyproject.toml")) == 64


def test_validation_questions_have_no_answers():
    val = C.questions("val")
    assert "answer" not in val.columns and len(val) == 159
    assert all("answer" not in k and "posterior" not in k for k in R.record_from_row(next(val.itertuples())))


def test_collect_and_summarize_on_synthetic_records(tmp_path):
    import json

    base = {"category": "2S", "video_id": "v1", "video_backend": "decord", "n_frames": 16, "total_num_frames": 74,
            "table_fps": 24.0, "decoded_avg_fps": 24.0, "video_grid_thw": [8, 32, 32], "short_video_deviation": None,
            "n_prompt_tokens": 2238, "finish_reason": "stop", "num_generated_tokens": 5, "has_think_end": False,
            "latency_s": 0.8, "video_prep_s": 0.5, "peak_vram_reserved_gib": 18.9, "error": None}
    recs = {1: base | {"reply": "150 cm"} | C.parse_fields("150 cm", "meters"),
            2: base | {"reply": "no idea"} | C.parse_fields("no idea", "meters"),
            3: {"category": "3D", "video_id": "v2", "reply": None, "error": "video decode failed: X"} | C.parse_fields(None, "m/s")}
    for q, r in recs.items():
        (tmp_path / f"{q}.json").write_text(json.dumps({"qid": q} | r), encoding="utf-8")
    items = C.collect(pd.DataFrame({"qid": [1, 2, 3]}), tmp_path)
    assert items.prediction.tolist()[0] == 1.5 and items.used_fallback.tolist() == [False, True, True]
    s = C.summarize(items)
    assert s["n"] == 3 and s["errors"] == 1 and s["video_decode_failures"] == 1 and s["used_fallback"] == 2
    assert s["video_decode_failure_videos"] == ["v2"] and s["authors_negative"] == 0
    assert s["parser_disagreements"]["differ"] == 1 and s["parser_disagreements"]["differ_unit_converted"] == 1
    assert s["port_differences"] == list(C.PORT_DIFFERENCES)
    with pytest.raises(SystemExit):
        C.collect(pd.DataFrame({"qid": [1, 4]}), tmp_path)


# --- failures, caching, code stamp ----------------------------------------------------------------------


class _FakeVP:
    """Stands in for qwen_vl_utils.vision_process: fetch_video pops the next outcome."""

    def __init__(self, outcomes):
        self.outcomes, self.asked = list(outcomes), []

    def fetch_video(self, info, **kw):
        self.asked.append(info["nframes"])
        o = self.outcomes.pop(0)
        if isinstance(o, BaseException):
            raise o
        return (o, {"fps": 24.0}), 2.0


def _interval(n):
    return ValueError(f"nframes should in interval [2, {n}], but got 16.")


@pytest.fixture
def fake_video(tmp_path, monkeypatch):
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"x")

    def make(outcomes, counts=None, why=None):
        vp = _FakeVP(outcomes)
        monkeypatch.setattr(C, "vision_process", lambda: vp)
        monkeypatch.setattr(C, "frame_counts", lambda p: counts or {"decord": 15, "pyav": 15})

        def unopenable(p):
            if why == "never":
                raise AssertionError("resource errors must not probe the file")
            return why

        monkeypatch.setattr(C, "unopenable", unopenable)
        return str(path), vp
    return make


def test_read_video_retries_like_the_authors_then_deviates(fake_video):
    path, vp = fake_video([_interval(15), _interval(15), "frames"])
    v = C.read_video(path)
    assert vp.asked == [16, 15, 14] and v["nframes_requested"] == 14 and "used nframes=14" in v["deviation"]


def test_read_video_caches_only_failures_of_the_file(fake_video):
    path, _ = fake_video([_interval(15)] * 3)  # still too few frames after both retries: the file's own length
    with pytest.raises(C.DecodeFailure):
        C.read_video(path)
    path, _ = fake_video([ValueError("nframes should in interval [2, 1], but got 16.")], counts={"decord": 1, "pyav": 1})
    with pytest.raises(C.DecodeFailure):  # the authors re-raise at once
        C.read_video(path)
    path, _ = fake_video([_interval(15)] * 3, counts={"decord": 40, "pyav": 40})  # a short fallback read, not the file
    with pytest.raises(RuntimeError) as e:
        C.read_video(path)
    assert not isinstance(e.value, C.DecodeFailure)
    path, _ = fake_video([RuntimeError("boom")], why="decord DECORDError: x; PyAV InvalidDataError: y")
    with pytest.raises(C.DecodeFailure):
        C.read_video(path)
    for exc in (MemoryError("low"), OSError("disk"), RuntimeError("cannot allocate memory"),
                C.FallbackRefused("needs 9 GiB")):
        path, _ = fake_video([exc], why="never")
        with pytest.raises(type(exc)) as e:
            C.read_video(path)
        assert not isinstance(e.value, C.DecodeFailure)
    path, _ = fake_video([RuntimeError("processor")], why=None)  # one of the readers opens the file
    with pytest.raises(RuntimeError) as e:
        C.read_video(path)
    assert not isinstance(e.value, C.DecodeFailure)
    with pytest.raises(FileNotFoundError):
        C.read_video(str(Path(path).with_name("missing.mp4")))


STAMP = {"versions": {}, "code": {"head": "h", "code_sha256": "c" * 64, "dirty": False}}


def _raise(exc):
    def f(*a, **k):
        raise exc
    return f


def test_run_video_caches_decode_failures_only(tmp_path, monkeypatch, test_rows):
    g = test_rows[test_rows.video_id == "simulation_0312"].reset_index(drop=True)
    monkeypatch.setattr(C, "prepare_video", _raise(MemoryError("low RAM")))
    n, failed = C.run_video(None, None, g, tmp_path, STAMP)
    assert n == 0 and sorted(failed) == sorted(g.qid) and list(tmp_path.iterdir()) == []
    monkeypatch.setattr(C, "prepare_video", _raise(C.DecodeFailure("too few frames")))
    n, failed = C.run_video(None, None, g, tmp_path, STAMP)
    assert n == len(g) and failed == {}
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(f"{q}.json" for q in g.qid)
    rec = C.load_record(tmp_path / f"{g.qid[0]}.json")
    assert rec["error"] == "video decode failed: too few frames" and rec["code"] == STAMP["code"] and rec["used_fallback"]


def test_atomic_records_and_unreadable_files(tmp_path):
    C.write_json_atomic(tmp_path / "1.json", {"a": 1})
    assert C.load_record(tmp_path / "1.json") == {"a": 1} and [p.name for p in tmp_path.iterdir()] == ["1.json"]
    (tmp_path / "2.json").write_text('{"qid": 2, "rep', encoding="utf-8")  # killed mid-write by the old code
    assert C.load_record(tmp_path / "2.json") is None
    with pytest.raises(SystemExit) as e:
        C.collect(pd.DataFrame({"qid": [2]}), tmp_path)
    assert "2.json" in str(e.value.code)


def test_code_stamp_and_foreign_records(tmp_path):
    state = C.code_state()
    assert len(state["head"]) == 40 and len(state["code_sha256"]) == 64 and isinstance(state["dirty"], bool)
    assert C.code_state()["code_sha256"] == state["code_sha256"]
    clean = {"head": "h", "code_sha256": "c" * 64, "dirty": False, "dirty_files": []}
    pre = {"file": "docs/p.md", "sha256": "p" * 64}
    recs = {1: {"code": clean}, 2: {"code": clean | {"code_sha256": "d" * 64}}, 3: {"code": clean | {"dirty": True}},
            4: {}, 6: {"code": clean, "prereg": pre}}
    for q, r in recs.items():
        C.write_json_atomic(tmp_path / f"{q}.json", r)
    (tmp_path / "5.json").write_text("{", encoding="utf-8")
    assert C.foreign_records(tmp_path, range(1, 8), clean) == [2, 3, 4, 5]
    assert C.foreign_records(tmp_path, [1, 6], clean, pre) == [1]
    C.require_committed_code(tmp_path, [1, 7], clean, "test")
    for qids, state in (([1, 2], clean), ([1], clean | {"dirty": True, "dirty_files": [" M methods/caw/run.py"]})):
        with pytest.raises(SystemExit) as e:
            C.require_committed_code(tmp_path, qids, state, "test")
        assert e.value.code == 2
    assert C.items_code(tmp_path, [1, 2, 4]) == sorted(["c" * 64, "d" * 64, "none"])


def test_main_exit_codes_without_the_model(tmp_path, monkeypatch):
    clean = {"head": "h", "code_sha256": "c" * 64, "dirty": False, "dirty_files": []}
    monkeypatch.setattr(C, "RUNS", tmp_path)
    monkeypatch.setattr(C, "code_state", lambda: clean)
    assert C.main(["--split", "test", "--max-videos", "0"]) == 3  # nothing run, questions pending
    assert C.main(["--split", "val", "--prereg", "pyproject.toml", "--max-videos", "0"]) == 3
    C.write_json_atomic(tmp_path / "test" / "items" / "1.json", {"code": clean | {"code_sha256": "d" * 64}})
    with pytest.raises(SystemExit) as e:  # a record made by other code
        C.main(["--split", "test", "--max-videos", "0"])
    assert e.value.code == 2
    monkeypatch.setattr(C, "code_state", lambda: clean | {"dirty": True, "dirty_files": ["?? methods/caw/x.py"]})
    with pytest.raises(SystemExit) as e:  # uncommitted code
        C.main(["--split", "val", "--prereg", "pyproject.toml", "--max-videos", "0"])
    assert e.value.code == 2


def test_one_video_input_and_probe_videos(test_rows):
    C.one_video_input(test_rows)
    C.one_video_input(C.questions("val"))
    bad = pd.DataFrame({"video_id": ["v", "v"], "fps": [24, 30], "video_path": ["a", "a"]})
    with pytest.raises(SystemExit):
        C.one_video_input(bad)
    assert C.pick_videos(test_rows, 2) == ["captured_0001", "simulation_0312"]  # the same whatever is pending


def test_track_b_code_imports_only_local_and_open_source_modules():
    """methods/caw and methods/common import only the standard library, this repository's own code
    and the local inference stack: no network API client."""
    import sys

    third_party = {"numpy", "pandas", "torch", "transformers", "qwen_vl_utils", "decord", "av", "jinja2", "psutil"}
    local = ("methods.caw", "methods.common", "methods.geometry", "methods.vlm_baseline", "quantiphy")
    for f in [*(ROOT / "methods" / "caw").glob("*.py"), *(ROOT / "methods" / "common").glob("*.py")]:
        for node in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            for n in names:
                top = n.split(".")[0]
                assert (top in sys.stdlib_module_names or top in third_party
                        or n.startswith(local)), (f.name, n)


# --- answer-free input checks (videoscan) ---------------------------------------------------------------


def test_sampling_plan_and_indices():
    from methods.caw import videoscan as V

    assert V.sampling_plan(74) == {"case": "ok", "nframes": 16}
    assert V.sampling_plan(12) == {"case": "authors_retry", "nframes": 12}
    assert V.sampling_plan(13) == {"case": "authors_retry", "nframes": 12}  # round(6.5) = 6
    assert V.sampling_plan(15) == {"case": "authors_retry_raises_port_deviation", "nframes": 14, "authors_retry_asks": 16}
    assert V.sampling_plan(1)["case"] == "too_short_for_any_retry"
    assert V.sample_indices(74, 16) == SIM7_INDICES


def test_github_csv_diff_never_reads_the_answer_column(tmp_path):
    from methods.caw import videoscan as V

    head = (",video_id,video_source,video_type,fps,inference_type,question,ground_truth_prior,depth_info,"
            "ground_truth_posterior\n")
    (tmp_path / "a.csv").write_text(head + "7,v1,sim,S2MC,24,SS,How long?,a = 1 m,,SENTINEL\n"
                                    "8,v2,sim,S2MC,30,SS,How fast?,b = 2 m,,1\n", encoding="utf-8")
    (tmp_path / "b.csv").write_text(head + "7,v1,sim,S2MC,24.0,SS,How long?,a = 1 m,,SENTINEL\n"
                                    "8,v2,sim,S2MC,30,SS,How far?,b = 2 m,,1\n", encoding="utf-8")
    a, b = V.read_input_columns(tmp_path / "a.csv"), V.read_input_columns(tmp_path / "b.csv")
    assert "ground_truth_posterior" not in a.columns and "SENTINEL" not in a.to_string()
    d = V.diff_input_columns(a, b)
    assert d["matched_by"] == "id" and d["n_compared"] == 2
    assert d["columns"]["fps"]["n_differ"] == 0 and d["columns"]["question"]["n_differ"] == 1
    assert d["columns"]["question"]["examples"] == [{"id": "8", "local": "How fast?", "github": "How far?"}]


def test_usable():
    assert R.usable(1.5) and not R.usable(0.0) and not R.usable(math.nan) and not R.usable(math.inf) and not R.usable(None)
