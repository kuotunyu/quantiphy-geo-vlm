"""Code-as-World-VL-9B answering every question with the authors' QuantiPhy recipe (open weights -> Track B).

Parts of this file follow MirroS-Lab/Code-as-World (https://github.com/MirroS-Lab/Code-as-World,
commit 1353bf07, code_as_world/evaluation.py), Apache License 2.0, changed to run with Hugging Face
transformers generate() instead of vLLM; every change is listed in PORT_DIFFERENCES below. See
THIRD_PARTY_NOTICES.md.

Runs in the separate env envs/caw (transformers 5.11; the main env cannot load `qwen3_5`),
from the repo root:

    uv run --project envs/caw python -m methods.caw.run --split test --limit-videos 2 --recheck 1  # probe
    uv run --project envs/caw python -m methods.caw.run --split test                  # all 3,289 questions
    uv run --project envs/caw python -m methods.caw.run --split test --max-videos 100 # next 100 videos, then stop
    uv run --project envs/caw python -m methods.caw.run --split val --prereg docs/<plan>.md  # 159 questions

Exit codes: 0 every question of the split has a record and the summary is written (probe: the
probe videos are done); 1 some questions failed in this pass (not cached, ids printed, retried by
the next run); 2 refused before any work (arguments, uncommitted code, records made by other code,
pre-registration not committed); 3 stopped after --max-videos with videos still pending.
Loading maps the 18.8 GB checkpoint (peak working set up to ~15.6 GiB in the probes): stop other
memory-heavy jobs (for example a virtual machine) before the full run.

Model: MirroS-Lab/Code-as-World-VL-9B at a pinned revision (Apache-2.0, base Qwen3.5-9B), bf16.
Recipe: the authors' external/Code-as-World/code_as_world/evaluation.py (methods/caw/recipe.py
mirrors its text side), with Hugging Face transformers generate() in place of vLLM 0.19.1,
which does not run on Windows:
  - prompt: their system prompt, "Given that <prior>. [depth info] <question>" with their
    templates/quantiphy_video.jinja, and their chat template (== the checkpoint's; thinking off);
  - video: qwen-vl-utils 0.0.14 fetch_video with their settings (16 frames, max_pixels 262144,
    decord reader), their retry for clips shorter than 16 frames, timestamps from the table fps;
  - pixels: the checkpoint's Qwen3VLProcessor called the way vLLM calls it (do_sample_frames=False);
  - tokens: the vLLM 0.19.1 layout (recipe.expand_video_tokens), M-RoPE from mm_token_type_ids;
  - decoding: argmax (their temperature 0.01 / top_p 0.001 keeps only the top token), max 512
    new tokens (fewer if the prompt is near vLLM's max_model_len), stop at <|im_end|>, logit bias
    -100 on <|image_pad|> / <|video_pad|>.
Every difference from the authors' run (another engine's bf16 numerics, exact probability ties,
a fix for a crash in their short-clip retry, ...) is listed in PORT_DIFFERENCES below, which is
also written into every summary JSON.

Per-question records (prompt, frames, reply, both parsers, latency, peak VRAM, versions, and the
code that made them) are cached in runs/caw/<split>/items/<qid>.json (written atomically), so an
interrupted run resumes. Only failures that are properties of the video file are cached (as
"video decode failed"; see read_video); every other error leaves the question pending. The full
run, the validation run and the final collect refuse uncommitted code in RUN_CODE_PATHS and
records made by other code. Validation answers are dropped before inference. This program writes
no prediction or submission file: methods/caw/combine.py does that.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from importlib import metadata as importlib_metadata
from pathlib import Path

import numpy as np
import pandas as pd

from quantiphy.data import ROOT, load_test, load_validation

from methods.caw import recipe as R
from methods.common.parse_sci import parse_answer_sci

VERSION = "caw"
RUNS = ROOT / "runs" / VERSION
# 9,409,813,744 bf16 parameters = 17.5 GiB of weights on a 24 GiB card (the desktop takes ~1.6 GiB).
# Hard cap for this process, as in methods/vlm_large: on Windows (WDDM) the driver would otherwise
# spill into system RAM instead of failing.
VRAM_CAP_GIB = 22.0
MIN_FREE_GPU_GIB = 20.5
# Below this much available RAM, loading the memory-mapped checkpoint pages heavily (a warning only).
LOW_RAM_GIB = 12.0
# qwen-vl-utils falls back to torchvision.io.read_video, which decodes the whole clip at full size,
# whenever decord raises (for the test set: only for clips shorter than 16 frames). Refuse it beyond this.
TV_FALLBACK_MAX_GIB = 3.0
VIDEO_PLACEHOLDER = "<|vision_start|><|video_pad|><|vision_end|>"
_VIDEO_METADATA_KEYS = {"total_num_frames", "fps", "width", "height", "duration", "video_backend", "frames_indices"}
_EVENTS: list[str] = []  # what the video readers did for the current video
_VP = None
# The committed files that decide the per-question records (prompt, frames, decoding, parsers).
# Every record stores a hash of them at HEAD; see code_state(). combine.py is not listed: it only
# reads items.csv, so changing it does not make the cached answers stale.
RUN_CODE_PATHS = (
    "methods/__init__.py", "methods/caw/__init__.py", "methods/caw/recipe.py", "methods/caw/run.py",
    "methods/common", "methods/vlm_baseline/__init__.py", "methods/vlm_baseline/answer.py",
    "methods/geometry/__init__.py", "methods/geometry/parse.py",
    "src/quantiphy/__init__.py", "src/quantiphy/data.py", "src/quantiphy/mra.py",
    "envs/caw/pyproject.toml", "envs/caw/uv.lock",
)

PORT_DIFFERENCES = (
    "engine: Hugging Face transformers 5.11 generate() on Windows (SDPA attention; torch fallback for the Gated "
    "DeltaNet layers and causal conv1d, since flash-linear-attention / causal-conv1d are not installed; no CUDA "
    "graphs; batch size 1) instead of vLLM 0.19.1 on Linux. Different bf16 numerics can flip argmax tokens.",
    "decoding: argmax (do_sample=False) instead of vLLM sampling with temperature 0.01, top_p 0.001, seed 1, "
    "which keeps only the top token; the two differ only on exact probability ties.",
    "logit bias -100 on <|image_pad|> and <|video_pad|> re-implemented as a transformers LogitsProcessor.",
    "token layout: vLLM's prompt replacement (per temporal group: timestamp text, <|vision_start|>, video pads, "
    "<|vision_end|>) rebuilt by hand; M-RoPE from mm_token_type_ids. The transformers processor text path "
    "(which keeps an extra outer <|vision_start|>/<|vision_end|>) is not used.",
    "short clips: after the authors' retry (nframes = clip length), qwen-vl-utils rounds to an even count; for "
    "4k+3 frames (test: internet_0038, 15 frames) that raises again and their run stops. Here a second retry "
    "uses the largest even count (14); flagged per question in short_video_deviation.",
    "video: decoded one video at a time and reused for all its questions (the authors preprocess every record "
    "up front); frames are deterministic, so the model inputs are the same.",
    "video: the qwen-vl-utils fallback to torchvision.io.read_video (whole clip in RAM) is refused above "
    f"{TV_FALLBACK_MAX_GIB} GiB; on the test set it is only reached for the six clips shorter than 16 frames.",
    "errors: the authors' run stops at the first video it cannot read. Here a video is given up on (cached as "
    "'video decode failed'; combine.py refuses to run until the video is named with --accept-failed-videos) only "
    "for failures that are properties of the file: too few frames for their sampling even after the retries, or "
    "neither decord nor PyAV can open it. Any other error (memory, I/O, the RAM guard above, processor, model) is "
    "not cached: the question stays pending and the run exits non-zero.",
    "prompt length: as in vLLM (max_model_len 4096 + 512), a prompt of 4608 tokens or more is refused and "
    "generation stops at 4608 tokens in total (max_new_tokens = min(512, 4608 - prompt)); no test prompt comes "
    "close (about 1.5-2.3k tokens).",
    "validation: videos resolved with quantiphy.data.resolve_video (one file name has a leading space, which the "
    "authors' resolver would not find); answers dropped before prompts are built and never written next to "
    "predictions.",
    "outputs: per-question JSON + items.csv instead of their CSV; the repo's text parser parse_answer_sci "
    "(methods/common/parse_sci.py) is recorded next to the authors' parser; the hybrid_v4 fallback "
    "(methods/caw/combine.py) is not part of their recipe.",
)


class DecodeFailure(Exception):
    """A failure that is a property of the video file and repeats on every run: cached as "video decode
    failed", so no question of the video gets a CaW answer."""


class FallbackRefused(RuntimeError):
    """The whole-clip torchvision decode would need more than TV_FALLBACK_MAX_GIB of RAM (not cached)."""


def refuse(msg: str):
    """Stop before any work, exit code 2 (like an argparse error)."""
    print(f"[caw] refused: {msg}", file=sys.stderr, flush=True)
    raise SystemExit(2)


def questions(split: str) -> pd.DataFrame:
    if split == "val":
        return load_validation().drop(columns=["answer"])  # inference never sees answers
    return load_test()


def versions() -> dict:
    out = {"python": platform.python_version()}
    for name in ("torch", "torchvision", "transformers", "tokenizers", "qwen-vl-utils", "decord", "accelerate",
                 "numpy", "av", "jinja2"):
        try:
            out[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            out[name] = None
    try:
        import torch

        out["cuda"] = torch.version.cuda
        out["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:  # noqa: BLE001 - informational only
        pass
    return out


# ---------------------------------------------------------------------------
# which code made a record
# ---------------------------------------------------------------------------


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", check=True).stdout


def code_state() -> dict:
    """HEAD, a sha256 of the committed RUN_CODE_PATHS (git ls-tree: mode, blob id, path) and whether
    any of them differs from HEAD in the working tree (modified, staged or untracked)."""
    tree = _git("ls-tree", "-r", "HEAD", "--", *RUN_CODE_PATHS)
    changed = [ln for ln in _git("status", "--porcelain", "--untracked-files=all", "--", *RUN_CODE_PATHS).splitlines()
               if ln.strip()]
    return {"head": _git("rev-parse", "HEAD").strip(), "code_sha256": hashlib.sha256(tree.encode("utf-8")).hexdigest(),
            "dirty": bool(changed), "dirty_files": changed[:20]}


def load_record(p: Path) -> dict | None:
    """A cached record, or None when the file cannot be read as JSON."""
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def write_json_atomic(path: Path, obj, **kw) -> None:
    """Write to <name>.tmp, then rename: a killed run never leaves a truncated record behind."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, **kw), encoding="utf-8")
    os.replace(tmp, path)


def foreign_records(items_dir: Path, qids, state: dict, prereg: dict | None = None) -> list[int]:
    """Cached records of these questions that the committed code `state` did not make: another
    code_sha256, made from a dirty tree, made before records carried a code stamp, or unreadable.
    With `prereg` (validation), records made under another pre-registration file count too."""
    bad = []
    for q in qids:
        p = items_dir / f"{q}.json"
        if not p.exists():
            continue
        rec = load_record(p) or {}
        code = rec.get("code") or {}
        if code.get("code_sha256") != state["code_sha256"] or code.get("dirty") is not False or (
                prereg is not None and (rec.get("prereg") or {}).get("sha256") != prereg["sha256"]):
            bad.append(int(q))
    return bad


def require_committed_code(items_dir: Path, qids, state: dict, what: str, prereg: dict | None = None) -> None:
    """Refuse (exit 2) when RUN_CODE_PATHS differ from HEAD or cached records were made by other code
    (or, for validation, under another pre-registration)."""
    if state["dirty"]:
        refuse(f"{what}: uncommitted changes in files that decide the answers: {state['dirty_files']}; commit first")
    bad = foreign_records(items_dir, qids, state, prereg)
    if bad:
        refuse(f"{what}: {len(bad)} cached records in {items_dir} were not made by the committed code "
               f"(code_sha256 {state['code_sha256'][:12]}){' under this pre-registration' if prereg else ''}, "
               f"e.g. qids {bad[:10]}; delete them and run again")


# ---------------------------------------------------------------------------
# video (evaluation.py _process_video / _video_input)
# ---------------------------------------------------------------------------


def _decoded_gib(path: str) -> float:
    """RAM torchvision.io.read_video needs for the whole clip (frames list + stacked copy, uint8 RGB)."""
    import av

    with av.open(path) as c:
        s = c.streams.video[0]
        n = s.frames or sum(1 for p in c.demux(s) if p.size > 0)
        return 2 * n * s.height * s.width * 3 / 2**30


def vision_process():
    """qwen_vl_utils.vision_process reading with decord, as in the authors' env (vLLM 0.19.1 brings no
    torchcodec; decord is pinned). Its two readers are wrapped only to record what happened and to
    refuse a whole-clip torchvision decode that would not fit in RAM; frames are unchanged."""
    global _VP
    if _VP is None:
        os.environ["FORCE_QWENVL_VIDEO_READER"] = "decord"  # read once, when the module is imported
        import torch  # noqa: F401 - on Windows, loading decord before torch breaks torch's c10.dll (WinError 1114)
        import qwen_vl_utils.vision_process as vp

        if vp.get_video_reader_backend() != "decord":
            raise RuntimeError("qwen-vl-utils was imported before the decord reader could be forced")
        if vp.MODEL_SEQ_LEN != 128000:
            raise RuntimeError(f"MODEL_SEQ_LEN={vp.MODEL_SEQ_LEN} changes qwen-vl-utils' pixel budget; unset it")
        decord_reader, torchvision_reader = vp.VIDEO_READER_BACKENDS["decord"], vp.VIDEO_READER_BACKENDS["torchvision"]

        def decord_logged(ele):
            try:
                return decord_reader(ele)
            except Exception as e:
                _EVENTS.append(f"decord raised {type(e).__name__}: {e}")
                raise

        def torchvision_guarded(ele):
            gib = _decoded_gib(ele["video"])
            if gib > TV_FALLBACK_MAX_GIB:
                raise FallbackRefused(f"torchvision fallback refused: whole-clip decode needs ~{gib:.1f} GiB RAM")
            _EVENTS.append("fell back to torchvision.io.read_video")
            return torchvision_reader(ele)

        vp.VIDEO_READER_BACKENDS.update(decord=decord_logged, torchvision=torchvision_guarded)
        _VP = vp
    return _VP


def _resource_error(e: BaseException) -> bool:
    """Errors that say something about this machine right now, not about the file."""
    msg = str(e).lower()
    return isinstance(e, (MemoryError, OSError)) or "memory" in msg or "alloc" in msg


def frame_counts(path: str) -> dict:
    """The clip length by decord's index and by PyAV's demuxer (None when that reader fails)."""
    import torch  # noqa: F401 - before decord (Windows)

    out: dict = {"decord": None, "pyav": None}
    try:
        import decord

        out["decord"] = len(decord.VideoReader(path, num_threads=1))
    except Exception as e:  # noqa: BLE001
        if _resource_error(e):
            raise
    try:
        import av

        with av.open(path) as c:
            s = c.streams.video[0]
            out["pyav"] = sum(1 for p in c.demux(s) if p.size > 0)
    except Exception as e:  # noqa: BLE001
        if _resource_error(e):
            raise
    return out


def unopenable(path: str) -> str | None:
    """Why neither decord nor PyAV can open the file and give a frame, or None when one of them can.
    Memory / OS errors propagate: they say nothing about the file."""
    import torch  # noqa: F401 - before decord (Windows)

    why = []
    try:
        import decord

        if len(decord.VideoReader(path, num_threads=1)) > 0:
            return None
        why.append("decord: 0 frames")
    except Exception as e:  # noqa: BLE001
        if _resource_error(e):
            raise
        why.append(f"decord {type(e).__name__}: {str(e)[:200]}")
    try:
        import av

        with av.open(path) as c:
            next(c.decode(video=0))
        return None
    except Exception as e:  # noqa: BLE001
        if _resource_error(e):
            raise
        why.append(f"PyAV {type(e).__name__}: {str(e)[:200]}")
    return "; ".join(why)


def _too_few_frames(path: str, exc: ValueError) -> Exception:
    """The error to raise when the frame-count ValueError survives the retries. Cached (DecodeFailure)
    only when the clip length in the message is the file's own length by decord or PyAV, i.e. not a
    whole-clip fallback decode that came back short."""
    reported = int(R.NFRAMES_INTERVAL_PATTERN.search(str(exc)).group(2))
    counts = frame_counts(path)
    if reported not in counts.values():
        return RuntimeError(f"reader saw {reported} frames but the file has {counts}: {exc}")
    return DecodeFailure(f"too few frames for the authors' 16-frame sampling even after the retries ({counts}): {exc}")


def read_video(path: str) -> dict:
    """evaluation.py _process_video (L318-346): fetch_video with the authors' settings; on the
    too-short-clip ValueError, one retry with nframes = the clip length (their code). qwen-vl-utils
    rounds that to an even number (half to even), so a clip of 4k+3 frames fails again and their
    run stops; here (pre-registered deviation, recorded in 'deviation') a second retry asks for the
    largest even count <= the clip length.

    Raises DecodeFailure (cached) only when the frame count is still too small after the retries or
    neither decord nor PyAV can open the file; everything else propagates and is not cached."""
    vp = vision_process()
    if not os.path.isfile(path):  # evaluation.py L319-320; a setup problem, never cached
        raise FileNotFoundError(f"video not found: {path}")
    info = {"video": path, "min_pixels": R.MIN_PIXELS, "max_pixels": R.MAX_PIXELS,
            "video_fps": R.VIDEO_FPS, "nframes": R.VIDEO_NFRAMES}
    _EVENTS.clear()
    deviation = None

    def fetch():
        return vp.fetch_video(info, return_video_sample_fps=True, return_video_metadata=True)

    def frame_error(exc: Exception) -> bool:
        return isinstance(exc, ValueError) and R.NFRAMES_INTERVAL_PATTERN.search(str(exc)) is not None

    try:
        try:
            out, sample_fps = fetch()
        except ValueError as exc:
            if not frame_error(exc):
                raise
            n = R.retry_nframes(str(exc))
            if n is None:  # the authors re-raise here
                raise _too_few_frames(path, exc) from exc
            _EVENTS.append(f"retry with nframes={n} (authors): {exc}")
            info["nframes"] = n
            try:
                out, sample_fps = fetch()
            except ValueError as exc2:
                if not frame_error(exc2):
                    raise
                n2 = int(R.NFRAMES_INTERVAL_PATTERN.search(str(exc2)).group(2)) // 2 * 2
                deviation = f"authors' retry nframes={n} raised '{exc2}' (their run stops); used nframes={n2}"
                _EVENTS.append(f"retry with nframes={n2} (deviation): {exc2}")
                info["nframes"] = n2
                try:
                    out, sample_fps = fetch()
                except ValueError as exc3:
                    if not frame_error(exc3):
                        raise
                    raise _too_few_frames(path, exc3) from exc3
    except (DecodeFailure, FallbackRefused):
        raise
    except Exception as exc:
        if not _resource_error(exc):
            why = unopenable(path)
            if why is not None:
                raise DecodeFailure(f"neither decord nor PyAV can open the file ({why})") from exc
        raise
    video, meta = out
    return {"video": video, "metadata": meta, "sample_fps": float(sample_fps), "nframes_requested": info["nframes"],
            "events": list(_EVENTS), "deviation": deviation}


def video_input(v: dict, fps_value) -> tuple:
    """evaluation.py _video_input (L349-393) for one video: (frames, VideoMetadata fields, processor kwargs).
    The timestamps use the table fps, not the container's."""
    timestamp_fps = R.positive_float(fps_value) or R.VIDEO_TIMESTAMP_FPS
    video, raw = v["video"], v["metadata"]
    meta = {k: val for k, val in dict(raw).items() if k in _VIDEO_METADATA_KEYS} if raw is not None else {}
    processor_fps = timestamp_fps or float(v["sample_fps"])
    processor_fps = float(processor_fps if processor_fps > 0 else 24.0)
    num_frames = int(video.shape[0])
    fi = meta.get("frames_indices")
    if hasattr(fi, "detach"):
        fi = fi.detach().cpu().tolist()
    elif fi is not None:
        fi = list(fi)
    if not fi or len(fi) != num_frames:
        fi = list(range(num_frames))
    meta["fps"] = processor_fps
    meta["frames_indices"] = [int(i) for i in fi]
    meta["total_num_frames"] = int(meta.get("total_num_frames", num_frames))
    return video, meta, {"fps": timestamp_fps, "do_sample_frames": False}


def one_video_input(df: pd.DataFrame) -> None:
    """A video is prepared once for all its questions, while the authors compute the timestamps per
    row: every question of a video must share its file and table fps (true on test and validation)."""
    n = df.groupby("video_id")[["fps", "video_path"]].nunique(dropna=False)
    bad = n.index[(n > 1).any(axis=1)].tolist()
    if bad:
        refuse(f"videos whose questions differ in fps or file: {bad[:10]}")


# ---------------------------------------------------------------------------
# tokenizer / processor (CPU) and model (GPU)
# ---------------------------------------------------------------------------


class Prep:
    """Tokenizer and processor, loaded as evaluation.py _load_model_tools does, from the local HF cache."""

    def __init__(self):
        from transformers import AutoProcessor, AutoTokenizer

        kw = dict(revision=R.MODEL_REVISION, local_files_only=True, trust_remote_code=True, use_fast=True)
        self.tokenizer = AutoTokenizer.from_pretrained(R.MODEL_ID, **kw)
        self.processor = AutoProcessor.from_pretrained(R.MODEL_ID, **kw)
        template = self.processor.chat_template
        if hashlib.sha256(template.encode("utf-8")).hexdigest() != R.CHAT_TEMPLATE_SHA256:
            raise RuntimeError("checkpoint chat template differs from the authors' qwen3_5_no_think.jinja")
        self.tokenizer.chat_template = template
        self.processor.chat_template = template
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        ids = self.tokenizer.convert_tokens_to_ids(
            ["<|vision_start|>", "<|video_pad|>", "<|vision_end|>", "<|image_pad|>", "<|im_end|>", "<|endoftext|>"])
        if ids != [R.VISION_START_ID, R.VIDEO_TOKEN_ID, R.VISION_END_ID, R.IMAGE_TOKEN_ID, R.EOS_TOKEN_ID, R.PAD_TOKEN_ID]:
            raise RuntimeError(f"unexpected special token ids {ids}")

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def prompt(self, record: dict) -> tuple[str, list[int], bool]:
        """evaluation.py _build_prompt_ids (L302-315): rendered chat text, ids[:4096], truncated?"""
        rendered = self.processor.apply_chat_template(
            R.build_messages(record), add_generation_prompt=True, tokenize=False, enable_thinking=False)
        ids = self.encode(rendered)
        return rendered, ids[:R.MAX_PROMPT_LENGTH], len(ids) > R.MAX_PROMPT_LENGTH

    def pixels(self, video, meta: dict, processor_kwargs: dict):
        """pixel_values_videos, video_grid_thw from the HF processor called as vLLM 0.19.1 calls it
        (Qwen3VLMultiModalProcessor._call_hf_processor: placeholder text, [[frames]], [[metadata]])."""
        from transformers.video_utils import VideoMetadata

        out = self.processor(text=VIDEO_PLACEHOLDER, videos=[[video]], video_metadata=[[VideoMetadata(**meta)]],
                             return_tensors="pt", **processor_kwargs)
        return out["pixel_values_videos"], out["video_grid_thw"]


def gpu_check() -> None:
    import torch

    if not torch.cuda.is_available():
        sys.exit("CUDA not available")
    free, total = torch.cuda.mem_get_info()
    if free / 2**30 < MIN_FREE_GPU_GIB:
        sys.exit(f"refusing to start: {free / 2**30:.1f} GiB of {total / 2**30:.1f} GiB GPU memory free, need "
                 f"{MIN_FREE_GPU_GIB} GiB for the 17.5 GiB of weights (another GPU job running?)")


class Model:
    def __init__(self, tokenizer):
        import torch
        from transformers import AutoModelForImageTextToText, LogitsProcessor

        class LogitBias(LogitsProcessor):
            """vLLM SamplingParams(logit_bias=...): add the bias to the raw logits."""

            def __call__(self, input_ids, scores):
                scores = scores.clone()
                for tid, b in R.LOGIT_BIAS.items():
                    scores[:, tid] += b
                return scores

        self.torch = torch
        self.tokenizer = tokenizer
        self.logit_bias = LogitBias()
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, VRAM_CAP_GIB * 2**30 / total))
        torch.manual_seed(R.SEED)
        self.model = AutoModelForImageTextToText.from_pretrained(
            R.MODEL_ID, revision=R.MODEL_REVISION, local_files_only=True, dtype=torch.bfloat16,
            attn_implementation="sdpa", device_map="cuda:0",
        ).eval()

    def generate(self, ids: list[int], pixel_values_videos, video_grid_thw, literal_sampler: bool = False) -> dict:
        """One question, batch size 1. Default: argmax. literal_sampler=True: HF sampling with the
        authors' vLLM settings (temperature 0.01, top_p 0.001, no top-k, seed 1), for --recheck only.
        At most recipe.max_new_tokens(len(ids)) new tokens (512 unless the prompt nears max_model_len)."""
        from transformers import LogitsProcessorList

        torch = self.torch
        dev = "cuda:0"
        input_ids = torch.tensor([ids], dtype=torch.long, device=dev)
        max_new = R.max_new_tokens(len(ids))
        kw = dict(
            input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
            mm_token_type_ids=torch.tensor([R.mm_token_type_ids(ids)], dtype=torch.long, device=dev),
            pixel_values_videos=pixel_values_videos.to(dev), video_grid_thw=video_grid_thw.to(dev),
            max_new_tokens=max_new, use_cache=True, repetition_penalty=1.0,
            eos_token_id=R.EOS_TOKEN_ID, pad_token_id=R.PAD_TOKEN_ID,
            logits_processor=LogitsProcessorList([self.logit_bias]),
        )
        if literal_sampler:
            kw.update(do_sample=True, temperature=R.SAMPLING_CONFIG["temperature"], top_p=R.SAMPLING_CONFIG["top_p"], top_k=0)
        else:
            kw.update(do_sample=False)
        torch.manual_seed(R.SEED)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t = time.perf_counter()
        with torch.inference_mode():
            out = self.model.generate(**kw)
        torch.cuda.synchronize()
        latency = time.perf_counter() - t
        gen = out[0, len(ids):].tolist()
        reply = self.tokenizer.decode(gen, skip_special_tokens=True)  # vLLM default: special tokens skipped
        rec = {
            "reply": reply, "finish_reason": "stop" if gen and gen[-1] == R.EOS_TOKEN_ID else "length",
            "num_generated_tokens": len(gen),  # includes the final <|im_end|> (vLLM's completion.token_ids keeps the EOS id too)
            "max_new_tokens": max_new,
            "generated_ids": gen, "has_think_end": "</think>" in reply.lower(),
            "latency_s": round(latency, 3),
            "peak_vram_alloc_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3),
            "peak_vram_reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 3),
        }
        torch.cuda.empty_cache()
        return rec


# ---------------------------------------------------------------------------
# inference
# ---------------------------------------------------------------------------


def prepare_video(prep: Prep, row) -> dict:
    """Frames -> pixels, grid and timestamps of one video; the same for every question of the video."""
    t = time.perf_counter()
    v = read_video(row.video_path)
    video, meta, pkw = video_input(v, row.fps)
    pv, grid = prep.pixels(video, meta, pkw)
    grid_thw = [int(x) for x in grid[0].tolist()]
    raw_fps = dict(v["metadata"]).get("fps")
    return {
        "pixel_values_videos": pv, "video_grid_thw": grid,
        "info": {
            "video_backend": meta.get("video_backend"), "total_num_frames": meta["total_num_frames"],
            "decoded_avg_fps": float(raw_fps) if raw_fps is not None else None, "table_fps": meta["fps"],
            "frames_indices": meta["frames_indices"], "n_frames": int(video.shape[0]),
            "nframes_requested": v["nframes_requested"], "sample_fps": v["sample_fps"],
            "resized_hw_qwen_vl_utils": [int(video.shape[2]), int(video.shape[3])],
            "resized_hw_processor": [grid_thw[1] * 16, grid_thw[2] * 16], "video_grid_thw": grid_thw,
            "timestamps": R.calculate_timestamps(meta["frames_indices"], meta["fps"]),
            "video_events": v["events"], "short_video_deviation": v["deviation"],
            "video_prep_s": round(time.perf_counter() - t, 3),
        },
    }


def parse_fields(reply: str | None, target_unit: str | None) -> dict:
    """Both parsers on the raw reply: the authors' (first number, no units) and the repo's
    parse_answer_sci (unit conversion, thousands separators, x10^n). `prediction` is the repo
    rule used by methods/caw/combine.py: |sci value| when finite and non-zero, else missing
    (used_fallback: the combine step substitutes another answer)."""
    a = R.parse_prediction(reply) if reply is not None else None
    s = parse_answer_sci(reply, target_unit)
    ok = bool(s["ok"]) and R.usable(s["value"])
    return {
        "parsed_value_authors": a, "parsed_value_authors_csv": R.format_number(a), "parse_sci": s,
        "sci_ok": bool(s["ok"]), "used_fallback": not ok, "prediction": abs(s["value"]) if ok else None,
    }


def run_video(prep: Prep, model: Model, g: pd.DataFrame, items_dir: Path, stamp: dict) -> tuple[int, dict[int, str]]:
    """Answer the pending questions `g` of one video. Returns (records written, {qid: error} of the
    questions that failed and stay pending). `stamp` (versions, code, prereg) goes into every record."""
    assert g.fps.nunique(dropna=False) == 1 and g.video_path.nunique(dropna=False) == 1, g.video_id.iloc[0]
    try:
        vid = prepare_video(prep, g.iloc[0])
        decode_err = None
    except DecodeFailure as e:  # a property of the file: cached below, no question of this video gets a CaW answer
        vid, decode_err = None, str(e)
    except Exception as e:  # memory, I/O, the RAM guard, processor: not cached, retried by the next run
        err = f"video preparation failed (not cached): {type(e).__name__}: {str(e)[:300]}"
        print(f"[caw] {g.video_id.iloc[0]}: {err}", flush=True)
        return 0, {int(q): err for q in g.qid}
    n, failed = 0, {}
    for r in g.itertuples():
        record = R.record_from_row(r)
        rec = {"qid": int(r.qid), "video_id": r.video_id, "category": r.category, "authors_category": record["category"],
               "question": r.question, "prior": r.prior, "target_unit": r.target_unit,
               "model": R.MODEL_ID, "revision": R.MODEL_REVISION, "recipe": f"{R.AUTHORS_REPO}@{R.AUTHORS_COMMIT}"}
        if decode_err is not None:
            rec.update({"reply": None, "error": f"video decode failed: {decode_err}"})
            rec.update(parse_fields(None, r.target_unit))
            rec.update(stamp)
            print(f"[caw] {r.qid}: {rec['error'][:300]}", flush=True)
            write_json_atomic(items_dir / f"{r.qid}.json", rec, indent=1)
            n += 1
            continue
        rec.update(vid["info"])
        try:
            rendered, prompt_ids, truncated = prep.prompt(record)
            grid = vid["info"]["video_grid_thw"]
            ids = R.expand_video_tokens(prompt_ids, grid, vid["info"]["timestamps"], prep.encode)
            R.check_layout(ids, grid)
            rec.update({"prompt": rendered, "prompt_sha256": hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
                        "n_prompt_tokens_text": len(prompt_ids), "prompt_truncated": truncated, "n_prompt_tokens": len(ids)})
            rec.update(model.generate(ids, vid["pixel_values_videos"], vid["video_grid_thw"]))
            rec["error"] = None
        except Exception as e:  # model / prompt failure: not cached, retried by the next run
            failed[int(r.qid)] = f"{type(e).__name__}: {str(e)[:300]}"
            print(f"[caw] {r.qid}: {failed[int(r.qid)]}", flush=True)
            continue
        rec.update(parse_fields(rec["reply"], r.target_unit))
        rec.update(stamp)
        write_json_atomic(items_dir / f"{r.qid}.json", rec, indent=1)
        n += 1
    return n, failed


def pick_videos(df: pd.DataFrame, limit_videos: int) -> list[str]:
    """Probe: N videos spread evenly over the sorted video list of the whole split (pending or not),
    so the same probe command always names the same videos."""
    videos = sorted(df.video_id.unique())
    if not videos:
        return []
    return [videos[i] for i in np.linspace(0, len(videos) - 1, min(limit_videos, len(videos))).round().astype(int)]


def ram_available_gib() -> float | None:
    try:
        import psutil

        return round(psutil.virtual_memory().available / 2**30, 1)
    except Exception:  # noqa: BLE001 - informational only
        return None


def ensure_loaded(loaded: dict) -> dict:
    if "model" not in loaded:
        gpu_check()
        import psutil
        import torch

        ram = ram_available_gib()
        loaded["ram_available_before_load_gib"] = ram
        if ram is not None and ram < LOW_RAM_GIB:
            print(f"[caw] warning: {ram} GiB RAM available; loading maps the 18.8 GB checkpoint and may page heavily. "
                  "Stop other memory-heavy jobs (e.g. a virtual machine) for the full run.", flush=True)
        t = time.time()
        loaded["prep"] = Prep()
        loaded["model"] = Model(loaded["prep"].tokenizer)
        loaded["load_s"] = round(time.time() - t, 1)
        loaded["vram_after_load_gib"] = round(torch.cuda.memory_allocated() / 2**30, 2)
        loaded["rss_after_load_gib"] = round(psutil.Process().memory_info().rss / 2**30, 2)
        print(f"[caw] model loaded in {loaded['load_s']}s, {loaded['vram_after_load_gib']} GiB allocated, "
              f"process RSS {loaded['rss_after_load_gib']} GiB", flush=True)
    return loaded


def run_inference(df: pd.DataFrame, items_dir: Path, videos: list[str] | None = None,
                  max_videos: int | None = None, loaded: dict | None = None,
                  stamp: dict | None = None) -> tuple[dict, dict[int, str]]:
    """Answer every pending question of `videos` (default: every video with pending questions, in
    sorted order), at most max_videos of them, one video at a time. Returns (the loaded prep/model,
    {qid: error} of the questions that failed in this pass and stay pending)."""
    items_dir.mkdir(parents=True, exist_ok=True)
    loaded = loaded if loaded is not None else {}
    todo = df[[not (items_dir / f"{q}.json").exists() for q in df.qid]]
    chosen = sorted(todo.video_id.unique())
    if videos is not None:
        chosen = [v for v in chosen if v in set(videos)]
    if max_videos is not None:
        chosen = chosen[:max_videos]
    print(f"[caw] {len(todo)} questions left in {todo.video_id.nunique()} videos ({len(df)} total); "
          f"this run: {len(chosen)} videos", flush=True)
    if not chosen:
        return loaded, {}
    ensure_loaded(loaded)
    stamp = {"versions": versions()} | (stamp or {})
    t0, done, failed = time.time(), 0, {}
    n_q = int(todo.video_id.isin(chosen).sum())
    for vi, vid in enumerate(chosen):
        n, f = run_video(loaded["prep"], loaded["model"], todo[todo.video_id == vid], items_dir, stamp)
        done, failed = done + n, failed | f
        el = time.time() - t0
        if (vi + 1) % 5 == 0 or vi + 1 == len(chosen):
            print(f"[caw] {vi + 1}/{len(chosen)} videos, {done}/{n_q} questions, {len(failed)} failed, "
                  f"{el / 60:.1f} min, eta {el / max(done, 1) * (n_q - done - len(failed)) / 60:.1f} min", flush=True)
    return loaded, failed


def record_failures(split_dir: Path, df: pd.DataFrame, items_dir: Path, failed: dict[int, str]) -> dict:
    """runs/caw/<split>/failures.json: per pending question, how many runs it failed in and the last
    error. Questions that now have a record are dropped. Returns the updated table."""
    p = split_dir / "failures.json"
    table = (load_record(p) or {}) if p.exists() else {}
    for q, err in failed.items():
        prev = table.get(str(q), {})
        table[str(q)] = {"attempts": int(prev.get("attempts", 0)) + 1, "last_error": err,
                         "video_id": str(df.loc[df.qid == q, "video_id"].iloc[0])}
    table = {q: v for q, v in table.items() if not (items_dir / f"{q}.json").exists()}
    split_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(p, table, indent=1)
    return table


def report_failures(table: dict, failed: dict[int, str]) -> None:
    persistent = sorted(int(q) for q, v in table.items() if v["attempts"] >= 2)
    print(f"[caw] {len(failed)} questions failed in this run and stay pending (not cached): {sorted(failed)[:50]}",
          flush=True)
    if persistent:
        print(f"[caw] failed in 2 or more runs: {persistent[:50]} (see failures.json)", flush=True)


def recheck(loaded: dict, df: pd.DataFrame, items_dir: Path, qids: list[int]) -> list[dict]:
    """Re-answer cached questions in this process: argmax again (bit-identical?) and HF sampling with
    the authors' literal vLLM settings (same tokens?). Nothing is written to the cache."""
    ensure_loaded(loaded)
    prep, model = loaded["prep"], loaded["model"]
    out = []
    for q in qids:
        cached = load_record(items_dir / f"{q}.json")
        if not cached or cached.get("error"):
            continue
        r = next(df[df.qid == q].itertuples())
        vid = prepare_video(prep, r)
        rendered, prompt_ids, _ = prep.prompt(R.record_from_row(r))
        ids = R.expand_video_tokens(prompt_ids, vid["info"]["video_grid_thw"], vid["info"]["timestamps"], prep.encode)
        again = model.generate(ids, vid["pixel_values_videos"], vid["video_grid_thw"])
        literal = model.generate(ids, vid["pixel_values_videos"], vid["video_grid_thw"], literal_sampler=True)
        out.append({
            "qid": q, "reply": cached["reply"],
            "same_prompt": hashlib.sha256(rendered.encode("utf-8")).hexdigest() == cached["prompt_sha256"],
            "same_frames": vid["info"]["frames_indices"] == cached["frames_indices"],
            "greedy_again_identical": again["generated_ids"] == cached["generated_ids"],
            "literal_sampler_identical": literal["generated_ids"] == cached["generated_ids"],
            "literal_sampler_reply": literal["reply"], "latency_again_s": again["latency_s"],
        })
    return out


# ---------------------------------------------------------------------------
# outputs
# ---------------------------------------------------------------------------

ITEM_COLUMNS = (
    "qid", "category", "video_id", "video_backend", "n_frames", "total_num_frames", "table_fps", "decoded_avg_fps",
    "short_video_deviation", "n_prompt_tokens", "reply", "finish_reason", "num_generated_tokens",
    "has_think_end", "parsed_value_authors", "parsed_value_authors_csv", "sci_ok", "used_fallback", "prediction",
    "latency_s", "video_prep_s", "peak_vram_reserved_gib", "error",
)


def collect(df: pd.DataFrame, items_dir: Path) -> pd.DataFrame:
    rows, missing, unreadable = [], [], []
    for q in df.qid:
        p = items_dir / f"{q}.json"
        if not p.exists():
            missing.append(int(q))
            continue
        rec = load_record(p)
        if rec is None:
            unreadable.append(p.name)
            continue
        row = {k: rec.get(k) for k in ITEM_COLUMNS}
        row["video_grid_thw"] = "x".join(map(str, rec["video_grid_thw"])) if rec.get("video_grid_thw") else None
        s = rec["parse_sci"]
        row |= {"sci_value": s.get("value"), "sci_reply_unit": s.get("reply_unit"), "sci_converted": bool(s.get("converted"))}
        rows.append(row)
    if missing:
        raise SystemExit(f"{len(missing)} questions have no record in {items_dir} (e.g. {missing[:10]}); run inference first")
    if unreadable:
        raise SystemExit(f"unreadable records in {items_dir} (delete them and run again): {unreadable[:20]}")
    return pd.DataFrame(rows)


def items_code(items_dir: Path, qids) -> list[str]:
    """The distinct code_sha256 values stamped on these records."""
    return sorted({((load_record(items_dir / f"{q}.json") or {}).get("code") or {}).get("code_sha256") or "none"
                   for q in qids})


def _stats(s: pd.Series) -> dict | None:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if not len(s):
        return None
    return {"mean": round(float(s.mean()), 3), "median": round(float(s.median()), 3),
            "p95": round(float(s.quantile(0.95)), 3), "max": round(float(s.max()), 3)}


def parser_disagreements(items: pd.DataFrame) -> dict:
    """Label-free: questions where the authors' first-number parser and parse_answer_sci read different numbers."""
    a = pd.to_numeric(items.parsed_value_authors, errors="coerce")
    s = pd.to_numeric(items.sci_value, errors="coerce")
    conv = items.sci_converted.astype(bool)
    differ = a.notna() & s.notna() & ~np.isclose(a.abs(), s.abs(), rtol=1e-9, atol=0)
    return {
        "differ": int(differ.sum()),
        "differ_unit_converted": int((differ & conv).sum()), "differ_other": int((differ & ~conv).sum()),
        "authors_only_parsed": int((a.notna() & s.isna()).sum()), "sci_only_parsed": int((s.notna() & a.isna()).sum()),
        "examples": items.loc[differ, ["qid", "reply", "parsed_value_authors", "sci_value"]].head(10).to_dict("records"),
    }


def summarize(items: pd.DataFrame, vers: dict | None = None) -> dict:
    err = items.error.fillna("")
    lat = _stats(items.latency_s)
    decode_failed = err.str.startswith("video decode failed")
    authors = pd.to_numeric(items.parsed_value_authors, errors="coerce")
    return {
        "version": VERSION, "model": R.MODEL_ID, "revision": R.MODEL_REVISION,
        "recipe": f"{R.AUTHORS_REPO}@{R.AUTHORS_COMMIT}", "engine": "transformers generate, argmax, batch size 1",
        "n": int(len(items)), "errors": int((err != "").sum()),
        "video_decode_failures": int(decode_failed.sum()),
        "video_decode_failure_videos": sorted(items.loc[decode_failed, "video_id"].unique().tolist()),
        "short_video_deviation_videos": sorted(items.loc[items.short_video_deviation.notna(), "video_id"].unique().tolist()),
        "video_backends": items.video_backend.value_counts(dropna=False).to_dict(),
        "authors_parse_failures": int(authors.isna().sum()),
        "authors_negative": int((authors < 0).sum()), "authors_zero": int((authors == 0).sum()),
        "sci_parse_failures": int((~items.sci_ok.astype(bool)).sum()),
        "used_fallback": int(items.used_fallback.astype(bool).sum()),
        "used_fallback_by_category": items[items.used_fallback.astype(bool)].category.value_counts().to_dict(),
        "sci_unit_converted": int(items.sci_converted.sum()),
        "parser_disagreements": parser_disagreements(items),
        "finish_reasons": items.finish_reason.value_counts(dropna=False).to_dict(),
        "has_think_end": int(items.has_think_end.fillna(False).astype(bool).sum()),
        "generated_tokens": _stats(items.num_generated_tokens),
        "prompt_tokens": _stats(items.n_prompt_tokens),
        "latency_seconds": (lat | {"total_minutes": round(float(items.latency_s.sum()) / 60, 1)}) if lat else None,
        "video_prep_minutes_total": round(float(items.drop_duplicates("video_id").video_prep_s.fillna(0).sum()) / 60, 1),
        "peak_vram_reserved_gib_max": _stats(items.peak_vram_reserved_gib)["max"] if lat else None,
        "versions": vers,
        "port_differences": list(PORT_DIFFERENCES),
    }


def prereg_sha256(path: str) -> str:
    """The validation run needs a committed, unmodified pre-registration file; returns its sha256."""
    p = Path(path) if Path(path).is_absolute() else (ROOT / path)
    if not p.is_file():
        refuse(f"--prereg {path}: no such file")
    rel = p.resolve().relative_to(ROOT).as_posix()
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", rel], cwd=ROOT, capture_output=True).returncode == 0
    clean = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", rel], cwd=ROOT).returncode == 0
    if not (tracked and clean):
        refuse(f"--prereg {rel} must be committed and unmodified before the validation run")
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["val", "test"], required=True)
    ap.add_argument("--limit-videos", type=int, default=None, help="probe (test only): N videos spread over the list")
    ap.add_argument("--max-videos", type=int, default=None, help="stop after the first N pending videos (chunked runs)")
    ap.add_argument("--recheck", type=int, default=0, help="probe only: re-answer the first N probe questions (determinism)")
    ap.add_argument("--prereg", default=None, help="val only: the committed pre-registration file")
    args = ap.parse_args(argv)
    if args.split == "val":
        if args.limit_videos is not None or args.recheck:
            ap.error("--limit-videos / --recheck are test-only for now")
        if not args.prereg:
            ap.error("--split val needs --prereg <committed pre-registration file>")
    if args.recheck and args.limit_videos is None:
        ap.error("--recheck needs --limit-videos")
    probe = args.limit_videos is not None
    prereg = {"file": args.prereg, "sha256": prereg_sha256(args.prereg)} if args.split == "val" else None
    df = questions(args.split)
    one_video_input(df)
    split_dir = RUNS / args.split
    items_dir = split_dir / "items"
    code = code_state()
    if not probe:
        require_committed_code(items_dir, df.qid, code, f"--split {args.split}", prereg)
    elif code["dirty"]:
        print(f"[caw] warning: uncommitted changes in {code['dirty_files']}: these probe records will block the "
              "full run until they are deleted", flush=True)
    t0 = time.time()
    videos = pick_videos(df, args.limit_videos) if probe else None
    stamp = {"code": code} | ({"prereg": prereg} if prereg else {})
    loaded, failed = run_inference(df, items_dir, videos, args.max_videos, stamp=stamp)
    if failed or (split_dir / "failures.json").exists():
        table = record_failures(split_dir, df, items_dir, failed)
        if failed:
            report_failures(table, failed)
    if probe:
        if failed:
            return 1
        d = collect(df[df.video_id.isin(videos)], items_dir)
        s = summarize(d, versions()) | {
            "code": code, "items_code_sha256": items_code(items_dir, d.qid),
            "probe_videos": videos, "wall_minutes": round((time.time() - t0) / 60, 2),
            "model_load_s": loaded.get("load_s"), "vram_after_load_gib": loaded.get("vram_after_load_gib"),
            "rss_after_load_gib": loaded.get("rss_after_load_gib"),
            "ram_available_before_load_gib": loaded.get("ram_available_before_load_gib"),
            "estimated_full_split_hours_excl_load": round(float(d.latency_s.mean()) * len(df) / 3600, 2)
            if d.latency_s.notna().any() else None,
            "replies": d[["qid", "video_id", "reply", "finish_reason", "num_generated_tokens", "parsed_value_authors",
                          "sci_value", "used_fallback", "latency_s"]].to_dict("records"),
        }
        if args.recheck:
            s["recheck"] = recheck(loaded, df, items_dir, d.qid.tolist()[:args.recheck])
        try:
            import psutil

            mi = psutil.Process().memory_info()
            s["process_peak_rss_gib"] = round(getattr(mi, "peak_wset", mi.rss) / 2**30, 2)
        except Exception:  # noqa: BLE001 - informational only
            pass
        out = split_dir / "probe_summary.json"
        write_json_atomic(out, s, indent=2, default=str)
        print(json.dumps(s, indent=1, default=str))
        print(f"wrote {out}")
        return 0
    if failed:
        return 1
    pending = df[[not (items_dir / f"{q}.json").exists() for q in df.qid]]
    if len(pending):
        print(f"[caw] {len(pending)} questions in {pending.video_id.nunique()} videos still pending; "
              "run the same command again to continue", flush=True)
        return 3
    final = code_state()  # the code may have been committed or edited while the run was going
    require_committed_code(items_dir, df.qid, final, "collect", prereg)
    items = collect(df, items_dir)
    items.to_csv(split_dir / "items.csv", index=False)
    summ = summarize(items, versions()) | {"code": final, "items_code_sha256": items_code(items_dir, df.qid)} | (
        {"prereg": prereg} if prereg else {})
    out = ROOT / "results" / f"{VERSION}_{args.split}_summary.json"
    write_json_atomic(out, summ, indent=2, default=str)
    print(json.dumps(summ, indent=1, default=str))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
