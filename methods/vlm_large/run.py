"""Qwen3-VL-32B-Instruct (4-bit, local) answering with exactly the vlm_baseline recipe.

    uv run python -m methods.vlm_large.run --split val              # all 159 questions
    uv run python -m methods.vlm_large.run --split test             # only the 797 questions
                                                                    # hybrid_v2 answers with the VLM
    uv run python -m methods.vlm_large.run --split test --limit-videos 5   # timing / VRAM probe

Same prompt, frame sampling, decoding and answer parsing as methods/vlm_baseline
(imported, not copied); the only change is the model: Qwen/Qwen3-VL-32B-Instruct at a
pinned revision, quantized on load with bitsandbytes (NF4, double quant, bf16 compute;
vision encoder and lm_head quantized too; leaving them unquantized ran out of memory in
a probe) so it fits a 24 GB GPU.
Per-question records are cached in runs/vlm_large/<split>/items/<qid>.json (resumable).
Validation answers are dropped before inference.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from quantiphy.data import ROOT, load_test, load_validation

from methods.geometry.parse import KIND_DIM, parse_prior, parse_question, si_to_unit
from methods.geometry.solver import fallback_value, reference_constants
from methods.vlm_baseline import run as base
from methods.vlm_baseline.answer import parse_answer

VERSION = "vlm_large"
MODEL_ID = "Qwen/Qwen3-VL-32B-Instruct"  # Apache-2.0, not gated
MODEL_REVISION = "0cfaf48183f594c314753d30a4c4974bc75f3ccb"
QUANT = {"load_in_4bit": True, "bnb_4bit_quant_type": "nf4", "bnb_4bit_use_double_quant": True,
         "bnb_4bit_compute_dtype": "bfloat16", "llm_int8_skip_modules": []}
VRAM_CAP_GIB = 22.0
MIN_FREE_GPU_GB = 21.0
RUNS = ROOT / "runs" / VERSION


class QwenVL32(base.QwenVL):
    """vlm_baseline.QwenVL with a different checkpoint loaded in 4-bit; ask() is inherited."""

    def __init__(self):
        import torch
        from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

        self.torch = torch
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, VRAM_CAP_GIB * 2**30 / total))
        q = dict(QUANT, bnb_4bit_compute_dtype=torch.bfloat16)
        self.processor = AutoProcessor.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL_ID, revision=MODEL_REVISION, dtype=torch.bfloat16, attn_implementation="sdpa",
            quantization_config=BitsAndBytesConfig(**q), device_map="cuda:0",
        ).eval()


def questions(split: str) -> pd.DataFrame:
    if split == "val":
        return load_validation().drop(columns=["answer"])  # inference never sees answers
    df = load_test()
    v2 = pd.read_csv(ROOT / "runs" / "hybrid_v2" / "test" / "items.csv").set_index("qid")
    keep = v2.index[v2.source.str.startswith("vlm")]
    return df[df.qid.isin(keep)]


def run_inference(df: pd.DataFrame, items_dir: Path, ref: dict, limit_videos: int | None = None) -> None:
    import torch

    items_dir.mkdir(parents=True, exist_ok=True)
    todo = df[[not (items_dir / f"{q}.json").exists() for q in df.qid]]
    videos = sorted(todo.video_id.unique())
    if limit_videos is not None and videos:
        videos = [videos[i] for i in np.linspace(0, len(videos) - 1, min(limit_videos, len(videos))).round().astype(int)]
    print(f"[vlm32] {len(todo)} questions left in {todo.video_id.nunique()} videos ({len(df)} total)", flush=True)
    if not videos:
        return
    free, _ = torch.cuda.mem_get_info()
    if free / 2**30 < MIN_FREE_GPU_GB:
        sys.exit(f"refusing to start: only {free / 2**30:.1f} GB free GPU memory (another GPU job running?)")
    t_load = time.time()
    vlm = QwenVL32()
    print(f"[vlm32] model loaded in {time.time() - t_load:.0f}s, "
          f"{torch.cuda.memory_allocated() / 2**30:.1f} GiB allocated", flush=True)
    t0, done = time.time(), 0
    n_q = int(todo.video_id.isin(videos).sum())
    for vi, vid in enumerate(videos):
        g = todo[todo.video_id == vid]
        td = time.perf_counter()
        try:
            fidx, frames, n_total = base.read_video(g.iloc[0].video_path)
            decode_err = None
        except Exception as e:
            fidx, frames, n_total, decode_err = None, None, 0, repr(e)
        decode_s = time.perf_counter() - td
        for r in g.itertuples():
            qs = parse_question(r.question)
            dim = KIND_DIM.get(qs.get("kind"), "L")
            fb_si, fb_kind = fallback_value(qs, parse_prior(r.prior), ref)
            rec = {"qid": int(r.qid), "video_id": vid, "category": r.category, "question": r.question,
                   "prior": r.prior, "target_unit": r.target_unit, "model": MODEL_ID, "revision": MODEL_REVISION,
                   "quantization": QUANT, "fallback_value": si_to_unit(fb_si, r.target_unit, dim),
                   "fallback_kind": fb_kind, "video_decode_s": round(decode_s, 3), "n_video_frames": n_total}
            if decode_err is None:
                text = base.build_prompt(r, n_total, len(frames))
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                tq = time.perf_counter()
                try:
                    reply, n_tok = vlm.ask(frames, fidx, r.fps, n_total, text)
                    err = None
                except Exception as e:
                    reply, n_tok, err = None, None, repr(e)
                torch.cuda.synchronize()
                rec.update({
                    "prompt": text, "n_frames_shown": int(len(frames)), "input_tokens": n_tok, "reply": reply,
                    "error": err, "infer_s": round(time.perf_counter() - tq, 3),
                    "peak_vram_alloc_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3),
                    "peak_vram_reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 3),
                })
                torch.cuda.empty_cache()
            else:
                rec.update({"reply": None, "error": f"video decode failed: {decode_err}", "infer_s": None})
            pa = parse_answer(rec["reply"], r.target_unit)
            rec["parse"] = pa
            rec["parse_ok"] = bool(pa["ok"])
            usable = pa["ok"] and math.isfinite(pa["value"]) and pa["value"] != 0
            rec["used_fallback"] = not usable
            rec["prediction"] = abs(pa["value"]) if usable else rec["fallback_value"]
            (items_dir / f"{r.qid}.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
            done += 1
        el = time.time() - t0
        if (vi + 1) % 5 == 0 or vi + 1 == len(videos):
            print(f"[vlm32] {vi + 1}/{len(videos)} videos, {done}/{n_q} questions, {el / 60:.1f} min, "
                  f"eta {el / max(done, 1) * (n_q - done) / 60:.1f} min", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["val", "test"], required=True)
    ap.add_argument("--limit-videos", type=int, default=None, help="probe: only N pending videos spread over the list")
    args = ap.parse_args()
    df = questions(args.split)
    items_dir = RUNS / args.split / "items"
    run_inference(df, items_dir, reference_constants(load_test()), args.limit_videos)
    if args.limit_videos is not None:
        return 0
    items = base.collect(df, items_dir)
    items.to_csv(RUNS / args.split / "items.csv", index=False)
    summ = base.summarize(items) | {"version": VERSION, "model": MODEL_ID, "revision": MODEL_REVISION,
                                    "quantization": QUANT, "questions": "all" if args.split == "val" else "hybrid_v2 VLM-answered only"}
    out = ROOT / "results" / f"{VERSION}_{args.split}_summary.json"
    out.write_text(json.dumps(summ, indent=2), encoding="utf-8")
    print(json.dumps(summ, indent=1))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
