"""One-command runner for the open-weight VLM baseline (direct answering).

    uv run python -m methods.vlm_baseline.run --split val
    uv run python -m methods.vlm_baseline.run --split test

Model: Qwen/Qwen3-VL-8B-Instruct (Apache-2.0, not gated), bf16, greedy decoding.
Input per question: up to NUM_FRAMES uniformly sampled frames passed as a *video*
(each frame is labeled by the processor with its timestamp computed from the table
fps), the prior, the depth info (3D), the fps / duration, and the question. The
prompt follows the official starter-kit zero-shot prompt (run_API_results.py,
version 3) and asks for the number and unit only.

Per-question records (raw reply, parsed value, latency, peak VRAM) are cached in
runs/vlm_baseline/<split>/items/<qid>.json, so an interrupted run resumes.
Outputs:
  val : results/predictions/val_vlm_baseline.csv, results/vlm_baseline_val_summary.json
  test: submissions/vlm_baseline_test.csv (official template, only parsed_value filled),
        results/vlm_baseline_test_summary.json
Predictions are produced 100% by this program; the validation answers are dropped
before inference and only used afterwards for scoring.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import av
import numpy as np
import pandas as pd

from quantiphy.data import ROOT, load_template, load_test, load_validation, write_predictions

from methods.geometry.parse import KIND_DIM, parse_prior, parse_question, si_to_unit
from methods.geometry.solver import fallback_value, reference_constants

from .answer import parse_answer

VERSION = "vlm_baseline"
MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"  # Apache-2.0, not gated
MODEL_REVISION = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
NUM_FRAMES = 32  # uniformly sampled frames per video (all frames if the clip is shorter)
MAX_SIDE = 960  # decode-time downscale (keeps 4K clips cheap); the processor resizes further
FRAME_PIXELS = 480 * 270  # per-frame pixel budget; 640x360 (~4k tokens) pushed peak VRAM past 24 GB with the desktop
MAX_NEW_TOKENS = 32
MIN_FREE_GPU_GB = 19.0
# Hard cap for this process. On Windows (WDDM) the driver silently spills past-capacity
# allocations into system RAM, which made single questions take minutes; with a cap the
# allocator frees its cache and retries, and a real OOM surfaces as a recorded error.
VRAM_CAP_GIB = 21.0
RUNS = ROOT / "runs" / VERSION

# Official starter-kit system prompt (zero-shot, run_API_results.py), unchanged.
SYSTEM_PROMPT = (
    "You are an expert video analyst specializing in physics measurements.\n"
    "Analyze the video frames carefully and provide ONLY the numerical answer with units. No explanation or reasoning needed.\n"
    "Format your response as: [value] [unit]\n"
    "Example: 2.5 cm\n"
    "Be as accurate as possible with measurements and calculations. Please give me an estimated answer even if you are not sure."
)


def context_prefix(prior, depth_info) -> str:
    """Same wording as build_context_prefix() in the official starter kit."""
    parts = []
    if isinstance(prior, str) and prior.strip():
        parts.append(f"Given that {prior.strip()}.")
    if isinstance(depth_info, str) and depth_info.strip():
        parts.append(
            "Additionally, you have the following information about the distance between the objects in the video "
            f"and the shooting camera: {depth_info.strip()}"
        )
    if not parts:
        return ""
    c = " ".join(parts).strip()
    if c[-1] not in ".!?":
        c += "."
    return c + " "


def build_prompt(row, n_total: int, n_shown: int) -> str:
    dur = n_total / row.fps if row.fps else float("nan")
    video_note = (
        f"The video is recorded at {row.fps} frames per second and lasts {dur:.2f} s ({n_total} frames). "
        f"{n_shown} frames sampled uniformly are shown, each labeled with its time in seconds. "
    )
    return (
        f"{video_note}{context_prefix(row.prior, row.depth_info)}{row.question}\n\n"
        "Please answer the question with numbers and units ONLY. No explanation needed."
    )


def read_video(path: str, num_frames: int = NUM_FRAMES, max_side: int = MAX_SIDE):
    """Decode all frames once, keep `num_frames` uniformly spaced ones (RGB uint8).

    Returns (kept frame indices, frames [T,H,W,3], total decoded frame count).
    Frame index / table fps is the timestamp, as in the official starter kit.
    """
    with av.open(path) as c:
        n_est = sum(1 for pkt in c.demux(c.streams.video[0]) if pkt.size > 0)
    want = set(np.linspace(0, max(n_est - 1, 0), min(num_frames, max(n_est, 1))).round().astype(int).tolist())
    idx, imgs, n = [], [], 0
    with av.open(path) as c:
        s = c.streams.video[0]
        s.thread_type = "AUTO"
        for fr in c.decode(s):
            if n in want:
                h0, w0 = fr.height, fr.width
                sc = min(1.0, max_side / max(h0, w0))
                if sc < 1.0:
                    fr = fr.reformat(width=int(round(w0 * sc)) // 2 * 2, height=int(round(h0 * sc)) // 2 * 2)
                imgs.append(fr.to_ndarray(format="rgb24"))
                idx.append(n)
            n += 1
    if len(imgs) == 1:  # the processor needs >= 2 frames (temporal patch size)
        imgs.append(imgs[0])
        idx.append(idx[0])
    return np.array(idx), np.stack(imgs), n


class QwenVL:
    def __init__(self):
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        self.torch = torch
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, VRAM_CAP_GIB * 2**30 / total))
        self.processor = AutoProcessor.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL_ID, revision=MODEL_REVISION, dtype=torch.bfloat16, attn_implementation="sdpa"
        ).to("cuda").eval()

    def ask(self, frames: np.ndarray, frame_idx: np.ndarray, fps: float, n_total: int, text: str) -> tuple[str, int]:
        from transformers.video_utils import VideoMetadata

        torch = self.torch
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [{"type": "video"}, {"type": "text", "text": text}]},
        ]
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        meta = VideoMetadata(
            total_num_frames=n_total, fps=float(fps), width=frames.shape[2], height=frames.shape[1],
            duration=n_total / fps, frames_indices=[int(i) for i in frame_idx],
        )
        t = len(frames)
        inputs = self.processor(
            text=[prompt], videos=[frames], video_metadata=[meta], do_sample_frames=False,
            # flat kwarg on purpose: a nested `videos_kwargs` dict is silently ignored by transformers 4.57
            size={"shortest_edge": 128 * 32 * 32, "longest_edge": FRAME_PIXELS * max(t, 2)},
            return_tensors="pt",
        ).to("cuda")
        with torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        gen = out[0, inputs["input_ids"].shape[1]:]
        reply = self.processor.decode(gen, skip_special_tokens=True).strip()
        return reply, int(inputs["input_ids"].shape[1])


def gpu_check() -> None:
    import torch

    if not torch.cuda.is_available():
        sys.exit("CUDA not available")
    free, _ = torch.cuda.mem_get_info()
    if free / 2**30 < MIN_FREE_GPU_GB:
        sys.exit(f"refusing to start: only {free / 2**30:.1f} GB free GPU memory (another GPU job running?)")


def run_inference(df: pd.DataFrame, items_dir: Path, ref: dict, limit_videos: int | None = None) -> None:
    items_dir.mkdir(parents=True, exist_ok=True)
    todo = df[[not (items_dir / f"{q}.json").exists() for q in df.qid]]
    videos = sorted(todo.video_id.unique())
    if limit_videos is not None:  # timing probe: N pending videos spread evenly over the list
        videos = [videos[i] for i in np.linspace(0, len(videos) - 1, min(limit_videos, len(videos))).round().astype(int)]
    print(f"[vlm] {len(todo)} questions left in {todo.video_id.nunique()} videos ({len(df)} total)", flush=True)
    if not videos:
        return
    gpu_check()
    import torch

    t_load = time.time()
    vlm = QwenVL()
    print(f"[vlm] model loaded in {time.time() - t_load:.0f}s, "
          f"{torch.cuda.memory_allocated() / 2**30:.1f} GiB allocated", flush=True)
    t0, done = time.time(), 0
    n_q = int(todo.video_id.isin(videos).sum())
    for vi, vid in enumerate(videos):
        g = todo[todo.video_id == vid]
        td = time.perf_counter()
        try:
            fidx, frames, n_total = read_video(g.iloc[0].video_path)
            decode_err = None
        except Exception as e:  # keep going; every question of this video gets the fallback
            fidx, frames, n_total, decode_err = None, None, 0, repr(e)
        decode_s = time.perf_counter() - td
        for r in g.itertuples():
            qs = parse_question(r.question)
            dim = KIND_DIM.get(qs.get("kind"), "L")
            fb_si, fb_kind = fallback_value(qs, parse_prior(r.prior), ref)
            rec = {
                "qid": int(r.qid), "video_id": vid, "category": r.category, "question": r.question,
                "prior": r.prior, "target_unit": r.target_unit, "model": MODEL_ID, "revision": MODEL_REVISION,
                "fallback_value": si_to_unit(fb_si, r.target_unit, dim), "fallback_kind": fb_kind,
                "video_decode_s": round(decode_s, 3), "n_video_frames": n_total,
            }
            if decode_err is None:
                text = build_prompt(r, n_total, len(frames))
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                tq = time.perf_counter()
                try:
                    reply, n_tok = vlm.ask(frames, fidx, r.fps, n_total, text)
                    err = None
                except Exception as e:
                    reply, n_tok, err = None, None, repr(e)
                torch.cuda.synchronize()
                infer_s = time.perf_counter() - tq
                rec.update({
                    "prompt": text, "n_frames_shown": int(len(frames)), "frame_size": list(frames.shape[1:3]),
                    "input_tokens": n_tok, "reply": reply, "error": err,
                    "infer_s": round(infer_s, 3),
                    "peak_vram_alloc_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3),
                    "peak_vram_reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 3),
                })
                torch.cuda.empty_cache()  # keep the cached pool from creeping up to the cap
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
            print(f"[vlm] {vi + 1}/{len(videos)} videos, {done}/{n_q} questions, {el / 60:.1f} min, "
                  f"eta {el / max(done, 1) * (n_q - done) / 60:.1f} min", flush=True)


def collect(df: pd.DataFrame, items_dir: Path) -> pd.DataFrame:
    rows = []
    for q in df.qid:
        p = items_dir / f"{q}.json"
        if not p.exists():
            raise SystemExit(f"missing {p}; run inference first")
        rec = json.loads(p.read_text(encoding="utf-8"))
        rows.append({k: rec.get(k) for k in (
            "qid", "category", "video_id", "parse_ok", "used_fallback", "prediction", "reply",
            "infer_s", "video_decode_s", "input_tokens", "peak_vram_alloc_gib", "peak_vram_reserved_gib", "error",
        )} | {"unit_converted": bool(rec["parse"].get("converted"))})
    return pd.DataFrame(rows)


def summarize(items: pd.DataFrame) -> dict:
    it = items.infer_s.dropna()
    vr = items.peak_vram_reserved_gib.dropna()
    va = items.peak_vram_alloc_gib.dropna()
    return {
        "version": VERSION, "model": MODEL_ID, "revision": MODEL_REVISION, "num_frames": NUM_FRAMES,
        "frame_pixels": FRAME_PIXELS, "max_new_tokens": MAX_NEW_TOKENS, "decoding": "greedy",
        "n": int(len(items)),
        "parse_failures": int((~items.parse_ok).sum()),
        "parse_failure_rate": float((~items.parse_ok).mean()),
        "parse_failures_by_category": items.groupby("category").parse_ok.apply(lambda s: int((~s).sum())).to_dict(),
        "fallback_used": int(items.used_fallback.sum()),
        "errors": int(items.error.notna().sum()),
        "unit_converted": int(items.unit_converted.sum()),
        "infer_seconds": {
            "mean": float(it.mean()), "median": float(it.median()), "p95": float(it.quantile(0.95)),
            "max": float(it.max()), "total_minutes": float(it.sum() / 60),
        },
        "video_decode_seconds_total": float(items.drop_duplicates("video_id").video_decode_s.sum()),
        "input_tokens": {"mean": float(items.input_tokens.mean()), "max": float(items.input_tokens.max())},
        "peak_vram_gib": {
            "allocated_max": float(va.max()), "reserved_max": float(vr.max()),
            "allocated_median": float(va.median()),
        },
    }


def write_probe(df: pd.DataFrame, items_dir: Path, split: str) -> None:
    """Runtime estimate for the whole split from the questions cached so far."""
    recs = [json.loads(p.read_text(encoding="utf-8")) for p in items_dir.glob("*.json")]
    d = pd.DataFrame(recs)
    v = d.drop_duplicates("video_id")
    n_q, n_v = len(df), df.video_id.nunique()
    est_min = (n_v * v.video_decode_s.mean() + n_q * d.infer_s.mean()) / 60
    done_min = (v.video_decode_s.sum() + d.infer_s.sum()) / 60
    out = {
        "split": split, "questions_done": int(len(d)), "videos_done": int(len(v)),
        "questions_total": int(n_q), "videos_total": int(n_v),
        "infer_s_mean": float(d.infer_s.mean()), "infer_s_median": float(d.infer_s.median()),
        "infer_s_max": float(d.infer_s.max()), "video_decode_s_mean": float(v.video_decode_s.mean()),
        "peak_vram_reserved_gib_max": float(d.peak_vram_reserved_gib.max()),
        "parse_failures": int((~d.parse_ok).sum()), "errors": int(d.error.notna().sum()),
        "estimated_total_minutes_excl_model_load": round(est_min, 1),
        "estimated_remaining_minutes": round(est_min - done_min, 1),
    }
    path = ROOT / "results" / f"{VERSION}_{split}_probe.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=1))
    print(f"wrote {path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["val", "test"], required=True)
    ap.add_argument("--limit-videos", type=int, default=None, help="timing probe: only run N pending videos spread evenly over the list")
    args = ap.parse_args()

    df = load_validation() if args.split == "val" else load_test()
    if args.split == "val":
        df = df.drop(columns=["answer"])  # inference never sees answers
    ref = reference_constants(load_test())
    items_dir = RUNS / args.split / "items"
    run_inference(df, items_dir, ref, args.limit_videos)
    if args.limit_videos is not None:
        write_probe(df, items_dir, args.split)
        return 0

    items = collect(df, items_dir)
    items.to_csv(RUNS / args.split / "items.csv", index=False)
    summ = summarize(items)
    if args.split == "val":
        val = load_validation()
        preds = items.set_index("qid").loc[val.qid, "prediction"].to_numpy()
        pred_path = ROOT / "results" / "predictions" / f"val_{VERSION}.csv"
        write_predictions(pred_path, val, preds)
        summ["predictions"] = pred_path.relative_to(ROOT).as_posix()
        out = ROOT / "results" / f"{VERSION}_val_summary.json"
    else:
        tmpl = load_template()
        tmpl["parsed_value"] = items.set_index("qid").loc[tmpl["id"], "prediction"].to_numpy()
        sub = ROOT / "submissions" / f"{VERSION}_test.csv"
        sub.parent.mkdir(parents=True, exist_ok=True)
        tmpl.to_csv(sub, index=False)
        summ["submission"] = sub.relative_to(ROOT).as_posix()
        out = ROOT / "results" / f"{VERSION}_test_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summ, indent=2), encoding="utf-8")
    print(json.dumps(summ, indent=1))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
