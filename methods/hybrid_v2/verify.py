"""Stage 1 of hybrid_v2 (GPU): ask the VLM whether each geometry track is the named object.

    uv run python -m methods.hybrid_v2.verify --split val
    uv run python -m methods.hybrid_v2.verify --split test

For every distinct track that geometry_v1 used (target objects and the prior object;
see tracks.py) we show Qwen3-VL-8B-Instruct (the same weights and revision as
vlm_baseline) a few frames of the track, one at a time: the full frame with the
tracked box outlined in red, plus a zoomed crop of that region, and ask

    "... Text that comes with the video: "<question or prior>"
     Is the object inside the red box the <query> referred to in that text? ... Answer yes or no."

The answer is read from the next-token distribution (one forward pass, no sampling):
p_yes = P(yes) / (P(yes) + P(no)), summed over "yes"/"Yes" and "no"/"No". Raw p_yes
per frame is cached in runs/hybrid_v2/<split>/verify/<video_id>.json (resumable);
the yes/no threshold is applied later by hybrid.py from config.json. Nothing here
reads answers, and it never changes a geometry or VLM prediction.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw

from quantiphy.data import ROOT, load_test, load_validation

from methods.vlm_baseline.run import MODEL_ID, MODEL_REVISION, VRAM_CAP_GIB

from .tracks import det_frame_size, frames_to_check, track_table, used_tracks

VERSION = "hybrid_v2"
RUNS = ROOT / "runs" / VERSION
CONFIG = Path(__file__).with_name("config.json")
MIN_FREE_GPU_GB = 19.0
DET_MAX_SIDE = 960  # geometry_v1 detection frames (detect.read_frames); boxes live in these pixels

PROMPT = (
    "Image 1 is a frame from a video (it may be a computer-generated simulation, so objects can look "
    "simplified, small or blurry). One region is outlined with a red box. "
    "Image 2 is a zoomed-in view of the same region.\n"
    "Text that comes with the video: \"{context}\"\n"
    "Is the object inside the red box the {query} referred to in that text? "
    "It counts even if it looks small, blurry or simplified. Answer yes or no."
)


def load_config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def read_det_frames(path: str, wanted: set[int]) -> dict[int, np.ndarray]:
    """Decode the wanted frame indices, downscaled exactly like detect.read_frames."""
    out: dict[int, np.ndarray] = {}
    last = max(wanted)
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        s.thread_type = "AUTO"
        for n, fr in enumerate(c.decode(s)):
            if n in wanted:
                h0, w0 = fr.height, fr.width
                sc = min(1.0, DET_MAX_SIDE / max(h0, w0))
                if sc < 1.0:
                    fr = fr.reformat(width=int(round(w0 * sc)), height=int(round(h0 * sc)))
                out[n] = fr.to_ndarray(format="rgb24")
            if n >= last:
                break
    return out


def _outline(img: Image.Image, box, gap: float, width: int) -> None:
    """Red rectangle drawn just outside the box, so it never covers a small object."""
    x1, y1, x2, y2 = box
    ImageDraw.Draw(img).rectangle([x1 - gap - width, y1 - gap - width, x2 + gap + width, y2 + gap + width],
                                  outline=(255, 0, 0), width=width)


def render(frame: np.ndarray, box: np.ndarray, crop_side: int, margin: float) -> tuple[Image.Image, Image.Image]:
    """(full frame with a red box, zoomed crop around the box with the same outline).

    The crop is resized first and outlined afterwards, so the line stays thin."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in box]
    img = Image.fromarray(frame)
    full = img.copy()
    _outline(full, (x1, y1, x2, y2), gap=2, width=max(2, round(max(h, w) / 480)))
    bw, bh = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
    m = margin * max(bw, bh) + 16
    cx1, cy1 = max(0, int(x1 - m)), max(0, int(y1 - m))
    cx2, cy2 = min(w, int(np.ceil(x2 + m))), min(h, int(np.ceil(y2 + m)))
    cx2, cy2 = max(cx2, cx1 + 1), max(cy2, cy1 + 1)
    sc = crop_side / max(cx2 - cx1, cy2 - cy1)
    crop = img.crop((cx1, cy1, cx2, cy2))
    crop = crop.resize((max(1, round((cx2 - cx1) * sc)), max(1, round((cy2 - cy1) * sc))), Image.BICUBIC)
    _outline(crop, ((x1 - cx1) * sc, (y1 - cy1) * sc, (x2 - cx1) * sc, (y2 - cy1) * sc), gap=3, width=3)
    return full, crop


class Verifier:
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
        tok = self.processor.tokenizer
        self.yes_ids = sorted({tok.encode(w, add_special_tokens=False)[0] for w in ("yes", "Yes")})
        self.no_ids = sorted({tok.encode(w, add_special_tokens=False)[0] for w in ("no", "No")})

    def p_yes(self, full: Image.Image, crop: Image.Image, query: str, context: str) -> tuple[float, float]:
        """-> (p_yes among yes/no tokens, total probability mass on yes/no tokens)."""
        torch = self.torch
        messages = [{"role": "user", "content": [
            {"type": "image"}, {"type": "image"}, {"type": "text", "text": PROMPT.format(query=query, context=context.strip())},
        ]}]
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[prompt], images=[full, crop], return_tensors="pt").to("cuda")
        with torch.inference_mode():
            logits = self.model(**inputs).logits[0, -1].float()
        prob = torch.softmax(logits, -1)
        py, pn = float(prob[self.yes_ids].sum()), float(prob[self.no_ids].sum())
        return py / max(py + pn, 1e-12), py + pn


def gpu_check() -> None:
    import torch

    if not torch.cuda.is_available():
        sys.exit("CUDA not available")
    free, _ = torch.cuda.mem_get_info()
    if free / 2**30 < MIN_FREE_GPU_GB:
        sys.exit(f"refusing to start: only {free / 2**30:.1f} GB free GPU memory (another GPU job running?)")


def run(split: str, limit_videos: int | None = None) -> None:
    cfg = load_config()["verifier"]
    df = load_validation().drop(columns=["answer"]) if split == "val" else load_test()
    paths = df.drop_duplicates("video_id").set_index("video_id").video_path
    table = track_table(used_tracks(df, split))
    out_dir = RUNS / split / "verify"
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = sorted(v for v in table if not (out_dir / f"{v}.json").exists())
    if limit_videos is not None:
        todo = [todo[i] for i in np.linspace(0, len(todo) - 1, min(limit_videos, len(todo))).round().astype(int)] if todo else []
    n_tracks = sum(len(table[v]) for v in todo)
    print(f"[verify] {split}: {len(todo)} videos / {n_tracks} tracks to check "
          f"({len(table)} videos, {sum(len(t) for t in table.values())} tracks in total)", flush=True)
    if not todo:
        return
    gpu_check()
    t_load = time.time()
    ver = Verifier()
    print(f"[verify] model loaded in {time.time() - t_load:.0f}s", flush=True)
    t0, done = time.time(), 0
    for vi, vid in enumerate(todo):
        refs = table[vid]
        picks = {k: frames_to_check(r, cfg["frame_quantiles"]) for k, r in refs.items()}
        wanted = {int(refs[k].frame_idx[p]) for k, ps in picks.items() for p in ps}
        frames = read_det_frames(paths[vid], wanted)
        h, w = det_frame_size(split, vid)
        rec = {"video_id": vid, "model": MODEL_ID, "revision": MODEL_REVISION, "prompt": PROMPT,
               "crop_side": cfg["crop_side"], "crop_margin": cfg["crop_margin"], "tracks": {}}
        for k, r in refs.items():
            per = []
            for p in picks[k]:
                fi = int(r.frame_idx[p])
                fr = frames.get(fi)
                if fr is None or fr.shape[:2] != (h, w):
                    per.append({"frame_idx": fi, "error": "frame_missing_or_size_mismatch"})
                    continue
                full, crop = render(fr, r.box[p], cfg["crop_side"], cfg["crop_margin"])
                py, mass = ver.p_yes(full, crop, r.query, r.context)
                per.append({"frame_idx": fi, "box": [round(float(x), 1) for x in r.box[p]],
                            "det_score": round(float(r.score[p]), 4), "p_yes": py, "yes_no_mass": mass})
            ok = [x["p_yes"] for x in per if "p_yes" in x]
            rec["tracks"][k] = {"query": r.query, "spatial": r.spatial, "instance": r.instance, "role": r.role,
                                "context": r.context,
                                "n_track_frames": int(len(r.frame_idx)), "frames": per,
                                "mean_p_yes": float(np.mean(ok)) if ok else None}
            done += 1
        ver.torch.cuda.empty_cache()
        (out_dir / f"{vid}.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
        if (vi + 1) % 10 == 0 or vi + 1 == len(todo):
            el = time.time() - t0
            print(f"[verify] {vi + 1}/{len(todo)} videos, {done}/{n_tracks} tracks, {el / 60:.1f} min, "
                  f"eta {el / max(done, 1) * (n_tracks - done) / 60:.1f} min", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["val", "test"], required=True)
    ap.add_argument("--limit-videos", type=int, default=None, help="probe: only N pending videos spread over the list")
    args = ap.parse_args()
    run(args.split, args.limit_videos)
    return 0


if __name__ == "__main__":
    sys.exit(main())
