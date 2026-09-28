"""Stage 1 (GPU): open-vocabulary detection on every (strided) frame with OWLv2.

For each video we run OWLv2 once per sampled frame with *all* text queries needed
by that video's questions (target objects, prior objects, free-fall candidates),
and keep the top-K boxes per query per frame after per-query NMS. Results are
cached as .npz under runs/, so stage 2 (measurement) is pure CPU and fast.

Boxes are stored as (x1, y1, x2, y2) pixels of the decoded frames, which are
downscaled so the long side is <= 960 (meta: width/height vs orig_width/orig_height).
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import av
import numpy as np
import torch
import torch.nn.functional as F

MODEL_ID = "google/owlv2-base-patch16-ensemble"  # Apache-2.0, not gated
MODEL_REVISION = "cfd3195ba4ea9592eec887ded089f4c08eff231d"  # the only snapshot ever used
MAX_FRAMES = 96  # frames sampled per video (uniform stride)
TOP_K = 8
NMS_IOU = 0.5
BATCH = 8

_CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
_CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)


def read_frames(path: str | Path, max_frames: int = MAX_FRAMES, max_side: int = 960):
    """Decode a uniform subset of frames, downscaled so the long side is <= max_side
    (OWLv2 resizes to 960 anyway; this keeps 4K videos cheap in memory).

    Returns (frame_indices, rgb frames, (H, W) of returned frames, n_total, (H0, W0) original).
    """
    with av.open(str(path)) as c:
        n_est = sum(1 for pkt in c.demux(c.streams.video[0]) if pkt.size > 0)
    stride = max(1, math.ceil(max(n_est, 1) / max_frames))
    idx, imgs = [], []
    n = 0
    h0 = w0 = 0
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        s.thread_type = "AUTO"
        for fr in c.decode(s):
            if n % stride == 0:
                h0, w0 = fr.height, fr.width
                sc = min(1.0, max_side / max(h0, w0))
                if sc < 1.0:
                    fr = fr.reformat(width=int(round(w0 * sc)), height=int(round(h0 * sc)))
                imgs.append(fr.to_ndarray(format="rgb24"))
                idx.append(n)
            n += 1
    if not imgs:
        return np.zeros(0, int), [], (0, 0), 0, (0, 0)
    h, w = imgs[0].shape[:2]
    return np.array(idx), imgs, (h, w), n, (h0, w0)


class OwlDetector:
    def __init__(self, device: str = "cuda"):
        from transformers import Owlv2ForObjectDetection, Owlv2Processor

        self.device = device
        self.processor = Owlv2Processor.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
        self.model = Owlv2ForObjectDetection.from_pretrained(MODEL_ID, revision=MODEL_REVISION, dtype=torch.float16).to(device).eval()
        self.size = self.processor.image_processor.size["height"]  # 960

    def _prep(self, imgs: list[np.ndarray]) -> torch.Tensor:
        """Same as Owlv2ImageProcessor: pad to square (bottom/right, value 0.5), resize, CLIP-normalise."""
        x = torch.from_numpy(np.stack(imgs)).to(self.device).permute(0, 3, 1, 2).float() / 255.0
        _, _, h, w = x.shape
        s = max(h, w)
        x = F.pad(x, (0, s - w, 0, s - h), value=0.5)
        x = F.interpolate(x, size=(self.size, self.size), mode="bilinear", align_corners=False, antialias=True)
        x = (x - _CLIP_MEAN.to(x.device)) / _CLIP_STD.to(x.device)
        return x.half()

    @torch.inference_mode()
    def detect(self, imgs: list[np.ndarray], queries: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """-> boxes [F, Q, K, 4] (xyxy, pixels of the given frames), scores [F, Q, K]."""
        texts = [f"a photo of a {q}" for q in queries]
        tok = self.processor.tokenizer(texts, padding="max_length", max_length=16, truncation=True, return_tensors="pt")
        input_ids = tok["input_ids"].to(self.device)
        attn = tok["attention_mask"].to(self.device)
        h, w = imgs[0].shape[:2]
        s = max(h, w)
        nq = len(queries)
        all_boxes = np.zeros((len(imgs), nq, TOP_K, 4), np.float32)
        all_scores = np.zeros((len(imgs), nq, TOP_K), np.float32)
        for b0 in range(0, len(imgs), BATCH):
            chunk = imgs[b0 : b0 + BATCH]
            pv = self._prep(chunk)
            bs = pv.shape[0]
            out = self.model(
                input_ids=input_ids.repeat(bs, 1),
                attention_mask=attn.repeat(bs, 1),
                pixel_values=pv,
            )
            logits = out.logits.float()  # [B, P, Q]
            boxes = out.pred_boxes.float()  # [B, P, 4] cxcywh in [0,1] of padded square
            scores = torch.sigmoid(logits)
            cx, cy, bw, bh = boxes.unbind(-1)
            xyxy = torch.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], -1) * s
            xyxy[..., 0::2] = xyxy[..., 0::2].clamp(0, w)
            xyxy[..., 1::2] = xyxy[..., 1::2].clamp(0, h)
            for i in range(bs):
                for qi in range(nq):
                    sc = scores[i, :, qi]
                    top = torch.topk(sc, k=min(200, sc.numel())).indices
                    keep = _nms(xyxy[i, top], sc[top], NMS_IOU)[:TOP_K]
                    sel = top[keep]
                    k = len(sel)
                    all_boxes[b0 + i, qi, :k] = xyxy[i, sel].cpu().numpy()
                    all_scores[b0 + i, qi, :k] = sc[sel].cpu().numpy()
        return all_boxes, all_scores


def _nms(boxes: torch.Tensor, scores: torch.Tensor, iou: float) -> torch.Tensor:
    from torchvision.ops import nms

    return nms(boxes, scores, iou)


def run_video(det: OwlDetector, video_path: str, queries: list[str], out_path: Path) -> dict:
    t0 = time.time()
    idx, imgs, (h, w), n_total, (h0, w0) = read_frames(video_path)
    t_dec = time.time() - t0
    if len(imgs) == 0:
        meta = {"ok": False, "reason": "video_decode_failed"}
        out_path.with_suffix(".json").write_text(json.dumps(meta), encoding="utf-8")
        return meta
    boxes, scores = det.detect(imgs, queries)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, frame_idx=idx, boxes=boxes, scores=scores)
    meta = {
        "ok": True,
        "queries": queries,
        "height": h,
        "width": w,
        "orig_height": h0,
        "orig_width": w0,
        "n_sampled": int(len(idx)),
        "last_frame_idx": int(idx[-1]),
        "stride": int(idx[1] - idx[0]) if len(idx) > 1 else 1,
        "decode_s": round(t_dec, 2),
        "detect_s": round(time.time() - t0 - t_dec, 2),
        "n_total_frames": int(n_total),
    }
    out_path.with_suffix(".json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return meta


def load_video_dets(npz_path: Path) -> tuple[dict, dict | None]:
    meta = json.loads(npz_path.with_suffix(".json").read_text(encoding="utf-8"))
    if not meta.get("ok"):
        return meta, None
    z = np.load(npz_path)
    return meta, {"frame_idx": z["frame_idx"], "boxes": z["boxes"], "scores": z["scores"]}
