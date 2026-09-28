"""Rebuild the object tracks that geometry_v1 actually used for each question.

geometry_v1 keeps only track summaries in runs/geometry_v1/<split>/items/<qid>.json.
The tracks themselves are deterministic functions of the cached detections, so we
rebuild them with the very same code (VideoContext.track) and check them against
those summaries (n_frames, t_range). Nothing here changes a geometry prediction.

Which tracks a question "used" follows solver.solve():
  * targets: every target track (the two instances for "the two X" distances);
    none for camera-distance questions answered straight from depth_info.
  * priors: every prior that produced a pixel scale, when the method used that scale
    (2d_scale, 2d_scale_in_3d); the first such prior when it calibrated the focal
    length (3d_calibrated_focal, or 3d_scene_depth with focal_source == prior_depth);
    none for 3d_default_focal. Gravity priors are skipped: any falling object serves,
    so there is no object identity to check.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from quantiphy.data import ROOT

from methods.geometry import detect as D
from methods.geometry.run import build_plan
from methods.geometry.solver import VideoContext

GEO_RUNS = ROOT / "runs" / "geometry_v1"

PRIOR_SCALE_METHODS = {"2d_scale", "2d_scale_in_3d"}


@dataclass
class TrackRef:
    video_id: str
    query: str
    spatial: str | None
    instance: int
    role: str  # "target" | "prior"
    context: str  # the question (target) or prior text that names this object
    t: np.ndarray = field(repr=False)  # [n] seconds (frame index / table fps)
    frame_idx: np.ndarray = field(repr=False)  # [n] video frame indices of the track
    box: np.ndarray = field(repr=False)  # [n, 4] xyxy in detection-frame pixels
    score: np.ndarray = field(repr=False)

    @property
    def key(self) -> str:
        return f"{self.query}|{self.spatial or ''}|{self.instance}"


def used_prior_indices(rec: dict) -> list[int]:
    """Indices into rec["prior_specs"] of the priors whose track the answer depends on."""
    contributing = [i for i, pi in enumerate(rec.get("prior_tracks") or []) if "scale_m_per_px" in pi]
    method = rec.get("method")
    if method in PRIOR_SCALE_METHODS:
        return contributing
    if method == "3d_calibrated_focal" or (method == "3d_scene_depth" and rec.get("focal_source") == "prior_depth"):
        return contributing[:1]
    return []


def target_objects(qspec: dict) -> list[tuple[dict, int]]:
    if qspec.get("kind") == "camera_dist":
        return []
    objs = qspec.get("objects") or []
    if qspec.get("kind") == "distance" and len(objs) == 1:
        return [(objs[0], 0), (objs[0], 1)]
    return [(o, 0) for o in objs[:2]]


def _to_ref(ctx: VideoContext, vid: str, obj: dict, instance: int, role: str, context: str) -> TrackRef | None:
    tr, _ = ctx.track(obj, instance=instance)
    if tr is None:
        return None
    return TrackRef(vid, obj["query"], obj.get("spatial"), instance, role, context,
                    tr["t"], np.rint(tr["t"] * ctx.fps).astype(int), tr["box"], tr["score"])


def load_items(split: str) -> pd.DataFrame:
    return pd.read_csv(GEO_RUNS / split / "items.csv")


def used_tracks(df: pd.DataFrame, split: str, check: bool = True) -> dict[int, list[TrackRef]]:
    """qid -> tracks used by geometry_v1's answer (only for questions it did not fall back on).

    `df` is the split's question table (answers dropped). With check=True every rebuilt
    track is compared with the summary geometry_v1 stored for that question."""
    qspecs, pspecs, _, _ = build_plan(df)
    items_dir = GEO_RUNS / split / "items"
    det_dir = GEO_RUNS / split / "det"
    out: dict[int, list[TrackRef]] = {}
    ctxs: dict[str, VideoContext] = {}
    for r in df.itertuples():
        rec = json.loads((items_dir / f"{r.qid}.json").read_text(encoding="utf-8"))
        if rec["used_fallback"]:
            continue
        if r.video_id not in ctxs:
            meta, dets = D.load_video_dets(det_dir / f"{r.video_id}.npz")
            ctxs[r.video_id] = VideoContext(meta, dets, r.fps)
        ctx = ctxs[r.video_id]
        refs: list[TrackRef] = []
        for k, (obj, inst) in enumerate(target_objects(qspecs[r.qid])):
            ref = _to_ref(ctx, r.video_id, obj, inst, "target", r.question)
            if check:
                _check(ref, rec["target_tracks"][k], r.qid)
            refs.append(ref)
        for i in used_prior_indices(rec):
            p = pspecs[r.qid][i]
            if p.get("gravity"):
                continue
            ref = _to_ref(ctx, r.video_id, p["objects"][0], 0, "prior", p["text"])
            if check:
                _check(ref, rec["prior_tracks"][i], r.qid)
            refs.append(ref)
        out[int(r.qid)] = refs
    return out


def _check(ref: TrackRef | None, info: dict, qid) -> None:
    if ref is None:
        raise AssertionError(f"qid {qid}: geometry_v1 used a track that could not be rebuilt")
    n = info.get("n_frames")
    if n is not None and n != len(ref.frame_idx):
        raise AssertionError(f"qid {qid}: rebuilt track has {len(ref.frame_idx)} frames, geometry_v1 had {n}")
    t_range = info.get("t_range")
    if t_range is not None and [round(float(ref.t[0]), 3), round(float(ref.t[-1]), 3)] != list(t_range):
        raise AssertionError(f"qid {qid}: rebuilt track spans {ref.t[0]:.3f}-{ref.t[-1]:.3f}s, geometry_v1 had {t_range}")


def frames_to_check(ref: TrackRef, quantiles: list[float]) -> list[int]:
    """Positions (into the track arrays) of the frames shown to the verifier."""
    n = len(ref.frame_idx)
    pos = sorted({int(round(q * (n - 1))) for q in quantiles})
    return pos


def track_table(tracks: dict[int, list[TrackRef]]) -> dict[str, dict[str, TrackRef]]:
    """video_id -> track key -> TrackRef (each distinct track is verified once)."""
    by_video: dict[str, dict[str, TrackRef]] = {}
    for refs in tracks.values():
        for ref in refs:
            by_video.setdefault(ref.video_id, {}).setdefault(ref.key, ref)
    return by_video


def det_frame_size(split: str, video_id: str) -> tuple[int, int]:
    meta = json.loads((GEO_RUNS / split / "det" / f"{video_id}.json").read_text(encoding="utf-8"))
    return int(meta["height"]), int(meta["width"])


def rel(p: Path) -> str:
    return p.relative_to(ROOT).as_posix()
