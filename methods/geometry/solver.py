"""Per-question solver: parsed question + parsed prior + cached detections -> number.

Decision order for the metric scale
-----------------------------------
2D questions: scale s [m/px] = prior value (SI) / the same quantity measured in pixels
on the prior object's track. Answer = target quantity in pixels * s.

3D questions (depth_info present):
  (a) prior object has depth -> calibrate the focal length f so that the prior quantity
      measured on the 3D track equals the prior value (rejected if f implies a field of
      view outside ~11-118 deg); targets with depth are then measured in 3D camera
      coordinates (lateral motion from pixels, radial motion from depth; a single depth
      anchor is extended over time with the apparent-size "looming" cue).
  (b) target has depth but f could not be calibrated -> assumed focal length
      (default_focal, ~64 deg horizontal FOV) with the target's own depth.
  (c) target has no depth and the prior is a size -> the prior's 2D pixel scale s.
  (d) target has no depth, kinematic prior -> f (calibrated or assumed) and the median
      annotated depth of the scene.
Every question that cannot be answered this way gets a label-free fallback (see
`fallback_value`) and a failure reason.
"""

from __future__ import annotations

import math

import numpy as np

from . import measure as M
from .parse import KIND_DIM, si_to_unit, unit_to_si

PLAUSIBLE_RATIO = 1e3  # predictions >1000x away from the reference value are rejected
FOCAL_RANGE = (0.3, 5.0)  # calibrated focal length must lie in [0.3, 5] x max(W, H) (FOV ~11-118 deg)


# ---------------------------------------------------------------------------
# label-free reference values (used by the fallback and the plausibility guard)
# ---------------------------------------------------------------------------


def reference_constants(test_df) -> dict:
    """Median |prior| per dimension over the *test* priors (text only, no answers)."""
    from .parse import parse_prior

    vals = {"L": [], "V": [], "A": []}
    for p in test_df["prior"]:
        for x in parse_prior(p):
            dim = KIND_DIM[x["kind"]]
            f, _ = unit_to_si(x["unit"], dim)
            vals[dim].append(abs(x["value"]) * f)
    return {k: float(np.median(v)) for k, v in vals.items()}


def fallback_value(qspec: dict, priors: list[dict], ref: dict) -> tuple[float, str]:
    """Same-dimension prior value if there is one, else the dimension's reference constant (SI)."""
    dim = KIND_DIM.get(qspec.get("kind"), "L")
    for p in priors:
        if KIND_DIM[p["kind"]] == dim:
            f, _ = unit_to_si(p["unit"], dim)
            return abs(p["value"]) * f, "prior_same_dimension"
    return ref[dim], "reference_constant"


# ---------------------------------------------------------------------------
# video context
# ---------------------------------------------------------------------------


class VideoContext:
    def __init__(self, meta: dict, dets: dict | None, fps: float):
        self.meta = meta
        self.dets = dets
        self.fps = float(fps)
        self.w = meta.get("width", 0)
        self.h = meta.get("height", 0)
        self.qindex = {q: i for i, q in enumerate(meta.get("queries", []))}
        self.t = dets["frame_idx"] / self.fps if dets is not None else None
        self._cache: dict = {}

    def track(self, obj: dict, instance: int = 0):
        key = (obj["query"], obj.get("spatial"), instance)
        if key in self._cache:
            return self._cache[key]
        if self.dets is None or obj["query"] not in self.qindex:
            res = (None, {"reason": "not_detected"})
        else:
            qi = self.qindex[obj["query"]]
            boxes, scores, n_sup = self._exclusive_scores(qi)
            res = M.build_track(
                boxes, scores, self.t,
                spatial=obj.get("spatial"), w=self.w, h=self.h, instance=instance,
            )
            res[1]["n_boxes_suppressed_by_other_queries"] = n_sup
        self._cache[key] = res
        return res

    def _exclusive_scores(self, qi: int, iou_thr: float = 0.7, min_keep_frac: float = 0.3):
        """Suppress boxes of query qi that another query of the same video claims with a
        higher score (e.g. 'soccer ball' firing on the yoga ball when 'yoga ball' is also
        asked). Not applied if it would leave the query without confident boxes in most
        frames (then the two names probably refer to the same object)."""
        boxes, scores = self.dets["boxes"], self.dets["scores"]
        s = scores[:, qi].copy()
        nq = boxes.shape[1]
        b = boxes[:, qi]  # [F,K,4]
        if nq < 2:
            return b, s, 0
        sup = np.zeros_like(s, dtype=bool)
        for qj in range(nq):
            if qj == qi:
                continue
            bj, sj = boxes[:, qj], scores[:, qj]
            x1 = np.maximum(b[:, :, None, 0], bj[:, None, :, 0])
            y1 = np.maximum(b[:, :, None, 1], bj[:, None, :, 1])
            x2 = np.minimum(b[:, :, None, 2], bj[:, None, :, 2])
            y2 = np.minimum(b[:, :, None, 3], bj[:, None, :, 3])
            inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
            iou = inter / (M._area(b)[:, :, None] + M._area(bj)[:, None, :] - inter + 1e-9)
            claimed = (iou > iou_thr) & (sj[:, None, :] > s[:, :, None]) & (sj[:, None, :] >= M.ABS_MIN_SCORE)
            sup |= claimed.any(-1)
        if not sup.any():
            return b, s, 0
        peak = np.quantile(s[:, 0], 0.9) if len(s) else 0.0
        thr = max(M.ABS_MIN_SCORE, M.REL_MIN_SCORE * peak)
        s2 = np.where(sup, 0.0, s)
        if np.mean(s2.max(1) >= thr) < min_keep_frac:
            return b, s, 0
        # keep candidates sorted by score (build_track reads column 0 as the frame's best)
        order = np.argsort(-s2, axis=1)
        s2 = np.take_along_axis(s2, order, 1)
        b2 = np.take_along_axis(b, order[:, :, None], 1)
        return b2, s2, int(sup.sum())


# ---------------------------------------------------------------------------
# quantity measurement on a "view" of a track
# ---------------------------------------------------------------------------


class View:
    """Positions and per-frame size factor of a track in some unit system."""

    def __init__(self, track: dict, pos: np.ndarray, size_factor: np.ndarray):
        self.t = track["t"]
        self.box = track["box"]
        self.pos = pos
        self.size_factor = size_factor


def pixel_view(tr: dict) -> View:
    return View(tr, M._centers(tr["box"]), np.ones(len(tr["t"])))


def metric3d_view(tr: dict, d: np.ndarray, f: float, w: int, h: int) -> View:
    return View(tr, M.positions_3d(tr, d, f, w, h), d / f)


def _resolve_t(t, views: list[View]):
    if t == "start":
        return max(v.t[0] for v in views)
    if t == "end":
        return min(v.t[-1] for v in views)
    return t


def quantity(spec: dict, views: list[View], flags: list[str]) -> float | None:
    kind = spec["kind"]
    times = spec.get("times", {})
    v = views[0]
    t0 = times.get("t")
    t1, t2 = times.get("t1"), times.get("t2")
    for tt in [x for x in (t0, t1, t2) if isinstance(x, (int, float))]:
        lo = min(w.t[0] for w in views)
        hi = max(w.t[-1] for w in views)
        if tt < lo - 0.25 or tt > hi + 0.25:
            flags.append("time_outside_track")
    if kind == "size":
        s = M.size_px_series(v.box, spec.get("dim") or "length") * v.size_factor
        return float(np.median(s))
    if kind == "speed":
        if isinstance(t0, (int, float)) and not spec.get("average"):
            vel = M.velocity_at(v.t, v.pos, t0)
            return None if vel is None else float(np.linalg.norm(vel))
        return M.average_speed(v.t, v.pos, t1, t2)
    if kind == "accel":
        if isinstance(t0, (int, float)):
            a = M.accel_at(v.t, v.pos, t0)
            return None if a is None else float(np.linalg.norm(a))
        return M.window_accel(v.t, v.pos, t1, t2)
    if kind == "displacement":
        a = t1 if t1 is not None else v.t[0]
        b = t2 if t2 is not None else v.t[-1]
        pa, pb = M.position_at(v.t, v.pos, a), M.position_at(v.t, v.pos, b)
        if pa is None or pb is None:
            return None
        return float(np.linalg.norm(pb - pa))
    if kind == "path":
        return M.path_length(v.t, v.pos, t1, t2)
    if kind == "orbit":
        ext = float(np.max(np.ptp(v.pos[:, :2], axis=0)))
        return ext / 2 if spec.get("dim") == "radius" else ext
    if kind == "distance":
        if len(views) < 2:
            return None
        a, b = views
        tq = _resolve_t(t0, views) if t0 is not None else None
        if tq is None:
            common = [g for g in a.t if b.t[0] <= g <= b.t[-1]]
            if not common:
                return None
            ds = []
            for g in common:
                pa, pb = M.position_at(a.t, a.pos, g), M.position_at(b.t, b.pos, g)
                if pa is not None and pb is not None:
                    ds.append(np.linalg.norm(pa - pb))
            return float(np.median(ds)) if ds else None
        pa, pb = M.position_at(a.t, a.pos, tq), M.position_at(b.t, b.pos, tq)
        if pa is None or pb is None:
            return None
        return float(np.linalg.norm(pa - pb))
    return None


# ---------------------------------------------------------------------------
# prior -> scale
# ---------------------------------------------------------------------------


def _prior_tracks(ctx: VideoContext, p: dict, gravity_candidates: list[dict]):
    """Return (track, obj, info) for the prior object; gravity picks the best faller."""
    if p.get("gravity"):
        best = None
        tried = []
        for obj in gravity_candidates:
            tr, info = ctx.track(obj)
            if tr is None or len(tr["t"]) < 7:
                tried.append({"query": obj["query"], "reason": info.get("reason", "short_track")})
                continue
            c = M._centers(tr["box"])
            hbox = np.median(tr["box"][:, 3] - tr["box"][:, 1])
            if np.ptp(c[:, 1]) < 0.5 * hbox:
                tried.append({"query": obj["query"], "reason": "no_vertical_motion"})
                continue
            ay = M.gravity_accel_px(tr["t"], c)
            tried.append({"query": obj["query"], "ay_px": ay, "mean_score": float(tr["score"].mean())})
            if ay is None:
                continue
            key = float(tr["score"].mean())
            if best is None or key > best[0]:
                best = (key, tr, obj, info, ay)
        if best is None:
            return None, None, {"reason": "gravity_no_falling_object", "tried": tried}
        _, tr, obj, info, ay = best
        return tr, obj, {**info, "gravity_ay_px": ay, "tried": tried}
    if p.get("unsupported"):
        return None, None, {"reason": "prior_unsupported:" + p["unsupported"]}
    if not p.get("objects"):
        return None, None, {"reason": "prior_object_unparsed"}
    obj = p["objects"][0]
    tr, info = ctx.track(obj)
    return tr, obj, info


def _prior_px(p: dict, tr: dict, info: dict, flags: list[str]) -> float | None:
    if p.get("gravity"):
        return info.get("gravity_ay_px")
    spec = {"kind": p["kind"], "dim": p.get("dim"), "times": p.get("times", {}), "average": p.get("average")}
    if p["kind"] in ("speed", "accel") and len(tr["t"]) < 5:
        return None
    return quantity(spec, [pixel_view(tr)], flags)


def _prior_si(p: dict) -> tuple[float, bool]:
    dim = KIND_DIM[p["kind"]]
    f, certain = unit_to_si(p["unit"], dim)
    return abs(p["value"]) * f, certain


def _solve_focal(p: dict, tr: dict, d: np.ndarray, w: int, h: int, target_si: float, flags) -> float | None:
    """Find f such that the prior quantity measured on the 3D track equals the prior."""
    spec = {"kind": p["kind"], "dim": p.get("dim"), "times": p.get("times", {}), "average": p.get("average")}
    if p.get("gravity"):
        return None

    def q(f):
        return quantity(spec, [metric3d_view(tr, d, f, w, h)], [])

    fs = np.geomspace(0.05 * max(w, h), 50 * max(w, h), 60)
    vals = [q(f) for f in fs]
    if any(v is None for v in vals):
        return None
    vals = np.array(vals)
    # quantity decreases with f; find the crossing
    diff = vals - target_si
    idx = np.where(np.sign(diff[:-1]) != np.sign(diff[1:]))[0]
    if len(idx) == 0:
        return None
    i = idx[0]
    lo, hi = fs[i], fs[i + 1]
    for _ in range(40):
        mid = math.sqrt(lo * hi)
        if (q(mid) - target_si) * (q(lo) - target_si) <= 0:
            hi = mid
        else:
            lo = mid
    return math.sqrt(lo * hi)


# ---------------------------------------------------------------------------
# main entry
# ---------------------------------------------------------------------------


def solve(row, qspec: dict, priors: list[dict], depth: list[dict], ctx: VideoContext,
          gravity_candidates: list[dict], ref: dict) -> dict:
    rec: dict = {"qid": int(row.qid), "category": row.category, "question_spec": qspec, "prior_specs": priors,
                 "flags": [], "reason": None}
    flags = rec["flags"]
    target_unit = row.target_unit
    tdim = KIND_DIM.get(qspec.get("kind"), "L")
    _, unit_ok = unit_to_si(target_unit, tdim)
    if not unit_ok:
        flags.append("target_unit_uncertain")
    fb_si, fb_kind = fallback_value(qspec, priors, ref)
    rec["fallback_si"], rec["fallback_kind"] = fb_si, fb_kind

    def finish(value_si: float | None, reason: str | None, method: str):
        if value_si is not None and (not np.isfinite(value_si) or value_si <= 0):
            value_si, reason = None, reason or "measurement_degenerate"
        if value_si is not None:
            r = value_si / fb_si if fb_si > 0 else 1.0
            if r > PLAUSIBLE_RATIO or r < 1 / PLAUSIBLE_RATIO:
                rec["rejected_si"] = value_si
                value_si, reason = None, "implausible_value"
        used_fb = value_si is None
        v = fb_si if used_fb else value_si
        rec.update({
            "method": "fallback" if used_fb else method,
            "used_fallback": used_fb,
            "reason": reason if used_fb else None,
            "value_si": v,
            "prediction": si_to_unit(v, target_unit, tdim),
        })
        return rec

    if not qspec.get("ok", True):
        return finish(None, "question_parse_failed:" + str(qspec.get("reason")), "")
    if not priors:
        return finish(None, "prior_parse_failed", "")
    if ctx.dets is None:
        return finish(None, "video_decode_failed", "")

    # camera distance straight from depth_info
    if qspec["kind"] == "camera_dist":
        ents = M.object_depth(qspec["objects"][0]["query"], depth)
        if not ents:
            return finish(None, "depth_missing_for_target", "")
        t0 = qspec["times"].get("t")
        tq = np.array([t0 if isinstance(t0, (int, float)) else 0.0])
        return finish(float(M.depth_series(ents, tq)[0]), None, "depth_info")

    # ---- target tracks ----
    tracks, tinfo = [], []
    objs = qspec["objects"]
    if qspec["kind"] == "distance" and len(objs) == 1:
        for inst in (0, 1):
            tr, info = ctx.track(objs[0], instance=inst)
            tracks.append(tr)
            tinfo.append(info)
        objs = [objs[0], objs[0]]
    else:
        for o in objs[:2]:
            tr, info = ctx.track(o)
            tracks.append(tr)
            tinfo.append(info)
    rec["target_tracks"] = tinfo
    if any(tr is None for tr in tracks):
        return finish(None, "target_not_detected", "")
    needs_motion = qspec["kind"] in ("speed", "accel", "displacement", "path", "orbit")
    if needs_motion and any(len(tr["t"]) < 5 for tr in tracks):
        return finish(None, "target_track_too_short", "")

    # ---- prior -> 2D scale ----
    s2d, pinfo_all = [], []
    prior_used = None
    for p in priors:
        ptr, pobj, pinfo = _prior_tracks(ctx, p, gravity_candidates)
        pinfo_all.append(pinfo)
        if ptr is None:
            continue
        pf: list[str] = []
        px = _prior_px(p, ptr, pinfo, pf)
        psi, pcertain = _prior_si(p)
        if not pcertain:
            flags.append("prior_unit_uncertain")
        if px is None or px <= 1e-6:
            pinfo["reason"] = "prior_motion_or_size_degenerate"
            continue
        s2d.append(psi / px)
        pinfo.update({"px_quantity": px, "si": psi, "scale_m_per_px": psi / px, "flags": pf})
        if prior_used is None:
            prior_used = (p, ptr, pobj, pinfo)
    rec["prior_tracks"] = pinfo_all
    scale2d = float(np.median(s2d)) if s2d else None
    rec["scale_2d_m_per_px"] = scale2d

    is3d = row.category in ("3S", "3D") and bool(depth)
    rec["mode"] = "3d" if is3d else "2d"

    if not is3d:
        if scale2d is None:
            reason = "prior_not_detected"
            if pinfo_all and all(pi.get("reason") == "gravity_no_falling_object" for pi in pinfo_all):
                reason = "gravity_no_falling_object"
            elif pinfo_all and any(pi.get("reason") == "prior_motion_or_size_degenerate" for pi in pinfo_all):
                reason = "prior_measurement_degenerate"
            elif pinfo_all and all(str(pi.get("reason", "")).startswith("prior_unsupported") for pi in pinfo_all):
                reason = "prior_unsupported"
            return finish(None, reason, "")
        views = [pixel_view(tr) for tr in tracks]
        qpx = quantity(qspec, views, flags)
        rec["target_px_quantity"] = qpx
        if qpx is None:
            return finish(None, "target_measurement_failed", "")
        return finish(qpx * scale2d, None, "2d_scale")

    # ---- 3D ----
    w, h = ctx.w, ctx.h
    f = None
    if prior_used is not None and not prior_used[0].get("gravity"):
        p, ptr, pobj, pinfo = prior_used
        pd_ents = M.object_depth(pobj["query"], depth)
        if pd_ents:
            d = M.depth_series(pd_ents, ptr["t"], ptr["box"])
            psi, _ = _prior_si(p)
            f = _solve_focal(p, ptr, d, w, h, psi, flags)
            rec["focal_calibrated_raw"] = f
            if f is not None and not (FOCAL_RANGE[0] * max(w, h) <= f <= FOCAL_RANGE[1] * max(w, h)):
                flags.append("focal_implausible")
                f = None
            rec["focal_source"] = "prior_depth" if f else None
            if f is None:
                flags.append("focal_calibration_failed")
        else:
            flags.append("prior_depth_missing")
    tdepths = [M.object_depth(o["query"], depth) for o in objs]
    has_td = all(bool(e) for e in tdepths)
    if not has_td:
        flags.append("target_depth_missing")
    size_prior = prior_used is not None and prior_used[0]["kind"] == "size"
    scene_d = float(np.median([e["d"] for e in depth]))
    if f is not None and has_td:
        method = "3d_calibrated_focal"
    elif has_td:
        # the target's own depth with an assumed field of view is bounded in error,
        # whereas a prior measured at another depth is not
        f = M.default_focal(w, h)
        rec["focal_source"] = "default"
        method = "3d_default_focal"
    elif scale2d is not None and size_prior:
        # target depth unknown; a size prior measured in pixels is a reliable local scale
        method = "2d_scale_in_3d"
    else:
        # target depth unknown and the prior is kinematic (noisy in pixels):
        # assume the target sits at the median annotated depth of the scene
        if f is None:
            f = M.default_focal(w, h)
            rec["focal_source"] = "default"
        scale2d = scene_d / f
        rec["scene_median_depth"] = scene_d
        method = "3d_scene_depth"
    rec["focal_px"] = f
    rec["scale_used_m_per_px"] = scale2d if method in ("2d_scale_in_3d", "3d_scene_depth") else None
    if method in ("2d_scale_in_3d", "3d_scene_depth"):
        views = [pixel_view(tr) for tr in tracks]
        qv = quantity(qspec, views, flags)
        rec["target_px_quantity"] = qv
        if qv is None:
            return finish(None, "target_measurement_failed", "")
        return finish(qv * scale2d, None, method)
    views = [metric3d_view(tr, M.depth_series(e, tr["t"], tr["box"]), f, w, h) for tr, e in zip(tracks, tdepths)]
    qv = quantity(qspec, views, flags)
    rec["target_metric_quantity"] = qv
    if qv is None:
        return finish(None, "target_measurement_failed", "")
    return finish(qv, None, method)
