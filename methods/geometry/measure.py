"""Stage 2 (CPU): turn cached detections into tracks, pixel measurements, a
pixel->metre scale (from the prior) and finally the asked quantity.

Conventions
-----------
* A track is {t: [n] seconds, box: [n,4] xyxy pixels, score: [n]}; t = frame_idx / fps
  with the *table* fps (the fps in some mp4 headers disagrees with the dataset table).
* "Positions" are [n, D] arrays: D=2 pixel/metric image-plane coordinates, or D=3
  metric camera coordinates when depth is available.
* All kinematics (speed, acceleration, displacement, path) are computed by the same
  local-polynomial code on positions, so 2D and 3D share one implementation.
"""

from __future__ import annotations

import math

import numpy as np

from .parse import depth_for

# ---------------------------------------------------------------------------
# tracking
# ---------------------------------------------------------------------------

ABS_MIN_SCORE = 0.08  # below this OWLv2 hits are treated as noise
REL_MIN_SCORE = 0.3  # candidate must reach 30% of the query's typical best score
MIN_DETECTED_PEAK = 0.10  # q90 of per-frame best score must exceed this


def _centers(b: np.ndarray) -> np.ndarray:
    return np.stack([(b[..., 0] + b[..., 2]) / 2, (b[..., 1] + b[..., 3]) / 2], -1)


def _area(b: np.ndarray) -> np.ndarray:
    return np.clip(b[..., 2] - b[..., 0], 1e-3, None) * np.clip(b[..., 3] - b[..., 1], 1e-3, None)


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return inter / (_area(a) + _area(b) - inter + 1e-9)


def _spatial_pick(boxes: np.ndarray, scores: np.ndarray, spatial: str, w: int, h: int) -> int:
    c = _centers(boxes)
    if spatial == "left":
        return int(np.argmin(c[:, 0]))
    if spatial == "right":
        return int(np.argmax(c[:, 0]))
    if spatial == "top":
        return int(np.argmin(c[:, 1]))
    if spatial == "bottom":
        return int(np.argmax(c[:, 1]))
    if spatial == "upper_left":
        return int(np.argmin(c[:, 0] + c[:, 1]))
    if spatial == "upper_right":
        return int(np.argmin(-c[:, 0] + c[:, 1]))
    if spatial == "middle":
        return int(np.argmin(np.hypot(c[:, 0] - w / 2, c[:, 1] - h / 2)))
    if spatial == "front":  # closer to camera: lower in the frame / larger
        return int(np.argmax(boxes[:, 3]))
    if spatial == "back":
        return int(np.argmin(boxes[:, 3]))
    return int(np.argmax(scores))


def build_track(boxes: np.ndarray, scores: np.ndarray, t: np.ndarray, *, spatial: str | None, w: int, h: int,
                instance: int = 0) -> tuple[dict | None, dict]:
    """Pick one box per frame for a query.

    boxes [F,K,4], scores [F,K]. With a spatial qualifier the choice per frame is
    geometric among confident candidates; otherwise a Viterbi path trades detector
    score against frame-to-frame jumps. `instance=1` returns the second instance
    (used for "the two X"), chosen as the best candidate not overlapping instance 0.
    """
    best = scores[:, 0]
    peak = float(np.quantile(best, 0.9)) if len(best) else 0.0
    info = {"peak_score": round(peak, 4)}
    if peak < MIN_DETECTED_PEAK:
        info["reason"] = "not_detected"
        return None, info
    thr = max(ABS_MIN_SCORE, REL_MIN_SCORE * peak)
    info["threshold"] = round(thr, 4)

    frames, cand = [], []
    for f in range(len(t)):
        ok = scores[f] >= thr
        if ok.any():
            frames.append(f)
            cand.append(np.where(ok)[0])
    if not frames:
        info["reason"] = "not_detected"
        return None, info

    chosen: list[int] = []
    if spatial is not None or instance > 0:
        for f, ks in zip(frames, cand):
            b, s = boxes[f, ks], scores[f, ks]
            if instance > 0:
                # instance 0 = highest score; instance 1 = best non-overlapping other box
                k0 = int(np.argmax(s))
                others = [i for i in range(len(ks)) if i != k0 and _iou(b[i], b[k0]) < 0.3]
                if not others:
                    chosen.append(-1)
                    continue
                o = max(others, key=lambda i: s[i])
                pair = [k0, o]
                # stable ordering of the pair: left-to-right
                pair.sort(key=lambda i: _centers(b[i])[0])
                chosen.append(int(ks[pair[instance]]))
            else:
                chosen.append(int(ks[_spatial_pick(b, s, spatial, w, h)]))
    else:
        # Viterbi over candidate boxes
        costs_prev = -np.log(np.clip(scores[frames[0], cand[0]], 1e-6, 1))
        back = []
        for i in range(1, len(frames)):
            f0, f1 = frames[i - 1], frames[i]
            b0, b1 = boxes[f0, cand[i - 1]], boxes[f1, cand[i]]
            c0, c1 = _centers(b0), _centers(b1)
            size0 = np.sqrt(_area(b0))
            gap = max(1, f1 - f0)
            d = np.linalg.norm(c1[None, :, :] - c0[:, None, :], axis=-1) / (size0[:, None] + 1.0) / gap
            ar = np.abs(np.log(_area(b1)[None, :] / _area(b0)[:, None]))
            trans = 1.5 * d + 1.0 * ar
            tot = costs_prev[:, None] + trans
            arg = np.argmin(tot, axis=0)
            costs_prev = tot[arg, np.arange(tot.shape[1])] - np.log(np.clip(scores[f1, cand[i]], 1e-6, 1))
            back.append(arg)
        path = [int(np.argmin(costs_prev))]
        for arg in reversed(back):
            path.append(int(arg[path[-1]]))
        path.reverse()
        chosen = [int(cand[i][k]) for i, k in enumerate(path)]

    keep = [(f, k) for f, k in zip(frames, chosen) if k >= 0]
    if not keep:
        info["reason"] = "not_detected"
        return None, info
    fi = np.array([f for f, _ in keep])
    ki = np.array([k for _, k in keep])
    tr = {"t": t[fi], "box": boxes[fi, ki].astype(float), "score": scores[fi, ki].astype(float)}
    tr = _drop_outliers(tr)
    info.update({
        "n_frames": int(len(tr["t"])),
        "t_range": [round(float(tr["t"][0]), 3), round(float(tr["t"][-1]), 3)],
        "median_box_wh": [round(float(np.median(tr["box"][:, 2] - tr["box"][:, 0])), 2),
                          round(float(np.median(tr["box"][:, 3] - tr["box"][:, 1])), 2)],
        "mean_score": round(float(tr["score"].mean()), 4),
    })
    return tr, info


def _drop_outliers(tr: dict) -> dict:
    """Remove frames whose box area is far from the track median (detector glitches)."""
    if len(tr["t"]) < 5:
        return tr
    a = _area(tr["box"])
    la = np.log(a)
    med = np.median(la)
    mad = np.median(np.abs(la - med)) + 1e-6
    ok = np.abs(la - med) < max(4 * 1.4826 * mad, math.log(2.0))
    if ok.sum() >= 3:
        tr = {k: v[ok] for k, v in tr.items()}
    return tr


# ---------------------------------------------------------------------------
# kinematics on positions
# ---------------------------------------------------------------------------


def _local_poly(t: np.ndarray, p: np.ndarray, t0: float, half: float, min_pts: int, deg: int = 2):
    """Fit p(t) ~ poly(t - t0) on points within +-half of t0 (window grows until min_pts).
    Returns (value, first derivative, second derivative) as [D] arrays, or None."""
    if len(t) < max(min_pts, deg + 1):
        return None
    hw = half
    for _ in range(12):
        m = np.abs(t - t0) <= hw + 1e-9
        if m.sum() >= min_pts:
            break
        hw *= 1.5
    if m.sum() < deg + 1:
        return None
    tt = t[m] - t0
    X = np.vander(tt, deg + 1, increasing=True)  # 1, t, t^2
    coef, *_ = np.linalg.lstsq(X, p[m], rcond=None)
    val = coef[0]
    d1 = coef[1] if deg >= 1 else np.zeros_like(val)
    d2 = 2 * coef[2] if deg >= 2 else np.zeros_like(val)
    return val, d1, d2


def position_at(t, p, t0):
    r = _local_poly(t, p, t0, half=0.15, min_pts=3, deg=1)
    return None if r is None else r[0]


def velocity_at(t, p, t0):
    r = _local_poly(t, p, t0, half=0.25, min_pts=5, deg=2)
    return None if r is None else r[1]


def accel_at(t, p, t0):
    r = _local_poly(t, p, t0, half=0.45, min_pts=7, deg=2)
    return None if r is None else r[2]


def _window(t, t1, t2):
    lo = t[0] if t1 is None else t1
    hi = t[-1] if t2 is None else t2
    return lo, hi


def average_speed(t, p, t1=None, t2=None):
    lo, hi = _window(t, t1, t2)
    grid = t[(t >= lo - 1e-9) & (t <= hi + 1e-9)]
    if len(grid) < 2:
        grid = np.linspace(max(lo, t[0]), min(hi, t[-1]), 5)
    vs = [velocity_at(t, p, g) for g in grid]
    vs = [np.linalg.norm(v) for v in vs if v is not None]
    return float(np.mean(vs)) if vs else None


def window_accel(t, p, t1=None, t2=None):
    """|a| from one quadratic fit over [t1, t2] (whole track by default)."""
    lo, hi = _window(t, t1, t2)
    m = (t >= lo - 1e-9) & (t <= hi + 1e-9)
    if m.sum() < 5:
        return None
    tt = t[m] - t[m].mean()
    X = np.vander(tt, 3, increasing=True)
    coef, *_ = np.linalg.lstsq(X, p[m], rcond=None)
    return float(np.linalg.norm(2 * coef[2]))


def gravity_accel_px(t, p):
    """Median local vertical acceleration (image y grows downward). None if not falling."""
    if len(t) < 7:
        return None
    ays = []
    for g in t:
        r = _local_poly(t, p, g, half=0.2, min_pts=6, deg=2)
        if r is not None:
            ays.append(r[2][1])
    if not ays:
        return None
    ay = float(np.median(ays))
    return ay if ay > 0 else None


def path_length(t, p, t1=None, t2=None):
    lo, hi = _window(t, t1, t2)
    grid = t[(t >= lo - 1e-9) & (t <= hi + 1e-9)]
    if len(grid) < 2:
        return None
    pts = [position_at(t, p, g) for g in grid]
    pts = np.array([q for q in pts if q is not None])
    if len(pts) < 2:
        return None
    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))


# ---------------------------------------------------------------------------
# sizes in pixels
# ---------------------------------------------------------------------------


def size_px_series(box: np.ndarray, dim: str) -> np.ndarray:
    w = box[:, 2] - box[:, 0]
    h = box[:, 3] - box[:, 1]
    if dim == "height":
        return h
    if dim == "width":
        return w
    if dim == "length":
        return np.maximum(w, h)
    if dim == "thickness":
        return np.minimum(w, h)
    if dim == "radius":
        return np.sqrt(w * h) / 2
    return np.sqrt(w * h)  # diameter / size


# ---------------------------------------------------------------------------
# depth / camera model
# ---------------------------------------------------------------------------


def _looming_depth(d_anchor: float, t_anchor: float, t: np.ndarray, box: np.ndarray) -> np.ndarray:
    """Depth over time from one anchor + apparent size: d(t) ~ 1 / size_px(t).

    1/size is fitted with a straight line in time (constant radial velocity) with one
    round of outlier rejection, so single-frame box noise does not leak into d(t)."""
    size = np.sqrt(np.clip(box[:, 2] - box[:, 0], 1e-3, None) * np.clip(box[:, 3] - box[:, 1], 1e-3, None))
    inv = 1.0 / size
    A = np.vstack([np.ones_like(t), t]).T
    c, *_ = np.linalg.lstsq(A, inv, rcond=None)
    res = inv - A @ c
    mad = np.median(np.abs(res)) + 1e-12
    ok = np.abs(res) < 3 * 1.4826 * mad
    if ok.sum() >= 5:
        c, *_ = np.linalg.lstsq(A[ok], inv[ok], rcond=None)
    base = c[0] + c[1] * t_anchor
    if base <= 0:
        return np.full(len(t), d_anchor)
    ratio = np.clip((c[0] + c[1] * t) / base, 0.2, 5.0)
    return d_anchor * ratio


def depth_series(entries: list[dict], t: np.ndarray, box: np.ndarray | None = None) -> np.ndarray | None:
    """Object-camera distance at times t from depth_info entries.

    >=2 timed entries: straight line in time through them. Exactly one timed entry and a
    track box: looming (apparent-size) extrapolation around that anchor. Otherwise constant."""
    if not entries:
        return None
    timed1 = [(e["t"], e["d"]) for e in entries if e["t"] is not None]
    if len({a for a, _ in timed1}) == 1 and box is not None and len(t) >= 5:
        return _looming_depth(timed1[0][1], timed1[0][0], t, box)
    timed = sorted([(e["t"], e["d"]) for e in entries if e["t"] is not None])
    static = [e["d"] for e in entries if e["t"] is None]
    if len(timed) >= 2:
        ts = np.array([a for a, _ in timed])
        ds = np.array([b for _, b in timed])
        if np.ptp(ts) < 1e-9:
            return np.full(len(t), float(np.mean(ds)))
        # least-squares line (exact for two points); extrapolates linearly
        A = np.vstack([np.ones_like(ts), ts]).T
        c, *_ = np.linalg.lstsq(A, ds, rcond=None)
        return np.clip(c[0] + c[1] * t, 0.05, None)
    if len(timed) == 1:
        return np.full(len(t), timed[0][1])
    return np.full(len(t), float(np.median(static)))


def positions_3d(track: dict, d: np.ndarray, f: float, w: int, h: int) -> np.ndarray:
    c = _centers(track["box"])
    x = (c[:, 0] - w / 2) / f
    y = (c[:, 1] - h / 2) / f
    norm = np.sqrt(1 + x * x + y * y)
    z = d / norm
    return np.stack([x * z, y * z, z], -1)


def default_focal(w: int, h: int) -> float:
    """Assumed pinhole focal length (pixels) when it cannot be calibrated: ~64 deg horizontal FOV."""
    return 0.8 * max(w, h)


def object_depth(obj_query: str, depth: list[dict]) -> list[dict]:
    return depth_for(obj_query, depth)
