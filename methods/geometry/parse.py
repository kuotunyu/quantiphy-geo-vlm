"""Rule-based parsing of questions, priors and depth strings into structured specs.

Everything here is pure text processing (no model, no labels). The output is a
plain dict so it can be dumped to JSON in the per-question debug record.

Quantity kinds (SI units internally):
    size          meters   (dim = height | width | length | diameter | thickness | radius)
    distance      meters   distance between two objects at a time
    displacement  meters   |p(t2) - p(t1)| of one object
    path          meters   distance travelled along the track
    speed         m/s      at a time, or average over a window
    accel         m/s^2    at a time, or over a window
    camera_dist   meters   object-to-camera distance (answered from depth_info)
"""

from __future__ import annotations

import math
import re
import unicodedata

# ---------------------------------------------------------------------------
# text normalisation
# ---------------------------------------------------------------------------

_TYPO = {
    r"\bvelolicty\b": "velocity",
    r"\bdiasplacement\b": "displacement",
    r"\bdiatance\b": "distance",
    r"\blenth\b": "length",
    r"\bbaskteball\b": "basketball",
    r"\bsoccor\b": "soccer",
    r"\bbycicle\b": "bicycle",
    r"\bconrner\b": "corner",
    r"\bballon\b": "ball on",
    r"\bbalck\b": "black",
    r"\bturing\b": "turning",
    r"\bround about\b": "roundabout",
    r"\bcenteral\b": "central",
    r"\bconveyer\b": "conveyor",
    r"\brom start\b": "from start",
    r"\bball ball\b": "ball",
    r"\bbill board\b": "billboard",
    r"\bpingpong\b": "ping pong",
}


def normalize(text: str) -> str:
    """Lower-case, repair mojibake apostrophes/quotes and a few frequent typos."""
    s = unicodedata.normalize("NFKC", str(text))  # full-width ？（ -> ? (
    s = s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    s = s.replace("�", " ")
    s = s.lower()
    for pat, rep in _TYPO.items():
        s = re.sub(pat, rep, s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ---------------------------------------------------------------------------
# units
# ---------------------------------------------------------------------------

# multiplier from unit prefix to SI (meters)
_LEN = {"m": 1.0, "cm": 0.01, "mm": 0.001, "km": 1000.0}

KIND_DIM = {
    "size": "L", "distance": "L", "displacement": "L", "path": "L", "camera_dist": "L",
    "speed": "V", "accel": "A",
}


def unit_to_si(unit: str | None, dim: str) -> tuple[float, bool]:
    """Factor converting a value in `unit` to SI for dimension `dim` (L, V, A).

    Returns (factor, certain). The length prefix (m/cm/mm/km) decides the factor;
    the time part is trusted to match `dim` (priors like 'acceleration = 9.8m/s'
    are treated as m/s^2). Unknown/missing unit -> (1.0, False).
    """
    if not isinstance(unit, str) or not unit:
        return 1.0, False
    u = unit.lower().replace(" ", "").replace("²", "^2")
    if u in ("meters", "meter", "metres", "metre"):
        u = "m"
    elif u in ("centimeters", "centimeter"):
        u = "cm"
    elif u in ("millimeters", "millimeter"):
        u = "mm"
    elif u in ("kilometers", "kilometer"):
        u = "km"
    if u == "km/h":
        return (1 / 3.6 if dim == "V" else 1.0), dim == "V"
    m = re.match(r"^([ckm]?m)(?:/s(\^2|2)?)?$", u)
    if not m:
        return 1.0, False
    base = _LEN[m.group(1)]
    has_time = "/s" in u
    is_sq = bool(m.group(2))
    certain = (dim == "L" and not has_time) or (dim == "V" and has_time and not is_sq) or (dim == "A" and is_sq)
    return base, certain


def si_to_unit(value_si: float, unit: str | None, dim: str) -> float:
    f, _ = unit_to_si(unit, dim)
    return value_si / f


# ---------------------------------------------------------------------------
# object phrases
# ---------------------------------------------------------------------------

_SPATIAL = [
    (r"\b(?:on|at|in) the upper left(?: corner)?\b|\bupper[- ]left\b", "upper_left"),
    (r"\b(?:on|at|in) the upper right(?: corner)?\b|\bupper[- ]right\b", "upper_right"),
    (r"\b(?:on|at|in) the left(?: side)?\b|\bleftmost\b|\bleft\b", "left"),
    (r"\b(?:on|at|in) the right(?: side)?\b|\brightmost\b|\bright\b", "right"),
    (r"\b(?:in|at) the (?:middle|center|centre)\b|\bmiddle\b|\bcentral\b|\bcenter\b", "middle"),
    (r"\b(?:in|at) the front\b|\bfront\b|\bnearest\b|\bnear\b|\bclosest\b", "front"),
    (r"\b(?:at|in) the back\b|\bfar\b|\bfarthest\b", "back"),
    (r"\bon the top\b|\btop\b|\bupper\b", "top"),
    (r"\bon the bottom\b|\bbottom\b|\blower\b", "bottom"),
]

# Cut an object phrase at time / relational words, but keep short attributes such
# as "in yellow", "in the black shirt", "in the roundabout".
_STOP_TAIL = re.compile(
    r"\s+(?:(?:at\s+(?:time\s+)?|in\s+|to\s+)\d.*"
    r"|(?:at\s+the\s+(?:start|end|beginning)|in the first|between|from|during|when|after|before|within|over|"
    r"of the video|including|excluding|with respect|above|below|relative)\b.*)$"
)

_ARTICLES = re.compile(r"^(?:the|a|an|one|each|this|that)\s+")


def clean_object(phrase: str) -> dict:
    """Split an object phrase into a detector query and a spatial qualifier."""
    p = normalize(phrase)
    p = re.sub(r"\([^)]*\)", " ", p)  # drop parentheticals
    p = re.sub(r"[?.,;:!\"]", " ", p)
    p = re.sub(r"\b(?:itself|model of|the model of)\b", " ", p)
    p = re.sub(r"\bmodel\b", " ", p) if re.search(r"\bmodel\b", p) and len(p.split()) > 2 else p
    p = re.sub(r"\s+", " ", p).strip()
    spatial = None
    for pat, tag in _SPATIAL:
        if re.search(pat, p):
            spatial = tag
            p = re.sub(pat, " ", p)
            break
    p = re.sub(r"\s+", " ", p).strip()
    p = _ARTICLES.sub("", p)
    p = re.sub(r"\b(?:on|in|at|of|the)$", "", p).strip()
    p = re.sub(r"'s$", "", p).strip()
    count = 1
    m = re.match(r"^(?:two|both|2)\s+(.*)$", p)
    if m:
        count = 2
        p = m.group(1)
        p = re.sub(r"(?<=[a-z])s\b", "", p, count=0) if not p.endswith("ss") else p
        # naive singular: 'black road signs' -> 'black road sign'
    p = p.strip()
    return {"query": p, "spatial": spatial, "count": count}


# ---------------------------------------------------------------------------
# times
# ---------------------------------------------------------------------------

_T = r"(\d+(?:\.\d+)?)\s*(?:s|sec|secs|seconds)?"


def parse_times(q: str) -> dict:
    """Extract time constraints from a normalised question/prior string."""
    out: dict = {}
    m = re.search(r"(?:between|from)\s+" + _T + r"\s*(?:and|to|-)\s*" + _T, q)
    if m:
        out["t1"], out["t2"] = float(m.group(1)), float(m.group(2))
        return out
    m = re.search(r"\bin\s+(\d+(?:\.\d+)?)\s*s\s*(?:to|-)\s*(\d+(?:\.\d+)?)\s*s", q)
    if m:
        out["t1"], out["t2"] = float(m.group(1)), float(m.group(2))
        return out
    m = re.search(r"from start to\s+" + _T, q)
    if m:
        out["t1"], out["t2"] = 0.0, float(m.group(1))
        return out
    m = re.search(r"in the first\s+(\d+(?:\.\d+)?)\s*(?:s|sec|seconds)", q)
    if m:
        out["t1"], out["t2"] = 0.0, float(m.group(1))
        return out
    m = re.search(r"\bbefore\s+" + _T, q)
    if m:
        out["t1"], out["t2"] = 0.0, float(m.group(1))
        return out
    m = re.search(r"\bafter\s+" + _T, q)
    if m:
        out["t1"], out["t2"] = float(m.group(1)), None
        return out
    m = re.search(r"\bt\s*=\s*(\d+(?:\.\d+)?)\s*s?\b", q)
    if m:
        out["t"] = float(m.group(1))
        return out
    m = re.search(r"\bat\s+(?:time\s+)?(\d+(?:\.\d+)?)\s*(?:s|sec|seconds)\b", q)
    if m:
        out["t"] = float(m.group(1))
        return out
    if re.search(r"\binitial(?:ly)?\b|\bat the (?:start|beginning)\b", q):
        out["t"] = "start"
    elif re.search(r"\bfinal(?:ly)?\b|\bat the end\b", q):
        out["t"] = "end"
    return out


# ---------------------------------------------------------------------------
# questions
# ---------------------------------------------------------------------------

_UNIT_TAIL = re.compile(
    r",?\s*\bin\s*(?:meters?|metres?|centimeters?|millimeters?|kilometers?|km/h|"
    r"[cmk]?m\s*/\s*s\s*(?:\^\s*2|2)?|[cmk]?m)\b"
)

_DIM_WORDS = [
    (r"\bwingspan\b", "width"),
    (r"\bthickness\b|\bthick\b|\bdepth\b", "thickness"),
    (r"\bradius\b", "radius"),
    (r"\bdiameter\b|\bsize\b", "diameter"),
    (r"\bheight\b|\btall\b|\bhigh\b", "height"),
    (r"\bwidth\b|\bwide\b|\bbreadth\b", "width"),
    (r"\blength\b|\blong\b", "length"),
]


def _strip_question(q: str) -> str:
    q = normalize(q)
    q = _UNIT_TAIL.sub(" ", q)
    q = re.sub(r"^(?:what|how)\s+(?:is|was|are)\s+", "", q)
    q = re.sub(r"^when\s+t\s*=\s*[\d.]+\s*s?\s*,\s*what\s+is\s+", "", q)
    q = re.sub(r"\?", " ", q)
    return re.sub(r"\s+", " ", q).strip()


def _obj_after_of(text: str) -> str | None:
    m = re.search(r"\b(?:of|by)\s+(.*)$", text)
    if not m:
        return None
    return _STOP_TAIL.sub("", m.group(1)).strip() or None


def parse_question(question: str) -> dict:
    """Parse a question into {kind, dim, objects, times, average, ok, reason}."""
    raw = normalize(question)
    times = parse_times(raw)
    q = re.sub(r"\([^)]*\)", " ", _strip_question(question))
    q = re.sub(r"\s+", " ", q).strip()
    spec: dict = {"text": q, "times": times, "average": False, "objects": [], "ok": True, "reason": None}

    # possessive form: "the eagle's wingspan", "the basketball's distance from the camera"
    poss = re.match(r"^(?:the\s+)?((?:[a-z0-9\-]+\s+){0,2}[a-z0-9\-]+)'s\s+(.*)$", q)
    head_obj = None
    body = q
    if poss:
        head_obj, body = poss.group(1), poss.group(2)

    if re.search(r"\baccelerat", body) or re.search(r"\bacc\b", body):
        spec["kind"] = "accel"
    elif re.search(r"\bspeed\b|\bvelocity\b", body) or re.match(r"^average of\b", body):
        spec["kind"] = "speed"
        spec["average"] = bool(re.search(r"\baverage\b|\bmean\b", raw)) or not ("t" in times or "t1" in times)
    elif re.search(r"\borbital (?:diameter|radius)\b|\bdiameter of the orbit\b", body):
        spec["kind"] = "orbit"
        spec["dim"] = "radius" if "radius" in body else "diameter"
    elif re.search(r"distance from the camera|distance to the camera|from the camera", body):
        spec["kind"] = "camera_dist"
    elif re.search(r"\bdisplacement\b|\bmoving distance\b|\bdistance moved\b|\bmoved\b", body):
        spec["kind"] = "displacement"
    elif re.search(r"\bdistance (?:travel+ed|covered)\b|\btotal distance\b|\bhow far\b", body):
        spec["kind"] = "path"
    elif re.search(r"\b(?:distance|spacing|gap)\b", body):
        spec["kind"] = "distance"
    else:
        spec["kind"] = "size"
        spec["dim"] = None
        for pat, dim in _DIM_WORDS:
            if re.search(pat, body):
                spec["dim"] = dim
                break
        if spec["dim"] is None:
            spec["ok"], spec["reason"] = False, "question_quantity_unknown"
            return spec

    # --- objects ---
    if spec["kind"] == "distance":
        m = re.search(r"\bbetween\s+(.*?)\s+and\s+(.*)$", body)
        m_from = re.search(r"\bdistance from\s+(.*?)\s+to\s+(.*)$", body)
        if poss and re.search(r"distance from\s+(.*)$", body):
            other = re.search(r"distance from\s+(.*)$", body).group(1)
            objs = [head_obj, _STOP_TAIL.sub("", other)]
        elif m:
            objs = [m.group(1), _STOP_TAIL.sub("", m.group(2))]
        elif m_from:
            objs = [m_from.group(1), _STOP_TAIL.sub("", m_from.group(2))]
        else:
            m2 = re.search(r"\bbetween\s+(?:the\s+)?(?:two|both)\s+(.*)$", body)
            if m2:
                objs = ["two " + _STOP_TAIL.sub("", m2.group(1))]
            else:
                spec["ok"], spec["reason"] = False, "question_objects_unparsed"
                return spec
        cleaned = [clean_object(o) for o in objs]
        cam = [o for o in cleaned if o["query"] == "camera"]
        if cam and len(cleaned) == 2:
            spec["kind"] = "camera_dist"
            spec["objects"] = [o for o in cleaned if o["query"] != "camera"]
            return spec
        # 'between the two X' -> one query, two instances
        if len(cleaned) == 1:
            spec["objects"] = cleaned
        else:
            m2 = re.search(r"\bbetween\s+(?:the\s+)?(?:two|both)\s+(.*)$", body)
            spec["objects"] = [clean_object("two " + _STOP_TAIL.sub("", m2.group(1)))] if m2 else cleaned
        if any(re.search(r"\b(?:floor|ground|desk|table surface|chair surface|surface|net|camera|court|wall|line|path)\b", o["query"]) for o in spec["objects"]):
            spec["ok"], spec["reason"] = False, "question_reference_unsupported"
        return spec

    obj = head_obj
    if obj is None:
        obj = _obj_after_of(body)
    if obj is None:
        # "how long is the shark", "the average vehicle spacing"
        m = re.match(r"^how\s+(?:long|tall|high|wide)\s+(?:is|was)\s+(.*)$", body)
        obj = m.group(1) if m else None
    if obj is None:
        spec["ok"], spec["reason"] = False, "question_objects_unparsed"
        return spec
    o = clean_object(obj)
    if not o["query"]:
        spec["ok"], spec["reason"] = False, "question_objects_unparsed"
        return spec
    o["query"] = " ".join(o["query"].split()[:6])
    spec["objects"] = [o]
    if spec["kind"] == "size" and o["count"] == 2:
        spec["ok"], spec["reason"] = False, "question_objects_unparsed"
    return spec


# ---------------------------------------------------------------------------
# priors
# ---------------------------------------------------------------------------

_NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"


def _parse_prior_line(line: str) -> dict | None:
    s = normalize(line)
    m = re.search(r"^(.*?)(?:=|~)\s*(" + _NUM + r")\s*([a-z/^²2]*)\s*$", s)
    if not m:
        m = re.search(r"^(.*?)(?:=|~)\s*(" + _NUM + r")\s*([a-z/^²2]*)", s)
        if not m:
            return None
    desc, value, unit = m.group(1).strip(), float(m.group(2)), m.group(3)
    # "t=1.5, ball acceleration = 3.0" : the first '=' belongs to the time
    times = {}
    mt = re.match(r"^t\s*=\s*(" + _NUM + r")\s*s?\s*,\s*(.*)$", desc)
    if mt:
        times["t"] = float(mt.group(1))
        desc = mt.group(2)
    else:
        times = parse_times(desc)
    desc_nt = re.sub(r"\b(?:at|before|after)\s+[\d.]+\s*s\b", " ", desc)
    desc_nt = re.sub(r"\bfrom\s+[\d.]+\s*s?\s+to\s+[\d.]+\s*s\b", " ", desc_nt)
    desc_nt = re.sub(r"[_]", " ", desc_nt)
    desc_nt = re.sub(r"\s+", " ", desc_nt).strip()

    spec: dict = {"text": s, "value": value, "unit": unit, "times": times, "average": False}
    if re.search(r"\bgravity\b", desc_nt) or re.fullmatch(r"acceleration|acc", desc_nt):
        spec["kind"] = "accel"
        spec["gravity"] = True
        spec["objects"] = []
        return spec
    if re.search(r"\baccelerat|\bacc\b", desc_nt):
        spec["kind"] = "accel"
    elif re.search(r"\bspeed\b|\bvelocity\b", desc_nt):
        spec["kind"] = "speed"
        spec["average"] = "t" not in times
    else:
        spec["kind"] = "size"
        spec["dim"] = None
        for pat, dim in _DIM_WORDS + [(r"\bcalibre\b", "length")]:
            if re.search(pat, desc_nt):
                spec["dim"] = dim
                break
        if spec["dim"] is None:
            spec["dim"] = "length"
    spec["gravity"] = False
    if re.search(r"\bcalibre\b|\bcaliber\b|\bgraduation\b|\btick\b", desc_nt):
        # ruler tick spacing: not measurable from a whole-object bounding box
        spec["objects"] = []
        spec["unsupported"] = "ruler_tick_prior"
        return spec
    obj = _obj_after_of(desc_nt)
    if obj is None:
        # "lane width", "billiard ball diameter", "ball acceleration", "walking speed"
        stripped = re.sub(
            r"\b(?:average|total|the|horizontal|vertical|lifting|model|walking)?\s*"
            r"(?:speed|velocity|acceleration|acc|width|length|height|diameter|size|radius|calibre)\b.*$",
            "",
            desc_nt,
        ).strip()
        obj = stripped or None
        if obj is None and re.search(r"walking|pedestrian", desc_nt):
            obj = "person"
        if obj and re.search(r"\bwalking\b", desc_nt) and obj in ("walking",):
            obj = "person"
    if obj and re.search(r"\bwalking\b", desc_nt) and re.fullmatch(r"(?:pedestrian)?", obj or ""):
        obj = obj or "person"
    if obj is None and re.search(r"walking|pedestrian", desc_nt):
        obj = "person"
    if obj is None:
        spec["objects"] = []
        spec["ok"] = False
        return spec
    o = clean_object(re.sub(r"'s$", "", obj.strip()))
    spec["objects"] = [o] if o["query"] else []
    return spec


def parse_prior(prior: str) -> list[dict]:
    """A prior may hold several lines (e.g. two tyre diameters); parse each."""
    if not isinstance(prior, str):
        return []
    out = []
    for line in re.split(r"[\n;]+", prior):
        if not line.strip():
            continue
        p = _parse_prior_line(line)
        if p is not None and not math.isnan(p["value"]):
            out.append(p)
    return out


# ---------------------------------------------------------------------------
# depth
# ---------------------------------------------------------------------------

_DEPTH_RE = re.compile(
    r"(?:t\s*=\s*(" + _NUM + r")\s*s?\s*,\s*)?distance[_ ]?(.*?)\s*=\s*(" + _NUM + r")\s*[ms]?\b"
)


def parse_depth(depth_info) -> list[dict]:
    """-> [{name_tokens, t (or None), d (meters)}]."""
    if not isinstance(depth_info, str):
        return []
    out = []
    for line in depth_info.splitlines():
        s = normalize(line)
        m = _DEPTH_RE.search(s)
        if not m:
            continue
        t = float(m.group(1)) if m.group(1) is not None else None
        name = m.group(2)
        name = re.sub(r"[_]+", " ", name)
        name = re.sub(r"\b(?:camera|and|the|to)\b", " ", name)
        toks = [w for w in re.split(r"[^a-z0-9]+", name) if w]
        out.append({"name": " ".join(toks), "tokens": toks, "t": t, "d": float(m.group(3))})
    return out


_SYN = {
    "human": {"person", "man", "woman", "pedestrian", "people", "boatman", "player", "swimmer", "astronaut"},
    "person": {"human", "man", "woman", "pedestrian", "people", "boatman", "player", "swimmer"},
    "pingpong": {"ping", "pong"},
    "bike": {"bicycle", "bike"},
    "bicycle": {"bike"},
    "notebook": {"note", "book"},
    "trashbin": {"trash", "bin", "can"},
}


def _singular(w: str) -> str:
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w


def depth_match_score(obj_query: str, tokens: list[str]) -> float:
    """Token-overlap score between an object query and a depth entry name."""
    q = {_singular(w) for w in re.split(r"[^a-z0-9]+", obj_query) if w and w not in ("the", "a", "of")}
    d = {_singular(w) for w in tokens}
    if not q or not d:
        return 0.0
    dx = set(d)
    for w in d:
        dx |= _SYN.get(w, set())
    qx = set(q)
    for w in q:
        qx |= _SYN.get(w, set())
    inter = len(q & dx) + len(d & qx)
    return inter / (len(q) + len(d))


def depth_for(obj_query: str, depth: list[dict], min_score: float = 0.3) -> list[dict]:
    """All depth entries whose name best matches the object (ties kept)."""
    if not depth:
        return []
    scored = [(depth_match_score(obj_query, e["tokens"]), i) for i, e in enumerate(depth)]
    best = max(s for s, _ in scored)
    if best < min_score:
        return []
    names = {depth[i]["name"] for s, i in scored if s == best}
    return [e for e in depth if e["name"] in names]
