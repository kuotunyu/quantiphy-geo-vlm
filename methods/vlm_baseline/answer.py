"""Turn a free-text VLM reply into one number in the question's unit.

Rules (fixed before any validation scoring, applied identically to every question):
  1. Take the first number in the reply (thousands separators like "1,086.5" allowed).
  2. If a recognised unit follows the number and it has the same dimension as the
     unit the question asks for (length / speed / acceleration) but a different scale
     (e.g. reply "150 cm" to a question "in meters"), convert it. Otherwise keep the
     number as-is.
  3. No number -> parse failure (NaN); the caller records it and substitutes the
     label-free fallback.
"""

from __future__ import annotations

import math
import re

from methods.geometry.parse import unit_to_si

_NUM = r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)?(?:\.\d+)?(?:[eE][-+]?\d+)?"
_UNIT = (
    r"km/h|kph|"
    r"(?:kilo|centi|milli)?met(?:er|re)s?\s*(?:per|/)\s*(?:second|sec|s)(?:\s*(?:squared|\^\s*2|²|2))?|"
    r"[ckm]?m\s*/\s*s(?:\s*(?:\^\s*2|²|2)|\s*/\s*s)?|"
    r"(?:kilo|centi|milli)?met(?:er|re)s?|"
    r"[ckm]?m"
)
_ANSWER_RE = re.compile(r"(?<![\w.])(" + _NUM + r")\s*(?:(" + _UNIT + r")(?![A-Za-z]))?", re.IGNORECASE)


def unit_dim(unit: str | None) -> str | None:
    """'L', 'V', 'A' for a length / speed / acceleration unit string, else None."""
    if not isinstance(unit, str) or not unit.strip():
        return None
    u = unit.lower().replace(" ", "").replace("²", "^2")
    if u in ("km/h", "kph"):
        return "V"
    if re.search(r"(/s(\^2|2|/s)|persec(ond)?(squared|\^2|2))$", u):
        return "A"
    if re.search(r"(/s|persec(ond)?|/sec(ond)?)$", u):
        return "V"
    if re.fullmatch(r"(kilo|centi|milli)?met(er|re)s?|[ckm]?m", u):
        return "L"
    return None


def _canonical(unit: str) -> str:
    """Map a reply unit to the short form understood by `unit_to_si`."""
    u = unit.lower().replace(" ", "").replace("²", "^2")
    if u in ("kph", "km/h"):
        return "km/h"
    prefix = {"kilo": "k", "centi": "c", "milli": "m"}
    m = re.match(r"(kilo|centi|milli)?met(?:er|re)s?", u)
    if m:
        base = prefix.get(m.group(1) or "", "") + "m"
    else:
        base = re.match(r"([ckm]?m)", u).group(1)
    dim = unit_dim(u)
    return base + {"L": "", "V": "/s", "A": "/s^2"}.get(dim, "")


def parse_answer(text: str | None, target_unit: str | None) -> dict:
    """Return {'value': float (NaN on failure), 'raw_number', 'reply_unit', 'converted', 'ok'}."""
    out = {"value": math.nan, "raw_number": None, "reply_unit": None, "converted": False, "ok": False}
    if not isinstance(text, str):
        return out
    for m in _ANSWER_RE.finditer(text):
        s = m.group(1)
        if not re.search(r"\d", s):
            continue
        try:
            v = float(s.replace(",", ""))
        except ValueError:
            continue
        if not math.isfinite(v):
            continue
        out.update(raw_number=v, value=v, ok=True)
        unit = m.group(2)
        if unit:
            out["reply_unit"] = unit
            d_reply, d_target = unit_dim(unit), unit_dim(target_unit)
            if d_reply and d_reply == d_target:
                f_reply, _ = unit_to_si(_canonical(unit), d_reply)
                f_target, _ = unit_to_si(target_unit, d_target)
                if not math.isclose(f_reply, f_target):
                    out.update(value=v * f_reply / f_target, converted=True)
        return out
    return out
