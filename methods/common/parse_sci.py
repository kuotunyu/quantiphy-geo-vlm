"""Read "2.27×10³ cm/s"-style replies as 2270 cm/s.

The frozen parser (methods/vlm_baseline/answer.parse_answer) reads only the mantissa of a
number written as "mantissa × 10^exponent", so such a reply comes out 10^exponent times too
small. It is left untouched (every earlier run and upload used it). parse_answer_sci first
rewrites "mantissa × 10^exponent" (× x X * · ⋅ \\times; exponent after ^, ^{...} or as
superscript digits, with an optional minus sign) as "mantissa e exponent" -- the form
parse_answer already reads -- and then calls parse_answer unchanged. Replies without that
pattern are parsed exactly as before. Only the reply text is used, never an answer.

Text only (no model, no API); used by methods/caw.
"""

from __future__ import annotations

import math
import re

from methods.vlm_baseline.answer import parse_answer

_SUPERSCRIPT = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺", "0123456789-+")
_MINUS = str.maketrans({"−": "-", "–": "-"})
_SCI_RE = re.compile(
    r"(?<![\w.])([-+−]?\d+(?:\.\d+)?)\s*(?:[×xX*·⋅]|\\times)\s*10\s*"
    r"(?:\^\s*\{?\s*([-+−–]?\s*\d+)(?:\s*\})?|([⁻⁺]?[⁰¹²³⁴⁵⁶⁷⁸⁹]+))"
)


def _repl(m: re.Match) -> str:
    exp = m.group(2) if m.group(2) is not None else m.group(3).translate(_SUPERSCRIPT)
    exp = exp.translate(_MINUS).replace(" ", "")
    out = f"{m.group(1).translate(_MINUS)}e{exp}"
    v = float(out)
    # an exponent that over- or underflows would change which number is read: keep the frozen behaviour
    return out if math.isfinite(v) and v != 0 else m.group(0)


def normalize_sci(text: str | None) -> str | None:
    """'2.27×10³ cm/s' -> '2.27e3 cm/s'; any other text is returned unchanged."""
    if not isinstance(text, str):
        return text
    return _SCI_RE.sub(_repl, text)


def parse_answer_sci(text: str | None, target_unit: str | None) -> dict:
    """parse_answer after normalize_sci; identical to parse_answer on replies without the pattern."""
    return parse_answer(normalize_sci(text), target_unit)
