"""Readers for the QuantiPhy validation set, test set and submission template.

All loaders return a DataFrame with the same normalized columns:

    qid             int   question id (validation: first CSV column; test: template `id`)
    video_id        str
    video_path      str   absolute path to the local .mp4
    video_source    str
    video_type      str   4-char code [P][D][O][B], e.g. "V3MS"
    fps             int
    inference_type  str   "SS" | "SD" | "DS" | "DD" (prior kind, target kind)
    question        str
    prior           str   free-text prior, e.g. "length of boat = 3.62m"
    depth_info      str | NaN    free text (meters), present for 3D questions
    category        str   "2S" | "2D" | "3S" | "3D" (website naming)
    official_category str "S2" | "D2" | "S3" | "D3" (evaluator.py naming)
    prior_value     float  last number after "=" or "~" in `prior` (NaN if none)
    prior_unit      str    unit string right after that number ("" if none)
    target_unit     str | None  unit asked in the question ("meters", "m/s", "cm/s^2", ...)
    answer          float  ground truth in the question's unit (validation only; absent for test)
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
VAL_DIR = DATA / "QuantiPhy-validation"
TEST_DIR = DATA / "QuantiPhy"
VAL_CSV = VAL_DIR / "validation_dataset.csv"
VAL_PARQUET = VAL_DIR / "validation_dataset.parquet"
TEST_PARQUET = TEST_DIR / "test_dataset.parquet"
TEMPLATE_CSV = DATA / "submission_template" / "quantiphy_submission_template.csv"

CATEGORIES = ("2S", "2D", "3S", "3D")
# Official evaluator builds the key as inference_type[0] + video_type[1].
OFFICIAL_TO_SITE = {"S2": "2S", "D2": "2D", "S3": "3S", "D3": "3D"}
SITE_TO_OFFICIAL = {v: k for k, v in OFFICIAL_TO_SITE.items()}

_NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
_PRIOR_RE = re.compile(r"[=~]\s*(" + _NUM + r")\s*([A-Za-z/^²2]*)")
_TARGET_UNIT_RE = re.compile(
    r"\bin\s*(meters?|metres?|centimeters?|millimeters?|kilometers?|km/h|"
    r"(?:[cmk]?m)\s*/\s*s\s*(?:\^\s*2|²)?|[cmk]?m)\b",
    re.IGNORECASE,
)


def parse_prior(prior: str | None) -> tuple[float, str]:
    """Return (value, unit) of the last `= <number><unit>` in a prior string."""
    if not isinstance(prior, str):
        return float("nan"), ""
    matches = list(_PRIOR_RE.finditer(prior))
    if not matches:
        return float("nan"), ""
    m = matches[-1]
    return float(m.group(1)), m.group(2)


def parse_target_unit(question: str | None) -> str | None:
    """Return the output unit requested in the question text (last `in <unit>`)."""
    if not isinstance(question, str):
        return None
    found = _TARGET_UNIT_RE.findall(question)
    if not found:
        return None
    return re.sub(r"\s+", "", found[-1]).lower().replace("²", "^2")


def official_category(inference_type: str, video_type: str) -> str | None:
    if not isinstance(inference_type, str) or not isinstance(video_type, str):
        return None
    return inference_type[0] + video_type[1]


def resolve_video(video_dir: Path, video_id: str) -> Path:
    """Local path of a video. One validation file ships with a leading space in its
    name (" captured_0041x.mp4"); fall back to that spelling when needed."""
    p = video_dir / f"{video_id}.mp4"
    if not p.exists():
        alt = video_dir / f" {video_id}.mp4"
        if alt.exists():
            return alt
    return p


def _finalize(df: pd.DataFrame, video_dir: Path) -> pd.DataFrame:
    df = df.copy()
    df["qid"] = df["qid"].astype(int)
    df["fps"] = df["fps"].astype(int)
    df["video_path"] = [str(resolve_video(video_dir, v)) for v in df["video_id"]]
    df["official_category"] = [
        official_category(i, v) for i, v in zip(df["inference_type"], df["video_type"])
    ]
    df["category"] = df["official_category"].map(OFFICIAL_TO_SITE)
    parsed = [parse_prior(p) for p in df["prior"]]
    df["prior_value"] = np.array([p[0] for p in parsed], dtype=float)
    df["prior_unit"] = [p[1] for p in parsed]
    df["target_unit"] = [parse_target_unit(q) for q in df["question"]]
    cols = [
        "qid", "video_id", "video_path", "video_source", "video_type", "fps",
        "inference_type", "question", "prior", "depth_info", "category",
        "official_category", "prior_value", "prior_unit", "target_unit",
    ]
    if "answer" in df.columns:
        cols.append("answer")
    return df[cols].reset_index(drop=True)


def load_validation(csv_path: Path = VAL_CSV) -> pd.DataFrame:
    """Validation set (159 questions, with answers).

    Uses the CSV because only it carries the question id (first, unnamed column)
    that the official evaluator matches on; the parquet has identical rows.
    """
    # Default CSV parsing on purpose: identical to how evaluator.py reads the GT file.
    raw = pd.read_csv(csv_path)
    raw = raw.rename(
        columns={
            raw.columns[0]: "qid",
            "ground_truth_prior": "prior",
            "ground_truth_posterior": "answer",
        }
    )
    raw = raw[[c for c in raw.columns if not str(c).startswith("Unnamed")]]
    return _finalize(raw, VAL_DIR / "validation_videos")


def load_test(parquet_path: Path = TEST_PARQUET, template_path: Path = TEMPLATE_CSV) -> pd.DataFrame:
    """Test set (3,289 questions, no answers). qid = submission template `id`.

    The template and the parquet are row-aligned; this is asserted on load.
    """
    test = pd.read_parquet(parquet_path)
    tmpl = pd.read_csv(template_path)
    if len(test) != len(tmpl):
        raise ValueError(f"row count mismatch: parquet={len(test)} template={len(tmpl)}")
    for col in ("video_id", "video_type", "inference_type", "question"):
        if not (test[col].to_numpy() == tmpl[col].to_numpy()).all():
            raise ValueError(f"template and parquet disagree on column {col!r}")
    test = test.copy()
    test["qid"] = tmpl["id"].to_numpy()
    return _finalize(test, TEST_DIR)


def load_template(template_path: Path = TEMPLATE_CSV) -> pd.DataFrame:
    """Official submission template as-is (fill `parsed_value`, keep `id` and row order)."""
    return pd.read_csv(template_path)


def read_predictions(path: str | Path) -> pd.Series:
    """Read a prediction CSV -> Series of raw `parsed_value` indexed by question id.

    Like evaluator.py, the id is taken from the *first* column (named `id` in files
    we write, `id` in the official template, `Unnamed: 0` in the starter-kit example).
    Non-numeric cells are kept as-is; the scorer coerces them to NaN like the official code.
    """
    # Default CSV parsing on purpose: identical to how evaluator.py reads predictions.
    df = pd.read_csv(path)
    if "parsed_value" not in df.columns:
        raise ValueError(f"{path}: need a 'parsed_value' column, got {list(df.columns)}")
    ids = pd.to_numeric(df[df.columns[0]], errors="raise").astype(int)
    if not ids.is_unique:
        raise ValueError(f"{path}: duplicate ids in first column")
    return pd.Series(df["parsed_value"].to_numpy(), index=ids.to_numpy(), name="parsed_value")


# Columns the official evaluator reads from a prediction file (besides the id).
OFFICIAL_META_COLUMNS = ("video_id", "video_source", "video_type", "inference_type", "question")


def write_predictions(path: str | Path, meta: pd.DataFrame, values) -> Path:
    """Write predictions in a format that both our scorer and evaluator.py accept.

    `meta` is a loader DataFrame (validation or test) and `values` is aligned with it.
    Columns: id, video_id, video_source, video_type, inference_type, question, parsed_value.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    values = list(values)
    if len(values) != len(meta):
        raise ValueError(f"{len(values)} values for {len(meta)} questions")
    out = pd.DataFrame({"id": meta["qid"].to_numpy()})
    for c in OFFICIAL_META_COLUMNS:
        out[c] = meta[c].to_numpy()
    out["parsed_value"] = values
    out.to_csv(path, index=False)
    return path
