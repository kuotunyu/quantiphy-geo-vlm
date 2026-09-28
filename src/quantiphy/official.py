"""Thin wrapper that runs the unmodified official evaluator (external/QuantiPhy/evaluator.py).

The evaluator is a script (argparse at import time), so we execute it with
runpy in-process, with sys.argv patched and stdout captured. Its module globals
are returned too, which gives access to the per-question `mra` column of the
last processed file for item-level comparisons. The evaluator file itself is
never edited.

The prediction file is copied byte-for-byte into a temporary input folder, so
the evaluator parses exactly the same text as our own scorer (read_predictions).
"""

from __future__ import annotations

import contextlib
import io
import runpy
import shutil
import sys
import tempfile
from pathlib import Path

import pandas as pd

from .data import ROOT, VAL_CSV

EVALUATOR = ROOT / "external" / "QuantiPhy" / "evaluator.py"


def evaluator_available() -> bool:
    return EVALUATOR.exists()


def run_official(pred_csv: Path | str, gt_csv: Path | str = VAL_CSV) -> tuple[dict, pd.DataFrame, str]:
    """Run evaluator.py on one prediction CSV.

    Returns (summary_row, per_item_df, captured_stdout). summary_row holds
    mra_S2/mra_D2/mra_S3/mra_D3/mra_average/invalid_percentage (read back from the
    evaluator's CSV with round-trip float parsing, so no precision is lost).
    per_item_df has columns qid, category, parsed_value, ground_truth_posterior, mra.
    """
    if not evaluator_available():
        raise FileNotFoundError(f"official evaluator not found at {EVALUATOR}")
    pred_csv = Path(pred_csv)
    with tempfile.TemporaryDirectory(prefix="qp_official_") as tmp:
        in_dir = Path(tmp) / "model_outputs"
        out_dir = Path(tmp) / "mra_results"
        in_dir.mkdir()
        shutil.copyfile(pred_csv, in_dir / "model.csv")
        argv = [str(EVALUATOR), str(in_dir), str(out_dir), "--gt_file", str(gt_csv)]
        buf = io.StringIO()
        old_argv = sys.argv
        try:
            sys.argv = argv
            with contextlib.redirect_stdout(buf):
                g = runpy.run_path(str(EVALUATOR), run_name="__main__")
        finally:
            sys.argv = old_argv
        summary = pd.read_csv(out_dir / "all_model_results.csv", float_precision="round_trip")
    row = summary.iloc[0].to_dict()
    df = g["df"]
    per_item = df[[df.columns[0], "category", "parsed_value", "ground_truth_posterior", "mra"]].copy()
    per_item = per_item.rename(columns={df.columns[0]: "qid"})
    return row, per_item, buf.getvalue()
