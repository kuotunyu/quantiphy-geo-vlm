from __future__ import annotations

import pytest

from quantiphy.data import TEMPLATE_CSV, TEST_PARQUET, VAL_CSV
from quantiphy.official import EVALUATOR


@pytest.fixture(scope="session", autouse=True)
def _require_local_data():
    missing = [str(p) for p in (VAL_CSV, TEST_PARQUET, TEMPLATE_CSV, EVALUATOR) if not p.exists()]
    if missing:
        pytest.fail(
            "Missing local data / starter kit. Run the setup steps in README.md first:\n  "
            + "\n  ".join(missing)
        )
