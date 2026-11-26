"""The website's rating estimate: its prior and temper come from scripts/build_rating_prior.py."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from chess_engine.config import load_config
from chess_engine.data.dataset_builder import dataset_output_dir

ROOT = Path(__file__).resolve().parents[1]


def _script():
    spec = importlib.util.spec_from_file_location("build_rating_prior", ROOT / "scripts" / "build_rating_prior.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_committed_file_is_up_to_date(tmp_path) -> None:
    """site/data/rating.json is what the script builds (needs the dataset whose games give the prior)."""
    if not (dataset_output_dir(load_config()) / "games.parquet").is_file():
        pytest.skip("the dataset is not built here (the prior would be flat)")
    out = tmp_path / "rating.json"
    messages: list[str] = []
    assert _script().main(["--out", str(out)], write=messages.append) == 0
    assert messages[0].startswith("data prior over 221 ratings")
    assert json.loads(out.read_text(encoding="utf-8")) == json.loads(
        (ROOT / "site" / "data" / "rating.json").read_text(encoding="utf-8"))


def test_the_committed_file_is_a_proper_prior() -> None:
    data = json.loads((ROOT / "site" / "data" / "rating.json").read_text(encoding="utf-8"))
    assert data["points"] == len(data["log_prior"]) == (data["grid"]["max"] - data["grid"]["min"]) // data["grid"]["step"] + 1
    assert abs(np.exp(data["log_prior"]).sum() - 1) < 1e-3, "a distribution"
    assert 1 <= data["temper"] < 3
    assert data["centers"] == [900, 1100, 1300, 1500, 1700, 1900, 2100, 2300]
    t = data["test"]
    assert t["low"] < t["median"] < t["high"] and t["moves"] == 12
