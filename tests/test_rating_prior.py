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


def test_the_opponent_prior_is_fitted_and_scored_on_held_out_games() -> None:
    data = json.loads((ROOT / "site" / "data" / "rating.json").read_text(encoding="utf-8"))
    o = data["opponent"]
    assert o["sigma"] > 0 and o["temper"] >= 1
    assert abs(o["val"]["coverage"] - 0.8) <= 0.02, "calibrated on the validation games"
    assert o["test"]["mae"] < 100, "far better than from the moves alone (about 250)"
    assert o["test"]["players"] > 1000
    assert o["test_vector"]["low"] < o["test_vector"]["estimate"] < o["test_vector"]["high"]


def test_opponent_log_prior_multiplies_over_games() -> None:
    from chess_engine.eval.rating_estimation import opponent_log_prior

    grid = np.arange(1000, 2001, 100, dtype=float)
    one = opponent_log_prior(grid, [1500], 100)
    assert one.argmax() == 5 and one[5] == 0 and one[6] == pytest.approx(-0.5)
    two = opponent_log_prior(grid, [1400, 1600], 100)
    assert two.argmax() == 5 and two[5] == pytest.approx(-1.0), "between them, and sharper"
    with pytest.raises(ValueError):
        opponent_log_prior(grid, [], 100)
    with pytest.raises(ValueError):
        opponent_log_prior(grid, [1500], 0)

