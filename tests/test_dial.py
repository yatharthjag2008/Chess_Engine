"""The honest rating dial (chess_engine.eval.dial, scripts/build_dial.py)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from chess_engine.eval.dial import calibrate, isotonic

ROOT = Path(__file__).resolve().parents[1]


def _bots(measured, width=200.0):
    return [{"dial": d, "measured": m, "interval": [m - width / 2, m + width / 2]}
            for d, m in zip([800, 1100, 1400, 1700, 2000, 2300], measured)]


def test_isotonic_pools_violators_by_weight() -> None:
    assert isotonic([1, 3, 2, 4], [1, 1, 1, 1]) == [1, 2.5, 2.5, 4]
    assert isotonic([1, 3, 2, 4], [1, 3, 1, 1]) == [1, 2.75, 2.75, 4], "the surer measurement counts more"
    assert isotonic([5, 4, 3], [1, 1, 1]) == [4, 4, 4]
    assert isotonic([1, 2], [1, 1]) == [1, 2]
    with pytest.raises(ValueError):
        isotonic([1], [0])


def test_labels_are_measured_differences_anchored_at_1500() -> None:
    c = calibrate(_bots([300, 300, 600, 900, 900, 1000]))
    labels = [round(p.label) for p in c.points]
    # strength at 1500: 600 + (900 - 600) / 3 = 700, so each label is 1500 + measured - 700
    assert labels == [1100, 1100, 1400, 1700, 1700, 1800]
    assert c.label_for(1500) == pytest.approx(1500)
    assert c.range() == (1100, 1800)
    assert c.input_for(1400) == 1400 and c.input_for(1550) == 1550
    assert c.input_for(1100) == 800, "the lowest input that plays that well"
    assert c.input_for(1750) == 2150, "past the plateau 1700-2000"
    assert c.input_for(500) == 800 and c.input_for(3000) == 2300, "the ends"
    for label in range(1100, 1801, 50):
        assert c.label_for(c.input_for(label)) == pytest.approx(label, abs=1), label


def test_an_inverted_pair_is_made_monotone() -> None:
    c = calibrate(_bots([300, 320, 650, 910, 830, 980]))
    fitted = [p.fitted for p in c.points]
    assert fitted == sorted(fitted)
    assert fitted[3] == pytest.approx(fitted[4]) == pytest.approx(870), "1700 and 2000 pooled"
    with pytest.raises(ValueError, match="anchor"):
        calibrate(_bots([1, 2, 3, 4, 5, 6]), anchor=3000)


def test_the_committed_dial_is_what_the_script_builds(tmp_path) -> None:
    spec = importlib.util.spec_from_file_location("build_dial", ROOT / "scripts" / "build_dial.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    out = tmp_path / "dial.json"
    messages: list[str] = []
    assert module.main(["--out", str(out)], write=messages.append) == 0
    assert messages[0].startswith("dial 1100-1750 (anchor 1500)")
    committed = json.loads((ROOT / "site" / "data" / "dial.json").read_text(encoding="utf-8"))
    assert json.loads(out.read_text(encoding="utf-8")) == committed
    assert [p["input"] for p in committed["points"]] == [800, 1100, 1400, 1700, 2000, 2300]
