"""The website's opening names: built from the Lichess data set by scripts/build_openings.py."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import chess

ROOT = Path(__file__).resolve().parents[1]


def _script():
    spec = importlib.util.spec_from_file_location("build_openings", ROOT / "scripts" / "build_openings.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _key(*sans: str) -> str:
    board = chess.Board()
    for san in sans:
        board.push_san(san)
    return _script().position_key(board)


def test_build_names_positions_and_prefers_the_shortest_line(tmp_path) -> None:
    script = _script()
    rows = [("C60", "Ruy Lopez", "1. e4 e5 2. Nf3 Nc6 3. Bb5"),
            ("C20", "King's Pawn Game", "1. e4 e5"),
            ("Z99", "A longer way there", "1. Nf3 Nc6 2. e4 e5 3. Bb5")]
    table = script.build(rows)
    assert table[_key("e4", "e5")] == ["C20", "King's Pawn Game"]
    assert table[_key("e4", "e5", "Nf3", "Nc6", "Bb5")] == ["C60", "Ruy Lopez"], "the shorter line names it"
    assert len(table) == 2


def test_position_keys_ignore_en_passant_and_move_counters() -> None:
    assert _key("e4") == "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq"
    assert _key("Nf3", "Nf6", "Ng1", "Ng8") == _key(), "back to the start: the same key, whatever the counters"


def test_the_committed_table_is_up_to_date(tmp_path) -> None:
    """site/data/openings.json is what the script builds from third_party/chess-openings."""
    out = tmp_path / "openings.json"
    messages: list[str] = []
    assert _script().main(["--out", str(out)], write=messages.append) == 0
    assert "positions named" in messages[0]
    committed = json.loads((ROOT / "site" / "data" / "openings.json").read_text(encoding="utf-8"))
    assert json.loads(out.read_text(encoding="utf-8")) == committed
    assert len(committed) > 3000
    assert committed[_key("e4", "c5")] == ["B20", "Sicilian Defense"]
    assert committed[_key("d4", "Nf6", "c4", "e6", "Nc3", "Bb4")][1] == "Nimzo-Indian Defense"


def test_no_source_files_is_an_error(tmp_path, capsys) -> None:
    assert _script().main(["--source", str(tmp_path), "--out", str(tmp_path / "x.json")]) == 1
    assert "no .tsv files" in capsys.readouterr().err
