"""Tests for finding and verifying Stockfish.

Most tests use a fake UCI engine (tests/fixtures/fake_uci_engine.py) so they run
everywhere. Tests marked ``stockfish`` use the real binary and are skipped with
install instructions when it is not installed.
"""

from __future__ import annotations

from pathlib import Path

import chess
import chess.engine
import pytest

from chess_engine import stockfish as sf
from chess_engine.stockfish import (
    ENV_VAR,
    EngineInfo,
    StockfishError,
    check_uci_engine,
    find_stockfish,
    install_instructions,
)


@pytest.fixture()
def no_stockfish_anywhere(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pretend Stockfish is not installed: no env var, not on PATH, no common locations."""
    monkeypatch.delenv(ENV_VAR, raising=False)
    monkeypatch.setattr(sf.shutil, "which", lambda name: None)
    monkeypatch.setattr(sf, "COMMON_LOCATIONS", ())


@pytest.fixture()
def fake_binary(tmp_path: Path) -> Path:
    path = tmp_path / "stockfish.exe"
    path.write_bytes(b"")
    return path


# ----------------------------------------------------------------------------- find_stockfish


def test_not_found_gives_install_instructions(no_stockfish_anywhere: None) -> None:
    with pytest.raises(StockfishError) as excinfo:
        find_stockfish(None)
    message = str(excinfo.value)
    assert "not found" in message
    assert "Install Stockfish" in message
    assert ENV_VAR in message


def test_finds_configured_path(no_stockfish_anywhere: None, fake_binary: Path) -> None:
    assert find_stockfish(fake_binary) == fake_binary.resolve()


def test_env_var_takes_priority_over_config(
    no_stockfish_anywhere: None, monkeypatch: pytest.MonkeyPatch, fake_binary: Path, tmp_path: Path
) -> None:
    monkeypatch.setenv(ENV_VAR, str(fake_binary))
    assert find_stockfish(tmp_path / "some_other_path") == fake_binary.resolve()


def test_wrong_explicit_path_raises_instead_of_falling_back(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fake_binary: Path
) -> None:
    monkeypatch.delenv(ENV_VAR, raising=False)
    # Even though a binary exists on "PATH", a wrong explicit path is an error.
    monkeypatch.setattr(sf.shutil, "which", lambda name: str(fake_binary) if name == "stockfish" else None)
    with pytest.raises(StockfishError, match="does not exist"):
        find_stockfish(tmp_path / "typo" / "stockfish.exe")


def test_wrong_env_var_mentions_env_var(no_stockfish_anywhere: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_VAR, "/definitely/not/here/stockfish")
    with pytest.raises(StockfishError, match=ENV_VAR):
        find_stockfish(None)


def test_falls_back_to_path(monkeypatch: pytest.MonkeyPatch, fake_binary: Path) -> None:
    monkeypatch.delenv(ENV_VAR, raising=False)
    monkeypatch.setattr(sf.shutil, "which", lambda name: str(fake_binary) if name == "stockfish" else None)
    assert find_stockfish(None) == fake_binary.resolve()


def test_falls_back_to_common_locations(no_stockfish_anywhere: None, monkeypatch: pytest.MonkeyPatch, fake_binary: Path) -> None:
    monkeypatch.setattr(sf, "COMMON_LOCATIONS", ("/not/here/stockfish", str(fake_binary)))
    assert find_stockfish(None) == fake_binary.resolve()


def test_common_locations_support_glob_patterns(no_stockfish_anywhere: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Mimics winget's layout: Packages/Stockfish.Stockfish_<hash>/stockfish/stockfish-windows-x86-64-avx2.exe
    binary = tmp_path / "Stockfish.Stockfish_abc123" / "stockfish" / "stockfish-windows-x86-64-avx2.exe"
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"")
    monkeypatch.setattr(sf, "COMMON_LOCATIONS", (str(tmp_path / "Stockfish.Stockfish_*" / "**" / "stockfish*.exe"),))
    assert find_stockfish(None) == binary.resolve()


@pytest.mark.parametrize(
    "system, expected",
    [("Windows", "winget"), ("Darwin", "brew install stockfish"), ("Linux", "apt install stockfish")],
)
def test_install_instructions_are_os_specific(system: str, expected: str) -> None:
    assert expected in install_instructions(system)


# ----------------------------------------------------------------------------- check_uci_engine (fake engine)


def test_check_uci_engine_with_fake_engine(fake_engine_command: list[str]) -> None:
    info = check_uci_engine(fake_engine_command, timeout=20)
    assert info == EngineInfo(path=" ".join(fake_engine_command), name="FakeFish 1.0", test_move="e2e4")


def test_check_uci_engine_rejects_illegal_move(fake_engine_command: list[str]) -> None:
    with pytest.raises(StockfishError, match="illegal|legal move"):
        check_uci_engine([*fake_engine_command, "--illegal"], timeout=20)


def test_check_uci_engine_missing_binary(tmp_path: Path) -> None:
    with pytest.raises(StockfishError, match="Could not run"):
        check_uci_engine(str(tmp_path / "missing_engine.exe"), timeout=5)


def test_check_uci_engine_non_executable_file(tmp_path: Path) -> None:
    not_an_engine = tmp_path / "notes.txt"
    not_an_engine.write_text("hello", encoding="utf-8")
    with pytest.raises(StockfishError, match="Could not run"):
        check_uci_engine(str(not_an_engine), timeout=5)


# ----------------------------------------------------------------------------- real Stockfish


@pytest.mark.stockfish
def test_real_stockfish_runs(stockfish: EngineInfo) -> None:
    assert "stockfish" in stockfish.name.lower()
    assert chess.Move.from_uci(stockfish.test_move) in chess.Board().legal_moves


@pytest.mark.stockfish
def test_real_stockfish_finds_mate_in_one(stockfish: EngineInfo) -> None:
    # Scholar's mate: after 1.e4 e5 2.Bc4 Nc6 3.Qh5 Nf6?? White mates with 4.Qxf7#.
    board = chess.Board()
    for san in ["e4", "e5", "Bc4", "Nc6", "Qh5", "Nf6"]:
        board.push_san(san)
    with chess.engine.SimpleEngine.popen_uci(stockfish.path) as engine:
        result = engine.play(board, chess.engine.Limit(depth=8))
    assert result.move == chess.Move.from_uci("h5f7")
