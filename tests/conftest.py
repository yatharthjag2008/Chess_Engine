"""Shared pytest fixtures."""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from chess_engine.config import Config, load_config
from chess_engine.stockfish import EngineInfo, StockfishError, get_stockfish
from range_server import RangeServer

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture()
def range_server() -> Iterator[RangeServer]:
    """A local HTTP server with Range support; add files via ``range_server.files[name] = ServedFile(...)``."""
    server = RangeServer().start()
    try:
        yield server
    finally:
        server.stop()


@pytest.fixture(scope="session")
def default_config() -> Config:
    """The project's default configuration."""
    return load_config()


@pytest.fixture()
def fake_engine_command() -> list[str]:
    """Command line that starts the fake UCI engine with the current Python interpreter."""
    return [sys.executable, str(FIXTURES_DIR / "fake_uci_engine.py")]


@pytest.fixture(scope="session")
def stockfish(default_config: Config) -> EngineInfo:
    """A verified real Stockfish binary, or skip the test with install instructions."""
    try:
        return get_stockfish(default_config)
    except StockfishError as exc:
        pytest.skip(f"Stockfish not available, skipping. Details:\n{exc}")
