"""Tests for saving games to Parquet and the stream -> filter -> save pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import zstandard

from chess_engine.config import Config
from chess_engine.data.game_store import (
    GAME_SCHEMA,
    GameWriter,
    default_source,
    games_output_path,
    prepare_games,
    read_games,
    write_games,
)
from chess_engine.data.pgn_stream import GameFilter, filter_games, iter_raw_games
from chess_engine.data.sources import lichess_filename
from range_server import RangeServer, ServedFile

FIXTURE = Path(__file__).parent / "fixtures" / "lichess_real.pgn"


@pytest.fixture()
def records(default_config: Config) -> list:
    with FIXTURE.open(encoding="utf-8") as f:
        return list(filter_games(iter_raw_games(f), GameFilter.from_config(default_config.data)))


def test_write_and_read_roundtrip(records: list, tmp_path: Path) -> None:
    path = tmp_path / "games.parquet"
    assert write_games(records, path, batch_size=3) == len(records) == 8
    df = read_games(path)
    assert list(df.columns) == GAME_SCHEMA.names
    assert len(df) == 8
    first = df.iloc[0]
    assert first["game_id"] == records[0].game_id
    assert list(first["moves"]) == records[0].moves
    assert list(first["clocks"]) == pytest.approx(records[0].clocks)
    assert first["num_plies"] == records[0].num_plies
    assert int(first["white_elo"]) == records[0].white_elo


def test_read_selected_columns(records: list, tmp_path: Path) -> None:
    path = tmp_path / "games.parquet"
    write_games(records, path)
    assert list(read_games(path, columns=["game_id", "white_elo"]).columns) == ["game_id", "white_elo"]


def test_writer_batches_into_row_groups(records: list, tmp_path: Path) -> None:
    import pyarrow.parquet as pq

    path = tmp_path / "games.parquet"
    write_games(records, path, batch_size=3)
    assert pq.ParquetFile(path).metadata.num_row_groups == 3  # 3 + 3 + 2


def test_writer_failure_leaves_no_file(records: list, tmp_path: Path) -> None:
    path = tmp_path / "games.parquet"
    with pytest.raises(RuntimeError):
        with GameWriter(path) as writer:
            writer.write(records[0])
            raise RuntimeError("interrupted")
    assert list(tmp_path.iterdir()) == []


def test_write_empty(tmp_path: Path) -> None:
    path = tmp_path / "empty.parquet"
    assert write_games([], path) == 0
    assert len(read_games(path)) == 0


# ----------------------------------------------------------------------------- pipeline


def test_output_path_and_default_source(default_config: Config, tmp_path: Path) -> None:
    path = games_output_path(default_config, "2026-10")
    assert path == default_config.paths.processed_dir / "games" / "lichess_2026-10_sample.parquet"
    src = default_source(default_config, "2099-01")  # never downloaded -> stream from Lichess
    assert src == f"{default_config.data.lichess.base_url}/{lichess_filename('2099-01')}"


def test_prepare_games_from_local_file(default_config: Config, tmp_path: Path) -> None:
    out = tmp_path / "out.parquet"
    result = prepare_games(default_config, "2026-10", source=FIXTURE, output_path=out, show_progress=False)
    assert result.output_path == out
    assert result.stats.games_accepted == 8
    assert len(read_games(out)) == 8

    report = json.loads(result.stats_path.read_text(encoding="utf-8"))
    assert result.stats_path.name == "out.stats.json"
    assert report["games_read"] == 16
    assert report["rejected_by_reason"] == {"speed": 7, "title": 1}
    assert report["filters"]["exclude_titles"] == ["BOT"]
    assert report["sample_mode"] is True


def test_prepare_games_max_games(default_config: Config, tmp_path: Path) -> None:
    result = prepare_games(default_config, "2026-10", source=FIXTURE, output_path=tmp_path / "o.parquet",
                           max_games=3, show_progress=False)
    assert result.stats.games_accepted == 3
    assert result.stats.games_read == 3


def test_prepare_games_from_url(default_config: Config, range_server: RangeServer, tmp_path: Path) -> None:
    data = zstandard.ZstdCompressor().compress(FIXTURE.read_bytes())
    range_server.files["d.pgn.zst"] = ServedFile(data, drop_after=[2000])
    result = prepare_games(default_config, "2026-10", source=range_server.url("d.pgn.zst"),
                           output_path=tmp_path / "o.parquet", show_progress=False, backoff_seconds=0.0)
    assert result.stats.games_accepted == 8
    assert result.bytes_read == len(data)
