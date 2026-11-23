"""Tests for game filtering, GameRecord and FilterStats."""

from __future__ import annotations

import pytest

from chess_engine.config import Config
from chess_engine.data.pgn_stream import (
    FilterStats,
    GameFilter,
    GameRecord,
    Movetext,
    RawGame,
    Reject,
    filter_games,
    game_id_from_site,
    is_rated,
)

GOOD_HEADERS = {
    "Event": "Rated Blitz game",
    "Site": "https://lichess.org/game/Q8ku5wgM",
    "UTCDate": "2026.10.01",
    "UTCTime": "00:00:25",
    "White": "alice",
    "Black": "bob",
    "Result": "1-0",
    "WhiteElo": "1825",
    "BlackElo": "1790",
    "TimeControl": "300+0",
    "Termination": "Normal",
    "ECO": "C20",
    "Opening": "King's Pawn Game",
}


def _movetext(plies: int, clocks: bool = True) -> str:
    """Legal-looking movetext with ``plies`` half-moves (knights shuffling back and forth)."""
    cycle = ["Nf3", "Nf6", "Ng1", "Ng8"]
    parts = []
    for i in range(plies):
        if i % 2 == 0:
            parts.append(f"{i // 2 + 1}.")
        parts.append(cycle[i % 4])
        if clocks:
            parts.append("{ [%clk 0:05:00] }")
    parts.append("1-0")
    return " ".join(parts)


def _raw(movetext_plies: int = 30, clocks: bool = True, **header_changes: str | None) -> RawGame:
    headers = dict(GOOD_HEADERS)
    for key, value in header_changes.items():
        if value is None:
            headers.pop(key, None)
        else:
            headers[key] = value
    return RawGame(headers, _movetext(movetext_plies, clocks))


@pytest.fixture()
def game_filter(default_config: Config) -> GameFilter:
    return GameFilter.from_config(default_config.data)


# ----------------------------------------------------------------------------- GameFilter


def test_from_config(game_filter: GameFilter) -> None:
    assert game_filter.speeds == {"blitz"}
    assert game_filter.min_rating == 800
    assert game_filter.max_rating is None
    assert game_filter.min_plies == 20
    assert "Abandoned" in game_filter.exclude_terminations
    assert game_filter.exclude_titles == {"BOT"}
    assert game_filter.require_clocks is True


def test_good_game_passes(game_filter: GameFilter) -> None:
    raw = _raw()
    assert game_filter.check_headers(raw.headers) is None


@pytest.mark.parametrize(
    "changes, reason",
    [
        ({"Event": "Casual Blitz game"}, Reject.NOT_RATED),
        ({"Event": None}, Reject.NOT_RATED),
        ({"Variant": "Chess960"}, Reject.VARIANT),
        ({"TimeControl": "60+0"}, Reject.SPEED),
        ({"TimeControl": "600+5"}, Reject.SPEED),
        ({"TimeControl": "-"}, Reject.SPEED),
        ({"TimeControl": None}, Reject.SPEED),
        ({"Termination": "Abandoned"}, Reject.TERMINATION),
        ({"Termination": "Rules infraction"}, Reject.TERMINATION),
        ({"WhiteTitle": "BOT"}, Reject.TITLE),
        ({"BlackTitle": "BOT"}, Reject.TITLE),
        ({"WhiteElo": "?"}, Reject.RATING_MISSING),
        ({"BlackElo": None}, Reject.RATING_MISSING),
        ({"BlackElo": "799"}, Reject.RATING_TOO_LOW),
    ],
)
def test_header_rejections(game_filter: GameFilter, changes: dict, reason: Reject) -> None:
    assert game_filter.check_headers(_raw(**changes).headers) == reason


@pytest.mark.parametrize(
    "headers, rated",
    [
        ({"Event": "Rated Blitz game"}, True),
        ({"Event": "Rated Blitz tournament https://lichess.org/tournament/x"}, True),
        # Swiss events don't say "Rated"; rating changes show they were.
        ({"Event": "Blitz swiss https://lichess.org/swiss/x", "WhiteRatingDiff": "+5"}, True),
        ({"Event": "Blitz swiss https://lichess.org/swiss/x", "BlackRatingDiff": "-5"}, True),
        ({"Event": "Blitz swiss https://lichess.org/swiss/x"}, False),
        ({"Event": "Casual Blitz game", "WhiteRatingDiff": "+0"}, False),
        ({}, False),
    ],
)
def test_is_rated(headers: dict, rated: bool) -> None:
    assert is_rated(headers) is rated


@pytest.mark.parametrize(
    "changes",
    [
        {"WhiteTitle": "GM"},
        {"Variant": "Standard"},
        {"Termination": "Time forfeit"},
        {"TimeControl": "180+2"},
        {"WhiteElo": "800"},
        {"Event": "Rated Blitz tournament https://lichess.org/tournament/x"},
    ],
)
def test_header_edge_cases_pass(game_filter: GameFilter, changes: dict) -> None:
    assert game_filter.check_headers(_raw(**changes).headers) is None


def test_max_rating() -> None:
    f = GameFilter(speeds=frozenset({"blitz"}), min_rating=1000, max_rating=2000)
    assert f.check_headers(_raw(WhiteElo="2000").headers) is None
    assert f.check_headers(_raw(WhiteElo="2001").headers) == Reject.RATING_TOO_HIGH
    assert f.check_headers(_raw(BlackElo="999").headers) == Reject.RATING_TOO_LOW


def test_rated_only_off_and_no_titles_excluded() -> None:
    f = GameFilter(speeds=frozenset({"blitz"}), rated_only=False)
    assert f.check_headers(_raw(Event="Casual Blitz game", WhiteTitle="BOT").headers) is None


def test_movetext_checks(game_filter: GameFilter) -> None:
    assert game_filter.check_movetext(Movetext(["e4"] * 20, [1.0] * 20, "1-0")) is None
    assert game_filter.check_movetext(Movetext(["e4"] * 19, [1.0] * 19, "1-0")) == Reject.TOO_SHORT
    assert game_filter.check_movetext(Movetext(["e4"] * 20, [], "1-0")) == Reject.NO_CLOCKS
    no_clock_filter = GameFilter(speeds=frozenset({"blitz"}), require_clocks=False)
    assert no_clock_filter.check_movetext(Movetext(["e4"] * 20, [], "1-0")) is None


# ----------------------------------------------------------------------------- filter_games


def test_filter_games_counts_and_records(game_filter: GameFilter) -> None:
    games = [
        _raw(),
        _raw(TimeControl="60+0"),
        _raw(WhiteTitle="BOT"),
        _raw(movetext_plies=10),
        _raw(clocks=False),
        _raw(BlackElo="2500"),
        _raw(TimeControl="60+0"),
    ]
    stats = FilterStats()
    records = list(filter_games(games, game_filter, stats))

    assert len(records) == 2
    assert stats.games_read == 7
    assert stats.games_accepted == 2
    assert stats.games_rejected == 5
    assert stats.rejected == {"speed": 2, "title": 1, "too_short": 1, "no_clocks": 1}
    summary = stats.as_dict()
    assert summary["acceptance_rate"] == pytest.approx(2 / 7, abs=1e-4)
    assert list(summary["rejected_by_reason"])[0] == "speed"  # most common first


def test_filter_games_max_games_stops_early(game_filter: GameFilter) -> None:
    def endless():
        while True:
            yield _raw()

    stats = FilterStats()
    records = list(filter_games(endless(), game_filter, stats, max_games=5))
    assert len(records) == 5
    assert stats.games_read == 5


def test_filter_games_max_games_zero(game_filter: GameFilter) -> None:
    assert list(filter_games([_raw()], game_filter, max_games=0)) == []


def test_empty_stats() -> None:
    assert FilterStats().as_dict() == {
        "games_read": 0, "games_accepted": 0, "games_rejected": 0, "acceptance_rate": 0.0, "rejected_by_reason": {}
    }


# ----------------------------------------------------------------------------- GameRecord


def test_game_record_fields(game_filter: GameFilter) -> None:
    (record,) = filter_games([_raw(movetext_plies=24, BlackTitle="FM")], game_filter)
    assert isinstance(record, GameRecord)
    assert record.game_id == "Q8ku5wgM"
    assert record.utc_date == "2026-10-01"
    assert record.utc_time == "00:00:25"
    assert (record.white, record.black) == ("alice", "bob")
    assert (record.white_elo, record.black_elo) == (1825, 1790)
    assert (record.white_title, record.black_title) == ("", "FM")
    assert (record.base_seconds, record.increment_seconds, record.speed) == (300, 0, "blitz")
    assert record.result == "1-0"
    assert record.num_plies == 24
    assert record.moves[:4] == ["Nf3", "Nf6", "Ng1", "Ng8"]
    assert record.clocks == [300.0] * 24
    as_dict = record.to_dict()
    assert as_dict["opening"] == "King's Pawn Game"
    assert set(as_dict) == set(GameRecord.__dataclass_fields__)


@pytest.mark.parametrize(
    "site, game_id",
    [
        ("https://lichess.org/game/Q8ku5wgM", "Q8ku5wgM"),
        ("https://lichess.org/abcd1234", "abcd1234"),
        ("https://lichess.org/abcd1234/", "abcd1234"),
        ("", ""),
    ],
)
def test_game_id_from_site(site: str, game_id: str) -> None:
    assert game_id_from_site(site) == game_id
