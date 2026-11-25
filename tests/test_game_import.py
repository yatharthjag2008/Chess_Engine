"""Tests for importing games (pasted PGN; the Lichess / Chess.com fetchers use a local fake server)."""

from __future__ import annotations

from pathlib import Path

import pytest

from chess_engine.data.game_import import ImportedGame, parse_pgn_text, speed_of_time_control

FIXTURE = Path(__file__).parent / "fixtures" / "lichess_real.pgn"

CHESSCOM_PGN = """[Event "Live Chess"]
[Site "Chess.com"]
[Date "2026.11.20"]
[White "PlayerOne"]
[Black "PlayerTwo"]
[Result "1-0"]
[WhiteElo "1234"]
[BlackElo "1250"]
[TimeControl "600"]
[Link "https://www.chess.com/game/live/123456789"]

1. e4 {[%clk 0:09:58.5]} 1... e5 {[%clk 0:09:57]} 2. Qh5 Nc6 3. Bc4 Nf6 4. Qxf7# 1-0
"""

VARIANT = '[Event "Rated Chess960 game"]\n[Variant "Chess960"]\n[FEN "bbqnnrkr/pppppppp/8/8/8/8/PPPPPPPP/BBQNNRKR w KQkq - 0 1"]\n[SetUp "1"]\n[Result "*"]\n\n1. e4 *\n'


def test_speed_of_time_control() -> None:
    assert speed_of_time_control("180+2") == "blitz"
    assert speed_of_time_control("600") == "rapid"            # Chess.com writes no "+0"
    assert speed_of_time_control("60") == "bullet"
    assert speed_of_time_control("1/86400") == "correspondence"
    assert speed_of_time_control("-") is None and speed_of_time_control(None) is None
    assert speed_of_time_control("abc") is None


def test_real_lichess_games() -> None:
    games = parse_pgn_text(FIXTURE.read_text(encoding="utf-8"), source="lichess")
    assert len(games) == 16 and len({g.id for g in games}) == 16
    first = games[0]
    assert isinstance(first, ImportedGame) and first.source == "lichess"
    assert first.url.startswith("https://lichess.org/") and first.id == first.url.rsplit("/", 1)[-1]
    assert first.speed in {"ultrabullet", "bullet", "blitz", "rapid", "classical"} and first.plies > 0
    assert any(g.rated for g in games) and first.pgn.startswith("[Event")
    assert "pgn" not in first.summary() and first.summary()["white"] == first.white


def test_chesscom_pgn_and_duplicates_and_variants() -> None:
    games = parse_pgn_text(CHESSCOM_PGN + "\n" + CHESSCOM_PGN + "\n" + VARIANT)
    assert len(games) == 1                                   # the repeat and the Chess960 game are dropped
    g = games[0]
    assert g.id == "123456789" and g.url == "https://www.chess.com/game/live/123456789"
    assert (g.white, g.black, g.white_elo, g.black_elo, g.result) == ("PlayerOne", "PlayerTwo", 1234, 1250, "1-0")
    assert g.speed == "rapid" and g.date == "2026.11.20" and g.plies == 7 and g.rated is None
    assert "[%clk 0:09:58.5]" in g.pgn                       # clocks are kept for the rating estimate


def test_pasted_games_without_ids_and_junk() -> None:
    pasted = '[White "A"]\n[Black "B"]\n[Result "*"]\n\n1. d4 d5 *\n\n[White "A"]\n[Black "B"]\n[Result "*"]\n\n1. c4 *\n'
    games = parse_pgn_text(pasted)
    assert len(games) == 2 and games[0].id != games[1].id and len(games[0].id) == 12
    assert games[0].date is None and games[0].speed is None and games[0].white_elo is None
    assert parse_pgn_text("") == [] and parse_pgn_text("hello there") == []
    assert parse_pgn_text("1. e4 e5 2. Ke7 *") == []           # illegal moves


@pytest.mark.parametrize("event, rated", [("Rated Blitz game", True), ("Casual Rapid game", False),
                                          ("Live Chess", None), ("Unrated blitz", False)])
def test_rated_from_the_event(event: str, rated: bool | None) -> None:
    (g,) = parse_pgn_text(f'[Event "{event}"]\n[Result "*"]\n\n1. e4 *\n')
    assert g.rated is rated


@pytest.mark.parametrize("source, name, ok", [
    ("lichess", "DrNykterstein", True), ("lichess", " some_player-2 ", True), ("lichess", "ab", True),
    ("lichess", "a", False), ("lichess", "-dash", False), ("lichess", "x" * 31, False),
    ("lichess", "../admin", False), ("lichess", "a/b", False), ("lichess", "name?x=1", False),
    ("chess.com", "Hikaru", True), ("chess.com", "ab", False), ("chess.com", "with space", False),
    ("chess.com", "x" * 26, False),
])
def test_check_username(source: str, name: str, ok: bool) -> None:
    from chess_engine.data.game_import import InvalidUsernameError, check_username

    if ok:
        assert check_username(source, name) == name.strip()
    else:
        with pytest.raises(InvalidUsernameError):
            check_username(source, name)


def test_unknown_source() -> None:
    from chess_engine.data.game_import import InvalidUsernameError, check_username

    with pytest.raises(InvalidUsernameError, match="unknown source"):
        check_username("chess24", "someone")


# --------------------------------------------------------------------------- HTTP client


def _client(**kwargs):
    from chess_engine.data.game_import import HttpClient

    waits: list[float] = []
    return HttpClient(timeout=5, sleep=waits.append, **kwargs), waits


def test_http_client_retries_rate_limits_and_server_errors() -> None:
    from chess_engine.data.game_import import USER_AGENT
    from fake_api import FakeApi, Reply

    with FakeApi() as api:
        api.routes["/x"] = [Reply(429, headers={"Retry-After": "7"}), Reply(503), Reply(200, "ok")]
        client, waits = _client(retries=3, backoff=0.5)
        response = client.get(api.url + "/x", params={"max": 3}, accept="application/x-chess-pgn")
        assert response.text == "ok" and waits == [7.0, 1.0]   # Retry-After honoured, then 0.5 * 2**1
        path, query, headers = api.requests[0]
        assert query == {"max": ["3"]} and headers["accept"] == "application/x-chess-pgn"
        assert headers["user-agent"] == USER_AGENT


def test_http_client_errors() -> None:
    from chess_engine.data.game_import import SourceUnavailableError, UserNotFoundError
    from fake_api import FakeApi, Reply

    with FakeApi() as api:
        api.routes["/down"] = Reply(502)
        api.routes["/forbidden"] = Reply(403)
        client, waits = _client(retries=2)
        with pytest.raises(UserNotFoundError):
            client.get(api.url + "/missing", what="Lichess")
        with pytest.raises(SourceUnavailableError, match="HTTP 502.*3 attempts"):
            client.get(api.url + "/down")
        assert len(waits) == 2
        with pytest.raises(SourceUnavailableError, match="HTTP 403"):
            client.get(api.url + "/forbidden")
    client, _ = _client(retries=0)
    with pytest.raises(SourceUnavailableError, match="cannot reach"):
        client.get("http://127.0.0.1:9/")                       # nothing listens on port 9


# --------------------------------------------------------------------------- Lichess


def test_fetch_lichess_sends_the_filters_and_parses_the_pgn() -> None:
    from chess_engine.data.game_import import fetch_lichess
    from fake_api import FakeApi, Reply

    pgn = FIXTURE.read_text(encoding="utf-8")
    with FakeApi() as api:
        api.routes["/api/games/user/SomePlayer"] = Reply(200, pgn, {"Content-Type": "application/x-chess-pgn"})
        client, _ = _client()
        games = fetch_lichess(" SomePlayer ", max_games=5, speeds=["blitz", "ultrabullet"], rated_only=True,
                              base_url=api.url, client=client)
        assert len(games) == 5 and all(g.source == "lichess" for g in games)
        path, query, headers = api.requests[0]
        assert path == "/api/games/user/SomePlayer"
        assert query["max"] == ["5"] and query["perfType"] == ["blitz,ultraBullet"] and query["rated"] == ["true"]
        assert query["clocks"] == ["true"] and headers["accept"] == "application/x-chess-pgn"


def test_fetch_lichess_errors() -> None:
    from chess_engine.data.game_import import InvalidUsernameError, UserNotFoundError, fetch_lichess
    from fake_api import FakeApi

    with FakeApi() as api:
        client, _ = _client()
        with pytest.raises(UserNotFoundError, match="Lichess"):
            fetch_lichess("nobody_here", base_url=api.url, client=client)
        with pytest.raises(InvalidUsernameError):
            fetch_lichess("bad/name", base_url=api.url, client=client)
        assert len(api.requests) == 1          # the invalid name never reached the server


# --------------------------------------------------------------------------- Chess.com


def _chesscom_game(i: int, time_class: str = "rapid", rated: bool = True, rules: str = "chess") -> dict:
    pgn = CHESSCOM_PGN.replace("123456789", str(1000 + i)).replace('[Date "2026.11.20"]', f'[Date "2026.11.{i:02d}"]')
    return {"url": f"https://www.chess.com/game/live/{1000 + i}", "pgn": pgn, "time_class": time_class,
            "rated": rated, "rules": rules, "white": {"username": "PlayerOne"}, "black": {"username": "PlayerTwo"}}


def test_fetch_chesscom_walks_archives_newest_first() -> None:
    import json

    from chess_engine.data.game_import import fetch_chesscom
    from fake_api import FakeApi, Reply

    with FakeApi() as api:
        base = api.url
        api.routes["/pub/player/playerone/games/archives"] = Reply(200, json.dumps({"archives": [
            f"{base}/pub/player/playerone/games/2026/10", f"{base}/pub/player/playerone/games/2026/11",
            "https://evil.example/pub/player/playerone/games/2026/12"]}))
        api.routes["/pub/player/playerone/games/2026/11"] = Reply(200, json.dumps({"games": [
            _chesscom_game(1), _chesscom_game(2, rules="chess960"), _chesscom_game(3, time_class="bullet"),
            _chesscom_game(4, rated=False), _chesscom_game(5)]}))
        api.routes["/pub/player/playerone/games/2026/10"] = Reply(200, json.dumps({"games": [
            _chesscom_game(6), _chesscom_game(7)]}))
        client, _ = _client()
        games = fetch_chesscom("PlayerOne", max_games=3, speeds=["rapid"], rated_only=True, base_url=base,
                               client=client)
        assert [g.id for g in games] == ["1005", "1001", "1007"]       # newest first, filters applied
        assert all(g.source == "chess.com" and g.rated and g.speed == "rapid" for g in games)
        assert games[0].url == "https://www.chess.com/game/live/1005"
        assert [path for path, _, _ in api.requests] == [               # the foreign link was not followed
            "/pub/player/playerone/games/archives", "/pub/player/playerone/games/2026/11",
            "/pub/player/playerone/games/2026/10"]
        everything = fetch_chesscom("playerone", max_games=50, base_url=base, client=client)
        assert len(everything) == 6                                     # only the Chess960 game is left out
        assert {g.speed for g in everything} == {"rapid", "bullet"}


def test_fetch_chesscom_unknown_user() -> None:
    from chess_engine.data.game_import import UserNotFoundError, fetch_chesscom
    from fake_api import FakeApi

    with FakeApi() as api:
        client, _ = _client()
        with pytest.raises(UserNotFoundError, match="Chess.com"):
            fetch_chesscom("nobody", base_url=api.url, client=client)


# --------------------------------------------------------------------------- entry point


def test_filter_games() -> None:
    from chess_engine.data.game_import import filter_games

    games = parse_pgn_text(FIXTURE.read_text(encoding="utf-8"), source="lichess")
    blitz = filter_games(games, ["blitz"])
    assert blitz and all(g.speed == "blitz" for g in blitz)
    assert all(g.rated for g in filter_games(games, rated_only=True))
    assert len(filter_games(games, max_games=3)) == 3 and filter_games(games) == games


def test_import_games_from_every_source(default_config) -> None:
    import json

    from chess_engine.data.game_import import GameImportError, InvalidUsernameError, import_games
    from fake_api import FakeApi, Reply

    pasted = import_games(default_config.game_import, "pgn", text=FIXTURE.read_text(encoding="utf-8"),
                          speeds=["blitz"], max_games=2)
    assert pasted.source == "pgn" and pasted.username is None and len(pasted.games) == 2
    assert all(g.speed == "blitz" for g in pasted.games)
    with pytest.raises(GameImportError, match="paste"):
        import_games(default_config.game_import, "pgn")
    with pytest.raises(GameImportError, match="username"):
        import_games(default_config.game_import, "lichess")
    with pytest.raises(InvalidUsernameError, match="unknown source"):
        import_games(default_config.game_import, "fics", username="someone")

    with FakeApi() as api:
        cfg = default_config.game_import.model_copy(update={"lichess_url": api.url, "chesscom_url": api.url,
                                                           "max_games": 4})
        api.routes["/api/games/user/someone"] = Reply(200, FIXTURE.read_text(encoding="utf-8"))
        api.routes["/pub/player/someone/games/archives"] = Reply(200, json.dumps({"archives": []}))
        client, _ = _client()
        lichess = import_games(cfg, "lichess", username="someone", client=client)
        assert lichess.source == "lichess" and lichess.username == "someone" and len(lichess.games) == 4
        assert api.requests[0][1]["max"] == ["4"]              # the config's limit reached the API
        assert import_games(cfg, "chess.com", username="someone", client=client).games == []
