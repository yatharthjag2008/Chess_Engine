"""HTTP tests of the API (FastAPI TestClient, small in-memory model)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from chess_engine.api.app import create_app
from chess_engine.config import Config
from bot_helpers import small_predictor


@pytest.fixture(scope="module")
def client(default_config: Config) -> TestClient:
    with TestClient(create_app(default_config, small_predictor())) as c:
        yield c


def test_health(client: TestClient) -> None:
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok" and response.json()["model"] == "in-memory model"


def test_create_and_get_a_game(client: TestClient) -> None:
    response = client.post("/api/games", json={"rating": 1200, "color": "black", "seed": 3})
    assert response.status_code == 201
    state = response.json()
    assert state["human_color"] == "black" and state["turn"] == "black" and len(state["moves_san"]) == 1
    assert state["bot_rating"] == 1200 and state["status"] == "ongoing"
    again = client.get(f"/api/games/{state['id']}")
    assert again.status_code == 200 and again.json() == state
    assert client.get("/api/health").json()["games"] >= 1


def test_defaults_and_validation(client: TestClient) -> None:
    state = client.post("/api/games", json={}).json()
    assert state["human_color"] == "white" and state["bot_rating"] == 1500 and len(state["legal_moves"]) == 20
    for body in ({"rating": 50}, {"color": "green"}, {"time_control": "soon"}, {"fen": "nonsense"}, {"extra": 1}):
        response = client.post("/api/games", json=body)
        assert response.status_code == 422, body


def test_unknown_game(client: TestClient) -> None:
    response = client.get("/api/games/does-not-exist")
    assert response.status_code == 404 and response.json()["error"] == "game_not_found"


def test_model_loaded_from_the_config(tmp_path, default_config: Config) -> None:
    from chess_engine.model.inference import save_model

    predictor = small_predictor()
    path = save_model(predictor.model, tmp_path / "m.pt", predictor.ratings)
    cfg = default_config.model_copy(update={"bot": default_config.bot.model_copy(update={"model": path})})
    with TestClient(create_app(cfg)) as c:
        assert c.get("/api/health").json()["model"] == "m.pt"
        assert c.post("/api/games", json={}).status_code == 201


def _new(client: TestClient, **body) -> dict:
    return client.post("/api/games", json={"seed": 5, **body}).json()


def test_legal_moves_endpoint(client: TestClient) -> None:
    game = _new(client)
    moves = client.get(f"/api/games/{game['id']}/legal-moves").json()
    assert len(moves) == 20 and {"uci", "san", "from_square", "to_square", "promotion"} <= set(moves[0])
    assert client.get("/api/games/nope/legal-moves").status_code == 404


def test_play_a_move_and_get_the_reply(client: TestClient) -> None:
    game = _new(client)
    response = client.post(f"/api/games/{game['id']}/moves", json={"move": "e2e4"})
    assert response.status_code == 200
    body = response.json()
    assert body["played"] == "e4" and body["bot_move"]["san"] == body["state"]["moves_san"][1]
    assert len(body["bot_move"]["candidates"]) == 5 and 0 < body["bot_move"]["prob"] <= 1
    assert body["state"]["turn"] == "white" and body["state"]["last_bot_move"] == body["bot_move"]


def test_move_errors(client: TestClient) -> None:
    game = _new(client)
    url = f"/api/games/{game['id']}/moves"
    illegal = client.post(url, json={"move": "e5"})
    assert illegal.status_code == 422 and illegal.json()["error"] == "illegal_move"
    assert client.post(url, json={"move": "Nf3", "reply": False}).status_code == 200
    turn = client.post(url, json={"move": "d4"})
    assert turn.status_code == 409 and turn.json()["error"] == "not_your_turn"
    assert client.post(url, json={}).status_code == 422          # missing move
    assert client.post("/api/games/nope/moves", json={"move": "e4"}).status_code == 404


def test_play_to_checkmate_over_http(client: TestClient) -> None:
    game = _new(client, fen="7k/8/6K1/8/8/8/R7/1R6 w - - 0 1")
    url = f"/api/games/{game['id']}/moves"
    client.post(url, json={"move": "Ra7"})
    final = client.post(url, json={"move": "Rb8#"}).json()
    assert final["state"]["status"] == "over" and final["state"]["outcome"]["result"] == "1-0"
    assert client.get(f"/api/games/{game['id']}/legal-moves").json() == []
    over = client.post(url, json={"move": "Kh7"})
    assert over.status_code == 409 and over.json()["error"] == "game_over"


def test_bot_move_endpoint(client: TestClient) -> None:
    game = _new(client)
    base = f"/api/games/{game['id']}"
    early = client.post(f"{base}/bot-move")
    assert early.status_code == 409 and early.json()["error"] == "not_bots_turn"
    client.post(f"{base}/moves", json={"move": "d4", "reply": False})
    answer = client.post(f"{base}/bot-move").json()
    assert answer["played"] == answer["bot_move"]["san"] and answer["state"]["turn"] == "white"


def test_resign_and_pgn(client: TestClient) -> None:
    game = _new(client, your_rating=1650)
    base = f"/api/games/{game['id']}"
    client.post(f"{base}/moves", json={"move": "e4"})
    state = client.post(f"{base}/resign").json()
    assert state["status"] == "over" and state["outcome"]["termination"] == "resignation"
    assert state["outcome"]["winner"] == "black"
    assert client.post(f"{base}/resign").status_code == 409
    pgn = client.get(f"{base}/pgn")
    assert pgn.status_code == 200 and pgn.headers["content-type"].startswith("text/plain")
    assert '[Result "0-1"]' in pgn.text and '[WhiteElo "1650"]' in pgn.text and "1. e4" in pgn.text


def test_takeback(client: TestClient) -> None:
    game = _new(client)
    base = f"/api/games/{game['id']}"
    nothing = client.post(f"{base}/takeback")
    assert nothing.status_code == 409 and nothing.json()["error"] == "nothing_to_take_back"
    client.post(f"{base}/moves", json={"move": "e4"})
    client.post(f"{base}/moves", json={"move": "Nf3"})
    state = client.post(f"{base}/takeback").json()
    assert len(state["moves_san"]) == 2 and state["moves_san"][0] == "e4" and state["turn"] == "white"
    client.post(f"{base}/moves", json={"move": "d4", "reply": False})
    state = client.post(f"{base}/takeback").json()  # the bot has not answered: only your move goes
    assert len(state["moves_san"]) == 2 and state["last_bot_move"] is None


def test_takeback_never_undoes_the_bots_opening(client: TestClient) -> None:
    game = _new(client, color="black")
    base = f"/api/games/{game['id']}"
    assert client.post(f"{base}/takeback").status_code == 409
    client.post(f"{base}/moves", json={"move": game["legal_moves"][0]["uci"]})
    state = client.post(f"{base}/takeback").json()
    assert len(state["moves_san"]) == 1 and state["turn"] == "black"


def test_timed_game_over_http(client: TestClient) -> None:
    game = _new(client, time_control="3+2")
    assert game["clock"] == {"white": 180, "black": 180, "running": None, "base_seconds": 180, "increment": 2}
    state = client.post(f"/api/games/{game['id']}/moves", json={"move": "e4"}).json()["state"]
    assert state["clock"]["running"] == "white" and 179 < state["clock"]["white"] <= 180


def test_validation_errors_have_the_common_shape(client: TestClient) -> None:
    response = client.post("/api/games", json={"rating": 50, "color": "green"})
    body = response.json()
    assert response.status_code == 422 and body["error"] == "invalid_request"
    assert "rating" in body["detail"] and "color" in body["detail"]
    broken = client.post("/api/games", content="{not json", headers={"Content-Type": "application/json"})
    assert broken.status_code == 422 and broken.json()["error"] == "invalid_request"


def test_cors_and_root(client: TestClient) -> None:
    allowed = client.options("/api/games", headers={"Origin": "http://localhost:8000",
                                                    "Access-Control-Request-Method": "POST"})
    assert allowed.headers.get("access-control-allow-origin") == "http://localhost:8000"
    other = client.get("/api/health", headers={"Origin": "http://evil.example"})
    assert "access-control-allow-origin" not in other.headers
    assert client.get("/docs").status_code == 200
    paths = client.get("/openapi.json").json()["paths"]
    assert {"/api/games", "/api/games/{game_id}/moves", "/api/games/{game_id}/takeback"} <= set(paths)


def test_web_ui_is_served_at_the_root(client: TestClient) -> None:
    page = client.get("/")
    assert page.status_code == 200 and page.headers["content-type"].startswith("text/html")
    assert '<div id="board"' in page.text and 'src="js/app.js"' in page.text
    assert client.get("/index.html").status_code == 200
    assert client.get("/api/health").json()["status"] == "ok"  # the API is not shadowed
    assert client.get("/no-such-file.js").status_code == 404


def test_without_a_web_directory_the_root_redirects_to_the_docs(default_config: Config, tmp_path) -> None:
    for web_dir in (None, tmp_path):
        with TestClient(create_app(default_config, small_predictor(), web_dir=web_dir)) as c:
            root = c.get("/", follow_redirects=False)
            assert root.status_code == 307 and root.headers["location"] == "/docs"


def test_stylesheet_is_served(client: TestClient) -> None:
    css = client.get("/css/style.css")
    assert css.status_code == 200 and css.headers["content-type"].startswith("text/css")
    assert "--square-light" in css.text and "prefers-color-scheme: dark" in css.text


# --------------------------------------------------------------------------- game import


FIXTURE_PGN = (__import__("pathlib").Path(__file__).parent / "fixtures" / "lichess_real.pgn").read_text(encoding="utf-8")


def test_import_pasted_pgn(client: TestClient) -> None:
    response = client.post("/api/import/pgn", json={"pgn": FIXTURE_PGN, "max_games": 5})
    body = response.json()
    assert response.status_code == 200 and body["source"] == "pgn" and body["count"] == 5
    game = body["games"][0]
    assert {"white", "black", "speed", "plies", "pgn", "id"} <= set(game) and game["pgn"].startswith("[Event")
    assert client.post("/api/import/pgn", json={"pgn": "nothing here"}).json()["count"] == 0
    assert client.post("/api/import/pgn", json={"pgn": ""}).status_code == 422


def test_import_by_username(default_config: Config, tmp_path) -> None:
    from fake_api import FakeApi, Reply

    with FakeApi() as api:
        api.routes["/api/games/user/someone"] = Reply(200, FIXTURE_PGN)
        api.routes["/api/games/user/broken"] = Reply(503)
        cfg = default_config.model_copy(update={
            "game_import": default_config.game_import.model_copy(update={"lichess_url": api.url, "retries": 0}),
            "paths": default_config.paths.model_copy(update={"data_dir": tmp_path}),
        })
        with TestClient(create_app(cfg, small_predictor())) as c:
            body = c.get("/api/import/lichess/someone", params={"max_games": 4, "speed": ["blitz"]}).json()
            assert body["username"] == "someone" and body["count"] <= 4
            assert all(g["speed"] == "blitz" for g in body["games"])
            assert api.requests[0][1]["perfType"] == ["blitz"]
            again = c.get("/api/import/lichess/someone", params={"max_games": 4, "speed": ["blitz"]})
            assert again.json() == body and len(api.requests) == 1   # served from the cache
            missing = c.get("/api/import/lichess/nobody")
            assert missing.status_code == 404 and missing.json()["error"] == "user_not_found"
            bad = c.get("/api/import/lichess/bad..name")
            assert bad.status_code == 422 and bad.json()["error"] == "invalid_username"
            site = c.get("/api/import/fics/someone")
            assert site.status_code == 422 and site.json()["error"] == "invalid_username"
            down = c.get("/api/import/lichess/broken")
            assert down.status_code == 502 and down.json()["error"] == "source_unavailable"


# --------------------------------------------------------------------------- rating estimate


def test_rating_estimate_endpoint(client: TestClient) -> None:
    from chess_engine.data.game_import import parse_pgn_text

    name = parse_pgn_text(FIXTURE_PGN)[0].white
    response = client.post("/api/rating-estimate", json={"pgn": FIXTURE_PGN, "player": name})
    body = response.json()
    assert response.status_code == 200
    o = body["overall"]
    assert o["low"] <= o["estimate"] <= o["high"] and o["moves"] > 0 and body["temper"] > 0
    assert body["games"] and all(name.lower() in (g["white"].lower(), g["black"].lower()) for g in body["games"])
    by_colour = client.post("/api/rating-estimate", json={"pgn": FIXTURE_PGN, "color": "black",
                                                           "use_opponent_rating": False}).json()
    assert len(by_colour["games"]) == 16 and all(g["color"] == "black" for g in by_colour["games"])
    for bad in ({"pgn": FIXTURE_PGN}, {"pgn": FIXTURE_PGN, "player": "x", "color": "white"},
                {"pgn": FIXTURE_PGN, "player": "nobody-at-all"}):
        r = client.post("/api/rating-estimate", json=bad)
        assert r.status_code == 422 and r.json()["error"] == "invalid_request", bad


# --------------------------------------------------------------------------- review


def test_review_endpoints(default_config: Config, tmp_path) -> None:
    from test_review_analyzer import PGN, _scripted

    cfg = default_config.model_copy(update={"paths": default_config.paths.model_copy(update={"data_dir": tmp_path})})
    with TestClient(create_app(cfg, small_predictor(), review_analyzer=_scripted())) as c:
        body = c.post("/api/review", json={"pgn": PGN}).json()
        assert [m["san"] for m in body["moves"]][:3] == ["e4", "e5", "Qh5"]
        assert body["moves"][5]["classification"] == "blunder" and body["summary"]["black"]["counts"]["blunder"] == 1
        assert body["key_moments"][0]["san"] == "Nf6" and len(body["graph"]) == 8
        bad = c.post("/api/review", json={"pgn": PGN, "game_index": 3})
        assert bad.status_code == 422 and bad.json()["error"] == "invalid_request"
        game = c.post("/api/games", json={"seed": 1}).json()
        c.post(f"/api/games/{game['id']}/moves", json={"move": "e4"})
        bot_review = c.get(f"/api/games/{game['id']}/review").json()
        assert [m["san"] for m in bot_review["moves"]][0] == "e4" and len(bot_review["moves"]) == 2
        assert c.get("/api/games/nope/review").status_code == 404


def test_review_without_stockfish(default_config: Config, tmp_path, monkeypatch) -> None:
    from chess_engine.stockfish import StockfishError
    import chess_engine.review.run as run_module

    def missing(path):
        raise StockfishError("Stockfish was not found\n(install instructions)")

    monkeypatch.setattr(run_module, "find_stockfish", missing)
    cfg = default_config.model_copy(update={
        "paths": default_config.paths.model_copy(update={"data_dir": tmp_path}),
        "review": default_config.review.model_copy(update={"cache": False}),
    })
    with TestClient(create_app(cfg, small_predictor())) as c:
        r = c.post("/api/review", json={"pgn": "1. e4 e5 *"})
        assert r.status_code == 503 and r.json()["error"] == "stockfish_unavailable"
        assert "not found" in r.json()["detail"] and "install" not in r.json()["detail"]



def test_coach_endpoint(default_config: Config, tmp_path) -> None:
    from test_review_analyzer import PGN, _scripted

    cfg = default_config.model_copy(update={"paths": default_config.paths.model_copy(update={"data_dir": tmp_path})})
    with TestClient(create_app(cfg, small_predictor(), review_analyzer=_scripted())) as c:
        body = c.post("/api/coach", json={"pgn": PGN, "color": "black", "rating": 1300}).json()
        assert body["rating_source"] == "given" and body["next_level"] == 1500
        report = body["report"]
        assert report["color"] == "black" and report["summary"]["counts"]["blunder"] == 1
        assert any(n["san"] == "Nf6" and "mistake" in n["kinds"] for n in report["notes"])
        by_name = c.post("/api/coach", json={"pgn": PGN, "player": "w", "rating": 1600}).json()
        assert by_name["report"]["color"] == "white"
        for bad in ({"pgn": PGN}, {"pgn": PGN, "player": "w", "color": "white"}, {"pgn": PGN, "player": "zed"}):
            r = c.post("/api/coach", json=bad)
            assert r.status_code == 422 and r.json()["error"] == "invalid_request", bad



def test_review_page_and_navigation_are_served(client: TestClient) -> None:
    page = client.get("/review.html")
    assert page.status_code == 200 and 'id="graph"' in page.text and 'src="js/review.js"' in page.text
    assert 'href="index.html"' in page.text and 'aria-current="page">Review' in page.text
    assert 'href="review.html"' in client.get("/").text
    for asset in ("/css/review.css", "/js/review.js", "/js/api.js"):
        assert client.get(asset).status_code == 200
    assert "reviewBotGame" in client.get("/js/api.js").text


def test_review_styles(client: TestClient) -> None:
    css = client.get("/css/review.css").text
    for rule in (".graph .marker.blunder", ".square.best-from", ".review-moves .san.current", "prefers-color-scheme"):
        assert rule in css, rule


def test_hint_endpoint(default_config: Config) -> None:
    from review_helpers import ScriptedAnalyzer

    with TestClient(create_app(default_config, small_predictor(), review_analyzer=ScriptedAnalyzer())) as c:
        game = c.post("/api/games", json={"seed": 5, "hint_limit": 1}).json()
        url = f"/api/games/{game['id']}/hint"
        nudge = c.post(url).json()
        assert nudge["tier"] == "nudge" and nudge["text"] and nudge["from_square"] is None
        assert nudge["next_tier"] == "point" and nudge["hints"] == {"limit": 1, "used": 1, "left": 0, "rated": False}
        assert c.post(url, json={}).json()["tier"] == "point"
        show = c.post(url).json()
        assert show["tier"] == "show" and show["san"] == "a3" and (show["from_square"], show["to_square"]) == ("a2", "a3")
        assert show["next_tier"] is None, "the scripted engine's best move is the first legal one"
        c.post(f"/api/games/{game['id']}/moves", json={"move": "e4"})
        r = c.post(url)
        assert r.status_code == 409 and r.json()["error"] == "no_hints_left"
        state = c.get(f"/api/games/{game['id']}").json()
        assert state["hints"]["rated"] is False
        assert '[Hints "1"]' in c.get(f"/api/games/{game['id']}/pgn").text
        bad = c.post(url, json={"tier": "everything"})
        assert bad.status_code == 422
        black = c.post("/api/games", json={"color": "black", "seed": 5}).json()
        c.post(f"/api/games/{black['id']}/resign")
        r = c.post(f"/api/games/{black['id']}/hint")
        assert r.status_code == 409 and r.json()["error"] == "game_over"
        timed = c.post("/api/games", json={"time_control": "5+0"}).json()
        assert c.post(f"/api/games/{timed['id']}/hint").status_code == 200, "clock times go to the model"
        off = c.post("/api/games", json={"hint_limit": 0}).json()
        r = c.post(f"/api/games/{off['id']}/hint")
        assert r.status_code == 409 and "off" in r.json()["detail"]


def test_hint_without_stockfish(default_config: Config, monkeypatch) -> None:
    from chess_engine.stockfish import StockfishError
    import chess_engine.api.app as app_module

    def missing(path):
        raise StockfishError("Stockfish was not found")

    monkeypatch.setattr(app_module, "find_stockfish", missing)
    with TestClient(create_app(default_config, small_predictor())) as c:
        game = c.post("/api/games", json={}).json()
        r = c.post(f"/api/games/{game['id']}/hint")
        assert r.status_code == 503 and r.json()["error"] == "stockfish_unavailable"
        assert c.get(f"/api/games/{game['id']}").json()["hints"]["used"] == 0, "a failed hint is not counted"


def test_adaptive_endpoints(default_config: Config, tmp_path) -> None:
    cfg = default_config.model_copy(update={"paths": default_config.paths.model_copy(update={"data_dir": tmp_path})})
    with TestClient(create_app(cfg, small_predictor())) as c:
        fresh = c.get("/api/adaptive").json()
        assert fresh == {"rating": 1200.0, "games": 0, "next_bot_rating": 1200, "default_mode": "off", "history": []}
        assert c.post("/api/adaptive/reset", json={"rating": 1630}).json()["next_bot_rating"] == 1650
        game = c.post("/api/games", json={"adaptive": "between", "seed": 1}).json()
        assert game["bot_rating"] == 1650 and game["adaptive"]["mode"] == "between"
        over = c.post(f"/api/games/{game['id']}/resign").json()
        assert over["adaptive"]["result"]["rating_after"] == 1592.3 and over["adaptive"]["result"]["next_bot_rating"] == 1600
        profile = c.get("/api/adaptive").json()
        assert profile["games"] == 1 and profile["history"][0]["score"] == 0 and profile["history"][0]["bot_rating"] == 1650
        assert (tmp_path / cfg.adaptive.profile_file).is_file()
        assert c.post("/api/adaptive/reset").json()["rating"] == 1200
        assert c.post("/api/adaptive/reset", json={"rating": 5}).status_code == 422
        assert c.post("/api/games", json={"adaptive": "sometimes"}).status_code == 422


def test_styles_endpoint_and_playing_in_a_style(default_config: Config, tmp_path) -> None:
    from test_style import _biased_adapter, _info

    from chess_engine.style.store import StyleLibrary

    cfg = default_config.model_copy(update={"paths": default_config.paths.model_copy(update={"data_dir": tmp_path})})
    predictor = small_predictor()
    with TestClient(create_app(cfg, predictor)) as c:
        assert c.get("/api/styles").json() == []
        library = StyleLibrary(tmp_path / cfg.style.directory)
        library.save(_biased_adapter(predictor.model), _info("ann", games=200), predictor.model)
        styles = c.get("/api/styles").json()
        assert [s["name"] for s in styles] == ["ann"] and styles[0]["weight"] == 1.0
        assert styles[0]["styled_top1"] == 0.4 and styles[0]["kind"] == "style"
        game = c.post("/api/games", json={"color": "black", "style": "ann", "temperature": 0}).json()
        assert game["moves_san"] == ["a3"], "the style's favourite first move"
        assert game["style"]["name"] == "ann" and game["style"]["player"] == "Ann"
        assert '[White "Bot 1500 (Ann style)"]' in c.get(f"/api/games/{game['id']}/pgn").text
        plain = c.post("/api/games", json={"color": "black", "temperature": 0}).json()
        assert plain["style"] is None and plain["moves_san"] != ["a3"]
        missing = c.post("/api/games", json={"style": "nobody"})
        assert missing.status_code == 404 and missing.json()["error"] == "style_not_found"
