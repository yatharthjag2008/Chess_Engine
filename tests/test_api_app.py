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
