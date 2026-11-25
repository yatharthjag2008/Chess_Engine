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
