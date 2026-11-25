"""Style imitation and ghost mode (step 19)."""

from __future__ import annotations

import torch

from bot_helpers import small_predictor
from chess_engine.style.adapter import StyleAdapter, StyledNet


def _inputs(predictor, n: int = 4):
    import chess
    import numpy as np

    from chess_engine.encoding.board import encode_boards
    from chess_engine.model.features import buckets_from_ratings, context_features, make_inputs

    boards = [chess.Board() for _ in range(n)]
    for i, board in enumerate(boards):
        for uci in ["e2e4", "e7e5", "g1f3", "b8c6"][:i]:
            board.push_uci(uci)
    ratings = np.full(n, 1500)
    return make_inputs(encode_boards(boards), buckets_from_ratings(ratings, predictor.ratings),
                       buckets_from_ratings(ratings, predictor.ratings),
                       context_features(np.full(n, 180.0), np.full(n, 180.0), np.full(n, 180), np.zeros(n)))


def test_a_new_adapter_changes_nothing() -> None:
    predictor = small_predictor()
    base = predictor.model
    inputs = _inputs(predictor)
    with torch.no_grad():
        expected = base(inputs)
        styled = StyledNet(base)(inputs)
    assert torch.allclose(styled.policy, expected.policy) and torch.allclose(styled.value, expected.value)


def test_only_the_adapter_learns_and_the_base_stays_in_eval_mode() -> None:
    predictor = small_predictor()
    net = StyledNet(predictor.model)
    net.train()
    assert not net.base.training and net.adapter.training
    trainable = [name for name, p in net.named_parameters() if p.requires_grad]
    assert trainable == ["adapter.offset", "adapter.block_shift", "adapter.policy_bias"]
    out = net(_inputs(predictor))
    out.policy.logsumexp(dim=1).mean().backward()
    assert net.adapter.offset.grad is not None and all(p.grad is None for p in net.base.parameters())
    size = sum(p.numel() for p in net.adapter.parameters())
    cfg = predictor.model.config
    assert size == cfg.channels + cfg.residual_blocks * cfg.channels * 2 + 1858


def test_the_adapter_moves_the_policy() -> None:
    predictor = small_predictor()
    net = StyledNet(predictor.model)
    inputs = _inputs(predictor)
    with torch.no_grad():
        before = net(inputs).policy.clone()
        net.adapter.policy_bias[100] = 5.0
        net.adapter.offset += 0.5
        after = net(inputs).policy
    assert (after[:, 100] - before[:, 100]).min() > 4
    assert not torch.allclose(after, before)


def style_pgn(n_games: int = 6, player: str = "Ann", first_day: int = 1) -> str:
    """``n_games`` short games of ``player`` (alternating colours), one per day of March 2024; Ann always opens
    1. e4 as White and answers 1. d4 with ...Nf6 as Black (a recognisable style)."""
    out = []
    for g in range(n_games):
        white, black = (player, f"opp{g}") if g % 2 == 0 else (f"opp{g}", player)
        moves = "1. e4 e5 2. Nf3 Nc6 3. Bc4 Bc5 *" if g % 2 == 0 else "1. d4 Nf6 2. c4 e6 3. Nc3 Bb4 *"
        out.append(f'[White "{white}"]\n[Black "{black}"]\n[WhiteElo "1500"]\n[BlackElo "1480"]\n'
                   f'[UTCDate "2024.03.{first_day + g:02d}"]\n[TimeControl "300+3"]\n[Result "*"]\n\n{moves}\n')
    return "\n".join(out)


def test_game_dates_and_selection() -> None:
    import datetime

    from chess_engine.eval.player_games import player_games_from_pgn
    from chess_engine.style.data import game_date, select_games

    games = player_games_from_pgn(style_pgn(6), player="ann")
    assert len(games) == 6 and game_date(games[0]) == datetime.date(2024, 3, 1)
    assert len(select_games(games, until=datetime.date(2024, 3, 3))) == 3
    assert len(select_games(games, since=datetime.date(2024, 3, 5))) == 2
    assert select_games(games) == games
    games[0].headers = {"Date": "2024.??.??"}
    assert game_date(games[0]) is None and len(select_games(games, until=datetime.date(2030, 1, 1))) == 5


def test_split_by_game() -> None:
    from chess_engine.eval.player_games import player_games_from_pgn
    from chess_engine.style.data import split_games

    games = player_games_from_pgn(style_pgn(10), player="Ann")
    train, val = split_games(games, 0.2, seed=1)
    assert len(train) == 8 and len(val) == 2 and not ({id(g) for g in train} & {id(g) for g in val})
    assert split_games(games, 0.2, seed=1)[1][0] is val[0], "the same seed holds out the same games"
    assert split_games(games[:1], 0.2) == (games[:1], [])


def test_examples_from_games() -> None:
    import chess
    import numpy as np

    from chess_engine.encoding.moves import move_to_index
    from chess_engine.eval.player_games import player_games_from_pgn
    from chess_engine.style.data import examples_from_games

    predictor = small_predictor()
    games = player_games_from_pgn(style_pgn(2), player="Ann")
    examples = examples_from_games(games, predictor.ratings, default_rating=1500)
    assert len(examples) == 6 and examples.inputs.planes.shape[0] == 6
    assert examples.targets[0] == move_to_index(chess.Move.from_uci("e2e4"), chess.WHITE)
    assert examples.targets[3] == move_to_index(chess.Move.from_uci("g8f6"), chess.BLACK)
    assert examples.legal[np.arange(6), examples.targets].all(), "every move played is legal"
    skipped = examples_from_games(games, predictor.ratings, 1500, skip_opening_plies=2)
    assert len(skipped) == 4
    assert len(examples_from_games(games, predictor.ratings, 1500, max_positions=4)) == 4
    assert len(examples.subset(np.array([0, 2]))) == 2
    import pytest

    with pytest.raises(ValueError, match="no positions"):
        examples_from_games(games, predictor.ratings, 1500, skip_opening_plies=50)


def _examples(n_games: int = 12):
    from chess_engine.eval.player_games import player_games_from_pgn
    from chess_engine.style.data import examples_from_games, split_games

    predictor = small_predictor()
    games = player_games_from_pgn(style_pgn(n_games), player="Ann")
    train, val = split_games(games, 0.25, seed=0)
    return predictor, examples_from_games(train, predictor.ratings, 1500), examples_from_games(val, predictor.ratings,
                                                                                               1500)


def test_training_learns_the_players_moves() -> None:
    from chess_engine.config import StyleConfig
    from chess_engine.style.train import score, train_style

    predictor, train, val = _examples()
    before = score(predictor.model, val)
    logs: list[str] = []
    result = train_style(predictor.model, train, val, StyleConfig(epochs=30, learning_rate=0.05, patience=5),
                         log=logs.append)
    assert result.base == before
    assert result.styled.loss < before.loss * 0.5, (before, result.styled)
    # Not 100%: moves are encoded from the mover's side, so Ann's ...Nf6 as Black shares a slot with Nf3 as White.
    assert result.styled.top1 >= before.top1 + 0.3, (before, result.styled)
    assert result.best_epoch >= 1 and len(logs) == len(result.history) and logs[0].startswith("epoch 1:")
    assert result.train_positions == len(train) and result.val_positions == len(val)
    for p in predictor.model.parameters():
        assert p.requires_grad is False, "the base is frozen by the styled net"


def test_training_keeps_the_best_epoch_and_stops_early() -> None:
    from chess_engine.config import StyleConfig
    from chess_engine.style.train import train_style

    predictor, train, val = _examples()
    # A huge learning rate makes the held-out loss blow up: the best is the untrained adapter (epoch 0) or an
    # early epoch, and training stops after `patience` bad epochs.
    result = train_style(predictor.model, train, val, StyleConfig(epochs=30, learning_rate=50.0, patience=2))
    assert len(result.history) < 30
    assert result.styled.loss <= result.base.loss + 1e-6, "never worse than the base on held-out games"
    no_val = train_style(predictor.model, train, None, StyleConfig(epochs=2))
    assert no_val.base is None and no_val.styled is None and no_val.best_epoch == 2


def _info(name: str = "ann", **kw):
    from chess_engine.style.store import StyleInfo

    return StyleInfo(name=name, kind=kw.get("kind", "style"), player="Ann", games=kw.get("games", 12), positions=36,
                     rating=1500, until=kw.get("until"), metrics={"base_top1": 0.1, "styled_top1": 0.4})


def test_blend_weight() -> None:
    from chess_engine.config import StyleConfig
    from chess_engine.style.store import blend_weight

    cfg = StyleConfig(blend_games=200, min_weight=0.25)
    assert blend_weight(0, cfg) == 0.25 and blend_weight(100, cfg) == 0.625
    assert blend_weight(200, cfg) == 1.0 and blend_weight(5000, cfg) == 1.0


def test_library_round_trip(tmp_path) -> None:
    from chess_engine.style.adapter import StyleAdapter
    from chess_engine.style.store import StyleLibrary

    base = small_predictor().model
    adapter = StyleAdapter.for_model(base)
    with torch.no_grad():
        adapter.policy_bias[7] = 3.0
    library = StyleLibrary(tmp_path / "styles")
    assert library.list() == []
    path = library.save(adapter, _info(), base)
    library.save(StyleAdapter.for_model(base), _info("me-2024", kind="ghost", until="2024-06-30"), base)
    assert path.name == "ann.style.pt" and not list((tmp_path / "styles").glob("*.tmp"))
    loaded, info = library.load("ann", base)
    assert float(loaded.policy_bias[7]) == 3.0 and info.player == "Ann" and info.metrics["styled_top1"] == 0.4
    assert [i.name for i in library.list()] == ["ann", "me-2024"] and library.info("me-2024").until == "2024-06-30"
    library.delete("me-2024")
    assert [i.name for i in library.list()] == ["ann"]


def test_library_errors(tmp_path) -> None:
    import pytest

    from chess_engine.config import load_config
    from chess_engine.model.network import build_model
    from chess_engine.style.adapter import StyleAdapter
    from chess_engine.style.store import StyleError, StyleLibrary

    library = StyleLibrary(tmp_path)
    base = small_predictor().model
    for bad in ("Ann", "../x", "", "a" * 41, "has space"):
        with pytest.raises(StyleError, match="style names"):
            library.path(bad)
    with pytest.raises(StyleError, match="no style called 'nobody'"):
        library.load("nobody", base)
    (tmp_path / "junk.style.pt").write_bytes(b"not a model")
    with pytest.raises(StyleError, match="not a readable style"):
        library.info("junk")
    assert library.list() == [], "unreadable files are skipped"
    library.save(StyleAdapter.for_model(base), _info(), base)
    cfg = load_config()
    other = build_model(cfg.model_copy(update={"model": cfg.model.model_copy(update={"residual_blocks": 2})}))
    with pytest.raises(StyleError, match="different model"):
        library.load("ann", other)
    with pytest.raises(StyleError, match="no style"):
        library.delete("nobody")


def _biased_adapter(base, uci: str = "a2a3", amount: float = 8.0):
    import chess

    from chess_engine.encoding.moves import move_to_index
    from chess_engine.style.adapter import StyleAdapter

    adapter = StyleAdapter.for_model(base)
    with torch.no_grad():
        adapter.policy_bias[move_to_index(chess.Move.from_uci(uci), chess.WHITE)] = amount
    return adapter


def test_style_predictor_blends_with_the_base() -> None:
    import chess
    import pytest

    from chess_engine.style.predictor import StylePredictor

    base = small_predictor()
    board = chess.Board()
    plain = base.predict(board, 1500).as_dict()
    adapter = _biased_adapter(base.model)
    full = StylePredictor(base, adapter, 1.0).predict(board, 1500).as_dict()
    half = StylePredictor(base, adapter, 0.5).predict(board, 1500)
    assert full["a2a3"] > 0.9, "the style loves a3"
    assert half.as_dict()["a2a3"] == pytest.approx(0.5 * full["a2a3"] + 0.5 * plain["a2a3"], abs=1e-5)
    assert sum(p.prob for p in half.moves) == pytest.approx(1.0, abs=1e-5) and half.moves[0].san == "a3"
    assert half.wdl is not None and sum(half.wdl) == pytest.approx(1.0, abs=1e-5)
    none = StylePredictor(base, adapter, 0.0).predict(board, 1500).as_dict()
    assert none == pytest.approx(plain, abs=1e-6)
    with pytest.raises(ValueError, match="weight"):
        StylePredictor(base, adapter, 1.5)


def test_a_bot_plays_in_a_style() -> None:
    import chess

    from chess_engine.bot.player import BotPlayer
    from chess_engine.bot.sampling import SamplingSettings
    from chess_engine.style.predictor import StylePredictor

    base = small_predictor()
    styled = StylePredictor(base, _biased_adapter(base.model), 1.0)
    bot = BotPlayer(styled, 1500, SamplingSettings(temperature=0), seed=0)
    assert bot.choose_move(chess.Board()).san == "a3"
    assert "1400-1599" in bot.describe()
    many = styled.predict_many([chess.Board(), chess.Board()], [1200, 1800])
    assert len(many) == 2 and all(p.moves[0].san == "a3" for p in many)
