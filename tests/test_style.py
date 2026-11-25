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
