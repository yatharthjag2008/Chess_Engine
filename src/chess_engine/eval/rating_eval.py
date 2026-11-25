"""Score rating estimation on held-out games: data collection, metrics, tempering, convergence.

Every position of a split is scored once per rating bucket (the model told "the mover
is in bucket b", with the real opponent rating and clocks), giving a per-move
log-likelihood matrix. Positions are grouped into *player-games*: one side of one
game, whose true rating the dataset knows.

    data = collect_eval_data(model, PositionDataset(ds, "test"), ratings)
    data.save("reports/rating/test_ll.npz")
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from chess_engine.config import RatingsConfig
from chess_engine.data.dataset import PositionDataset
from chess_engine.eval.predictions import predict_split
from chess_engine.model.network import RatingConditionedNet


@dataclass
class EvalData:
    """Per-move bucket log-likelihoods of a split, grouped into player-games."""

    player: np.ndarray        # (N,) player-game index of each move (0 .. P-1)
    ply: np.ndarray           # (N,) ply of the move in its game
    ll: np.ndarray            # (N, K) log-probability of the move at each bucket
    rating: np.ndarray        # (P,) the player's true rating
    opp_rating: np.ndarray    # (P,) the opponent's rating
    game: np.ndarray          # (P,) game row in the dataset
    color: np.ndarray         # (P,) 0 White, 1 Black

    @property
    def num_players(self) -> int:
        return len(self.rating)

    def moves_per_player(self) -> np.ndarray:
        return np.bincount(self.player, minlength=self.num_players)

    def summed(self, max_moves: int | None = None) -> np.ndarray:
        """``(P, K)`` log-likelihood per player-game (only its first ``max_moves`` moves if given)."""
        keep = np.ones(len(self.player), dtype=bool)
        if max_moves is not None:
            order = np.lexsort((self.ply, self.player))
            rank = np.empty(len(order), dtype=np.int64)
            starts = np.searchsorted(self.player[order], np.arange(self.num_players))
            rank[order] = np.arange(len(order)) - starts[self.player[order]]
            keep = rank < max_moves
        out = np.zeros((self.num_players, self.ll.shape[1]))
        np.add.at(out, self.player[keep], self.ll[keep])
        return out

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, player=self.player, ply=self.ply, ll=self.ll, rating=self.rating,
                            opp_rating=self.opp_rating, game=self.game, color=self.color)
        return path

    @classmethod
    def load(cls, path: str | Path) -> EvalData:
        with np.load(Path(path)) as f:
            return cls(**{name: f[name] for name in f.files})


def collect_eval_data(
    model: RatingConditionedNet,
    dataset: PositionDataset,
    ratings: RatingsConfig,
    batch_size: int = 1024,
    device: torch.device | str = "cpu",
    max_positions: int | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> EvalData:
    """Score the split at every bucket and group the moves into player-games (see the module docstring).

    ``progress`` is called with (buckets done, number of buckets).
    """
    k = ratings.num_buckets
    columns = []
    base = None
    for bucket in range(k):
        table = predict_split(model, dataset, ratings, batch_size, device, max_positions, conditioned_bucket=bucket)
        if base is None:
            base = table
        columns.append(np.log(np.maximum(table["target_prob"].astype(np.float64), 1e-12)))
        if progress is not None:
            progress(bucket + 1, k)
    ll = np.stack(columns, axis=1).astype(np.float32)
    game = base["game"].astype(np.int64)
    color = (base["ply"].astype(np.int64) % 2)          # games start from the initial position: ply 0 is White
    key = game * 2 + color
    unique, player = np.unique(key, return_inverse=True)
    first = np.zeros(len(unique), dtype=np.int64)
    first[player[::-1]] = np.arange(len(player))[::-1]  # one representative move per player-game
    return EvalData(player=player.astype(np.int64), ply=base["ply"].astype(np.int64), ll=ll,
                    rating=base["rating"][first].astype(np.float64),
                    opp_rating=base["opp_rating"][first].astype(np.float64),
                    game=unique // 2, color=(unique % 2).astype(np.int8))

