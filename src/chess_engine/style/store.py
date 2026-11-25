"""Styles on disk: one file per style (the adapter's weights and what it was trained on), in a library folder.

    library = StyleLibrary(cfg.paths.data_dir / cfg.style.directory)
    library.save(adapter, info)
    library.list()                       # [StyleInfo, ...]
    adapter, info = library.load("ann", base_model)
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

from chess_engine.config import StyleConfig
from chess_engine.model.network import RatingConditionedNet
from chess_engine.style.adapter import StyleAdapter

STYLE_FORMAT_VERSION = 1
NAME = re.compile(r"[a-z0-9][a-z0-9_-]{0,39}")


class StyleError(Exception):
    """A style is missing, unreadable, badly named or made for another model."""


@dataclass
class StyleInfo:
    """What a style was trained on and how well it imitates the player."""

    name: str
    kind: str                          # "style" (any player) or "ghost" (you, up to a date)
    player: str
    games: int
    positions: int
    rating: int                        # the player's typical rating in these games
    until: str | None = None           # ghost: the last date included (YYYY-MM-DD)
    created: str = ""
    base_model: str = ""
    metrics: dict[str, float] = field(default_factory=dict)   # base / styled loss and top-1 on held-out games

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def blend_weight(games: int, cfg: StyleConfig) -> float:
    """How much the style counts next to the base model: ``min_weight`` with no games, 1 at ``blend_games``."""
    return min(1.0, cfg.min_weight + (1 - cfg.min_weight) * games / cfg.blend_games)


def check_name(name: str) -> str:
    if not NAME.fullmatch(name):
        raise StyleError(f"style names use lowercase letters, digits, '-' and '_' (up to 40), got {name!r}")
    return name


def _shape(base: RatingConditionedNet) -> dict[str, Any]:
    c = base.config
    return {"channels": c.channels, "residual_blocks": c.residual_blocks, "conditioning": c.conditioning}


class StyleLibrary:
    """The styles in one folder (``<name>.style.pt``)."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)

    def path(self, name: str) -> Path:
        return self.directory / f"{check_name(name)}.style.pt"

    def save(self, adapter: StyleAdapter, info: StyleInfo, base: RatingConditionedNet) -> Path:
        path = self.path(info.name)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"format_version": STYLE_FORMAT_VERSION, "info": info.to_dict(), "shape": _shape(base),
                   "adapter": adapter.state_dict()}
        tmp = path.with_name(path.name + ".tmp")
        torch.save(payload, tmp)
        tmp.replace(path)
        return path

    def _read(self, path: Path) -> dict[str, Any]:
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except FileNotFoundError:
            raise StyleError(f"no style called {path.name.removesuffix('.style.pt')!r}") from None
        except Exception as exc:  # noqa: BLE001 - torch raises many kinds of errors for bad files
            raise StyleError(f"{path.name} is not a readable style file") from exc
        if payload.get("format_version") != STYLE_FORMAT_VERSION:
            raise StyleError(f"{path.name}: style format {payload.get('format_version')}, expected "
                             f"{STYLE_FORMAT_VERSION}")
        return payload

    def info(self, name: str) -> StyleInfo:
        return StyleInfo(**self._read(self.path(name))["info"])

    def load(self, name: str, base: RatingConditionedNet) -> tuple[StyleAdapter, StyleInfo]:
        payload = self._read(self.path(name))
        if payload["shape"] != _shape(base):
            raise StyleError(f"style {name!r} was trained for a different model ({payload['shape']})")
        adapter = StyleAdapter.for_model(base)
        adapter.load_state_dict(payload["adapter"])
        return adapter, StyleInfo(**payload["info"])

    def list(self) -> list[StyleInfo]:
        """Every readable style, by name (unreadable files are skipped)."""
        out = []
        for path in sorted(self.directory.glob("*.style.pt")):
            try:
                out.append(StyleInfo(**self._read(path)["info"]))
            except (StyleError, TypeError):
                continue
        return out

    def delete(self, name: str) -> None:
        path = self.path(name)
        if not path.exists():
            raise StyleError(f"no style called {name!r}")
        path.unlink()
