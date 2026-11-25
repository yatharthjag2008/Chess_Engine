"""Typed project configuration loaded from YAML files in ``configs/``.

How it works
------------
* ``configs/default.yaml`` holds every setting and is always loaded first.
* Any other config file (e.g. ``configs/full.yaml``) only lists the settings that
  differ from the default; it is deep-merged on top of ``default.yaml``.
* The merged dictionary is validated into pydantic models, so a typo in a key
  name or a value of the wrong type fails immediately with a clear message.
* Relative paths are resolved against the project root, so scripts behave the
  same no matter which directory they are launched from.

Usage::

    from chess_engine.config import load_config

    cfg = load_config()                    # configs/default.yaml
    cfg = load_config("configs/full.yaml") # default.yaml + full.yaml overrides
    cfg.ratings.bucket_index(1530)         # -> 3
"""

from __future__ import annotations

import bisect
import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, PositiveInt, ValidationError, field_validator, model_validator

# src/chess_engine/config.py -> parents[0] = chess_engine, [1] = src, [2] = project root.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
CONFIG_DIR: Path = PROJECT_ROOT / "configs"
DEFAULT_CONFIG_PATH: Path = CONFIG_DIR / "default.yaml"

Speed = Literal["ultrabullet", "bullet", "blitz", "rapid", "classical"]


class ConfigError(Exception):
    """Raised when a config file is missing, is not valid YAML, or fails validation."""


class _StrictModel(BaseModel):
    """Base model: unknown keys are errors (catches typos) and configs are immutable."""

    model_config = ConfigDict(extra="forbid", frozen=True)


# --------------------------------------------------------------------------- sections


class PathsConfig(_StrictModel):
    """Where data, checkpoints and reports live. Relative paths are relative to the project root."""

    data_dir: Path
    raw_dir: Path
    processed_dir: Path
    checkpoints_dir: Path
    reports_dir: Path

    def resolved(self, root: Path) -> PathsConfig:
        """Return a copy with every relative path made absolute against ``root``."""
        return self.model_copy(update={name: _resolve(value, root) for name, value in self})


class StockfishConfig(_StrictModel):
    """Stockfish engine settings. ``path: null`` means "auto-detect" (see ``chess_engine.stockfish``)."""

    path: Path | None = None
    threads: PositiveInt
    hash_mb: PositiveInt
    analysis_depth: PositiveInt
    startup_timeout_seconds: float = Field(gt=0)


class LichessSourceConfig(_StrictModel):
    """Which Lichess open-database files to use."""

    base_url: str
    months: list[str] = Field(min_length=1)
    verify_checksum: bool = True

    @field_validator("months")
    @classmethod
    def _check_month_format(cls, months: list[str]) -> list[str]:
        for month in months:
            year, sep, mon = month.partition("-")
            if not (sep and len(year) == 4 and year.isdigit() and len(mon) == 2 and mon.isdigit()):
                raise ValueError(f"month {month!r} must look like 'YYYY-MM'")
            if not 1 <= int(mon) <= 12:
                raise ValueError(f"month {month!r} has an invalid month number")
        return months


class TimeControlConfig(_StrictModel):
    """Which Lichess speed categories to keep (e.g. ``["blitz"]``)."""

    speeds: list[Speed] = Field(min_length=1)


class GameFilterConfig(_StrictModel):
    """Game-level filters applied while streaming PGNs."""

    rated_only: bool
    min_rating: PositiveInt
    max_rating: PositiveInt | None = None
    min_plies: int = Field(ge=0)
    exclude_terminations: list[str]
    exclude_titles: list[str] = Field(default_factory=list)
    require_clocks: bool = True

    @model_validator(mode="after")
    def _check_rating_range(self) -> GameFilterConfig:
        if self.max_rating is not None and self.max_rating <= self.min_rating:
            raise ValueError(f"max_rating ({self.max_rating}) must be greater than min_rating ({self.min_rating})")
        return self


class SampleConfig(_StrictModel):
    """Sample mode: stop after ``max_games`` accepted games so everything runs on a laptop."""

    enabled: bool
    max_games: PositiveInt


class SplitConfig(_StrictModel):
    """Train/val/test fractions. Splitting is done by game, never by position."""

    train: float = Field(gt=0, lt=1)
    val: float = Field(ge=0, lt=1)
    test: float = Field(ge=0, lt=1)
    seed: int

    @model_validator(mode="after")
    def _check_sum(self) -> SplitConfig:
        total = self.train + self.val + self.test
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"split fractions must sum to 1.0, got {total:.4f}")
        return self


class DatasetConfig(_StrictModel):
    """How games become training positions (step 4)."""

    skip_first_plies: int = Field(default=0, ge=0)
    min_clock_seconds: float = Field(default=0, ge=0)
    shard_size: PositiveInt = 200_000
    games_per_task: PositiveInt = 500
    workers: PositiveInt | None = None

    def resolved_workers(self) -> int:
        """Number of encoding processes: ``workers``, or CPU count - 1 (1 to 8) when unset."""
        if self.workers is not None:
            return self.workers
        return max(1, min(8, (os.cpu_count() or 2) - 1))


class DataConfig(_StrictModel):
    """Everything about which games we use and how we split them."""

    lichess: LichessSourceConfig
    time_control: TimeControlConfig
    filters: GameFilterConfig
    sample: SampleConfig
    split: SplitConfig
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)


class RatingsConfig(_StrictModel):
    """Rating buckets defined by their lower edges; the last bucket is open-ended.

    With edges ``[800, 1000, ..., 2200]`` the buckets are
    ``800-999, 1000-1199, ..., 2000-2199, 2200+``.
    """

    bucket_edges: list[int] = Field(min_length=1)

    @field_validator("bucket_edges")
    @classmethod
    def _check_increasing(cls, edges: list[int]) -> list[int]:
        if any(b <= a for a, b in zip(edges, edges[1:])):
            raise ValueError(f"bucket_edges must be strictly increasing, got {edges}")
        return edges

    @property
    def num_buckets(self) -> int:
        """Number of rating buckets."""
        return len(self.bucket_edges)

    def bucket_index(self, rating: int) -> int:
        """Return the bucket index for ``rating``.

        Raises:
            ValueError: if ``rating`` is below the lowest bucket edge.
        """
        if rating < self.bucket_edges[0]:
            raise ValueError(f"rating {rating} is below the lowest bucket edge {self.bucket_edges[0]}")
        return bisect.bisect_right(self.bucket_edges, rating) - 1

    def bucket_labels(self) -> list[str]:
        """Human-readable labels, e.g. ``["800-999", ..., "2200+"]``."""
        edges = self.bucket_edges
        labels = [f"{low}-{high - 1}" for low, high in zip(edges, edges[1:])]
        labels.append(f"{edges[-1]}+")
        return labels


class ModelConfig(_StrictModel):
    """Rating-conditioned policy network hyperparameters (see ``chess_engine.model``)."""

    architecture: Literal["resnet"]
    channels: PositiveInt
    residual_blocks: PositiveInt
    rating_embedding_dim: PositiveInt
    conditioning: Literal["add", "film"] = "film"
    condition_on_opponent: bool = True
    use_clock: bool = True
    policy_head: Literal["conv", "fc"] = "conv"
    value_head: bool = True


class TrainingConfig(_StrictModel):
    """Training-loop settings (see ``chess_engine.training``)."""

    seed: int
    device: Literal["auto", "cpu", "cuda", "mps"]
    threads: PositiveInt | None = None
    batch_size: PositiveInt
    epochs: PositiveInt
    max_steps: PositiveInt | None = None
    max_minutes: float | None = Field(default=None, gt=0)

    optimizer: Literal["adamw", "sgd"] = "adamw"
    learning_rate: float = Field(gt=0)
    weight_decay: float = Field(ge=0)
    lr_schedule: Literal["constant", "cosine", "linear"] = "cosine"
    warmup_steps: int = Field(default=0, ge=0)
    min_lr_fraction: float = Field(default=0.0, ge=0, le=1)
    grad_clip_norm: float | None = Field(default=1.0, gt=0)
    mixed_precision: Literal["auto", "off", "bf16", "fp16"] = "auto"

    value_loss_weight: float = Field(default=0.25, ge=0)
    label_smoothing: float = Field(default=0.0, ge=0, lt=1)
    bucket_balancing: Literal["none", "weight", "sample"] = "weight"
    balance_power: float = Field(default=0.5, ge=0, le=1)

    in_memory: bool = True
    max_train_shards: PositiveInt | None = None
    prefetch_batches: int = Field(default=4, ge=0)

    log_every_steps: PositiveInt
    eval_every_steps: PositiveInt = 2000
    val_max_positions: PositiveInt | None = 50_000
    checkpoint_every_steps: PositiveInt
    keep_checkpoints: PositiveInt = 3


class EvaluationConfig(_StrictModel):
    """Evaluation settings (see ``chess_engine.eval``)."""

    split: Literal["train", "val", "test"] = "test"
    max_positions: PositiveInt | None = None
    batch_size: PositiveInt = 1024
    bootstrap_samples: int = Field(default=1000, ge=0)
    confidence: float = Field(default=0.95, gt=0, lt=1)
    calibration_bins: PositiveInt = 10
    cross_rating_positions: int = Field(default=40_000, ge=0)
    stockfish_positions: int = Field(default=10_000, ge=0)
    stockfish_depth: PositiveInt = 10
    seed: int = 0


class BotConfig(_StrictModel):
    """How the bot picks its moves (see ``chess_engine.bot``)."""

    model: Path = Path("checkpoints/laptop_64x6/best_model.pt")
    rating: PositiveInt = 1500
    temperature: float = Field(default=1.0, ge=0)
    top_k: PositiveInt | None = None
    top_p: float = Field(default=1.0, gt=0, le=1)
    min_prob: float = Field(default=0.0, ge=0, lt=1)
    seed: int | None = None


class ApiConfig(_StrictModel):
    """FastAPI backend settings (see ``chess_engine.api``)."""

    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    max_games: PositiveInt = 200
    idle_minutes: float = Field(default=120, gt=0)
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:8000", "http://127.0.0.1:8000"])


class EloValidationConfig(_StrictModel):
    """Matches that check the rating dial (see ``chess_engine.eval.elo_validation``)."""

    bot_ratings: list[PositiveInt] = Field(default_factory=lambda: [800, 1100, 1400, 1700, 2000, 2300], min_length=2)
    stockfish_elos: list[int] = Field(default_factory=lambda: [1320, 1600, 1900])
    stockfish_move_seconds: float = Field(default=0.05, gt=0)
    games_per_pairing: PositiveInt = 20
    self_play: Literal["none", "adjacent", "all"] = "adjacent"
    temperature: float | None = Field(default=None, ge=0)
    max_plies: PositiveInt = 300
    adjudicate_cp: PositiveInt = 800
    adjudicate_plies: PositiveInt = 8
    judge_nodes: PositiveInt = 20_000
    seed: int = 0

    @field_validator("stockfish_elos")
    @classmethod
    def _check_stockfish_elos(cls, elos: list[int]) -> list[int]:
        for elo in elos:
            if not 1320 <= elo <= 3190:
                raise ValueError(f"Stockfish UCI_Elo must be between 1320 and 3190, got {elo}")
        return elos


class Config(_StrictModel):
    """The full project configuration."""

    paths: PathsConfig
    stockfish: StockfishConfig
    data: DataConfig
    ratings: RatingsConfig
    model: ModelConfig
    training: TrainingConfig
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    bot: BotConfig = Field(default_factory=BotConfig)
    api: ApiConfig = Field(default_factory=ApiConfig)
    elo_validation: EloValidationConfig = Field(default_factory=EloValidationConfig)


# --------------------------------------------------------------------------- loading


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into a copy of ``base``.

    Nested dicts are merged key by key; any other value (including lists) in
    ``override`` replaces the value in ``base``. Neither input is modified.
    """
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def read_yaml(path: Path) -> dict[str, Any]:
    """Read a YAML file that must contain a mapping (an empty file counts as ``{}``)."""
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"Could not parse YAML in {path}:\n{exc}") from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"Top level of {path} must be a mapping, got {type(data).__name__}")
    return data


def load_config(path: str | Path | None = None, root: Path = PROJECT_ROOT) -> Config:
    """Load ``default.yaml``, merge ``path`` on top of it (if given) and validate.

    Args:
        path: Optional override config file. Relative paths are tried against the
            current directory first, then the project root.
        root: Directory that relative paths inside the config are resolved against.

    Returns:
        A validated, immutable :class:`Config` with absolute paths.

    Raises:
        ConfigError: if a file is missing, unparsable, or fails validation.
    """
    raw = read_yaml(DEFAULT_CONFIG_PATH)
    source = str(DEFAULT_CONFIG_PATH)

    if path is not None:
        override_path = _find_config_file(Path(path), root)
        if override_path.resolve() != DEFAULT_CONFIG_PATH.resolve():
            raw = deep_merge(raw, read_yaml(override_path))
            source = f"{DEFAULT_CONFIG_PATH} + {override_path}"

    try:
        config = Config.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(f"Invalid configuration ({source}):\n{exc}") from exc

    stockfish_path = config.stockfish.path
    return config.model_copy(
        update={
            "paths": config.paths.resolved(root),
            "stockfish": config.stockfish.model_copy(
                update={"path": _resolve(stockfish_path, root) if stockfish_path else None}
            ),
            "bot": config.bot.model_copy(update={"model": _resolve(config.bot.model, root)}),
        }
    )


def _find_config_file(path: Path, root: Path) -> Path:
    """Locate a config file given as absolute, relative to CWD, or relative to the project root."""
    if path.is_absolute() or path.exists():
        return path
    return root / path


def _resolve(path: Path, root: Path) -> Path:
    """Make ``path`` absolute against ``root`` (``~`` is expanded first)."""
    path = path.expanduser()
    return path if path.is_absolute() else (root / path).resolve()
