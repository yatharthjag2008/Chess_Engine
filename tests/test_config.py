"""Tests for the YAML config system."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from chess_engine.config import (
    CONFIG_DIR,
    PROJECT_ROOT,
    Config,
    ConfigError,
    RatingsConfig,
    deep_merge,
    load_config,
)


def _write_yaml(path: Path, data: dict) -> Path:
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


# ----------------------------------------------------------------------------- loading


def test_default_config_loads(default_config: Config) -> None:
    assert default_config.data.time_control.speeds == ["blitz"]
    assert default_config.data.sample.enabled is True
    assert default_config.data.sample.max_games == 50_000
    assert default_config.stockfish.path is None


def test_default_data_filters(default_config: Config) -> None:
    filters = default_config.data.filters
    assert filters.exclude_titles == ["BOT"]
    assert filters.require_clocks is True
    assert "Abandoned" in filters.exclude_terminations
    assert default_config.data.lichess.verify_checksum is True


def test_paths_are_absolute_and_under_project_root(default_config: Config) -> None:
    for _, path in default_config.paths:
        assert path.is_absolute()
        assert path.is_relative_to(PROJECT_ROOT)
    assert default_config.paths.raw_dir == PROJECT_ROOT / "data" / "raw"


def test_every_config_file_in_configs_dir_is_valid() -> None:
    files = sorted(CONFIG_DIR.glob("*.yaml"))
    assert files, "expected at least one config file"
    for path in files:
        load_config(path)


def test_full_config_overrides_only_what_it_lists() -> None:
    full = load_config(CONFIG_DIR / "full.yaml")
    default = load_config()
    assert full.data.sample.enabled is False
    assert full.training.device == "cuda"
    # Untouched settings come from default.yaml.
    assert full.ratings == default.ratings
    assert full.data.filters == default.data.filters


def test_default_training_settings(default_config: Config) -> None:
    t = default_config.training
    assert t.device == "auto" and t.optimizer == "adamw" and t.lr_schedule == "cosine"
    assert t.bucket_balancing == "weight" and 0 < t.balance_power <= 1
    assert t.max_steps is None and t.max_minutes is None
    assert t.value_loss_weight > 0 and t.mixed_precision == "auto"


def test_training_limits_can_be_unset(tmp_path: Path) -> None:
    override = _write_yaml(tmp_path / "t.yaml", {"training": {"grad_clip_norm": None, "val_max_positions": None,
                                                            "max_steps": 10, "max_minutes": 1.5}})
    t = load_config(override).training
    assert t.grad_clip_norm is None and t.val_max_positions is None
    assert t.max_steps == 10 and t.max_minutes == 1.5


def test_config_path_relative_to_project_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)  # run from somewhere else entirely
    cfg = load_config("configs/full.yaml")
    assert cfg.data.sample.enabled is False


def test_override_file_is_merged(tmp_path: Path) -> None:
    override = _write_yaml(tmp_path / "small.yaml", {"data": {"sample": {"max_games": 100}}})
    cfg = load_config(override)
    assert cfg.data.sample.max_games == 100
    assert cfg.data.sample.enabled is True  # sibling key kept from default


def test_stockfish_relative_path_is_resolved(tmp_path: Path) -> None:
    override = _write_yaml(tmp_path / "sf.yaml", {"stockfish": {"path": "tools/stockfish"}})
    cfg = load_config(override)
    assert cfg.stockfish.path == (PROJECT_ROOT / "tools" / "stockfish").resolve()


def test_config_is_immutable(default_config: Config) -> None:
    with pytest.raises(Exception):
        default_config.training.batch_size = 1  # type: ignore[misc]


# ----------------------------------------------------------------------------- errors


def test_missing_file_raises_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml")


def test_unknown_key_is_rejected(tmp_path: Path) -> None:
    override = _write_yaml(tmp_path / "typo.yaml", {"training": {"batchsize": 32}})
    with pytest.raises(ConfigError, match="batchsize"):
        load_config(override)


def test_wrong_type_is_rejected(tmp_path: Path) -> None:
    override = _write_yaml(tmp_path / "bad.yaml", {"training": {"batch_size": "big"}})
    with pytest.raises(ConfigError, match="batch_size"):
        load_config(override)


def test_invalid_yaml_is_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "broken.yaml"
    bad.write_text("data: [unclosed", encoding="utf-8")
    with pytest.raises(ConfigError, match="parse"):
        load_config(bad)


@pytest.mark.parametrize(
    "override, message",
    [
        ({"data": {"split": {"train": 0.8, "val": 0.1, "test": 0.05}}}, "sum to 1"),
        ({"data": {"time_control": {"speeds": ["hyperbullet"]}}}, "speeds"),
        ({"data": {"filters": {"min_rating": 1500, "max_rating": 1200}}}, "max_rating"),
        ({"data": {"lichess": {"months": ["2024-13"]}}}, "month"),
        ({"data": {"lichess": {"months": ["Jan 2024"]}}}, "YYYY-MM"),
        ({"ratings": {"bucket_edges": [800, 1200, 1000]}}, "strictly increasing"),
        ({"training": {"device": "tpu"}}, "device"),
        ({"data": {"dataset": {"shard_size": 0}}}, "shard_size"),
        ({"data": {"dataset": {"skip_first_plies": -1}}}, "skip_first_plies"),
        ({"data": {"dataset": {"workers": 0}}}, "workers"),
        ({"model": {"conditioning": "concat"}}, "conditioning"),
        ({"model": {"policy_head": "attention"}}, "policy_head"),
        ({"model": {"channels": 0}}, "channels"),
        ({"training": {"lr_schedule": "step"}}, "lr_schedule"),
        ({"training": {"bucket_balancing": "oversample"}}, "bucket_balancing"),
        ({"training": {"balance_power": 1.5}}, "balance_power"),
        ({"training": {"mixed_precision": "fp8"}}, "mixed_precision"),
        ({"training": {"label_smoothing": 1.0}}, "label_smoothing"),
        ({"training": {"max_minutes": 0}}, "max_minutes"),
        ({"training": {"grad_clip_norm": 0}}, "grad_clip_norm"),
        ({"training": {"prefetch_batches": -1}}, "prefetch_batches"),
    ],
)
def test_invalid_values_are_rejected(tmp_path: Path, override: dict, message: str) -> None:
    path = _write_yaml(tmp_path / "bad.yaml", override)
    with pytest.raises(ConfigError, match=message):
        load_config(path)


# ----------------------------------------------------------------------------- deep_merge


def test_deep_merge_nested_and_does_not_mutate() -> None:
    base = {"a": {"b": 1, "c": 2}, "d": [1, 2]}
    override = {"a": {"c": 3}, "d": [9]}
    merged = deep_merge(base, override)
    assert merged == {"a": {"b": 1, "c": 3}, "d": [9]}
    assert base == {"a": {"b": 1, "c": 2}, "d": [1, 2]}


# ----------------------------------------------------------------------------- rating buckets


def test_default_rating_buckets(default_config: Config) -> None:
    ratings = default_config.ratings
    assert ratings.num_buckets == 8
    assert ratings.bucket_labels() == [
        "800-999", "1000-1199", "1200-1399", "1400-1599",
        "1600-1799", "1800-1999", "2000-2199", "2200+",
    ]


@pytest.mark.parametrize(
    "rating, expected",
    [(800, 0), (999, 0), (1000, 1), (1530, 3), (2199, 6), (2200, 7), (3100, 7)],
)
def test_bucket_index(default_config: Config, rating: int, expected: int) -> None:
    assert default_config.ratings.bucket_index(rating) == expected


def test_default_dataset_settings(default_config: Config) -> None:
    dataset = default_config.data.dataset
    assert dataset.skip_first_plies == 0
    assert dataset.min_clock_seconds == 0
    assert dataset.shard_size == 200_000
    assert dataset.workers is None
    assert 1 <= dataset.resolved_workers() <= 8


def test_default_model_settings(default_config: Config) -> None:
    model = default_config.model
    assert (model.architecture, model.channels, model.residual_blocks) == ("resnet", 64, 6)
    assert model.conditioning == "film" and model.policy_head == "conv"
    assert model.condition_on_opponent and model.use_clock and model.value_head


def test_dataset_workers_override(tmp_path: Path) -> None:
    path = _write_yaml(tmp_path / "w.yaml", {"data": {"dataset": {"workers": 3}}})
    assert load_config(path).data.dataset.resolved_workers() == 3


def test_bucket_index_below_lowest_edge_raises() -> None:
    with pytest.raises(ValueError, match="below"):
        RatingsConfig(bucket_edges=[800, 1000]).bucket_index(799)
