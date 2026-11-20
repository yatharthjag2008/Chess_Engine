"""Smoke test: every module in the package (including placeholders) imports cleanly."""

from __future__ import annotations

import importlib
import pkgutil

import pytest

import chess_engine

EXPECTED_SUBPACKAGES = {"data", "encoding", "model", "training", "eval", "bot", "review", "api"}


def _all_module_names() -> list[str]:
    return sorted(
        info.name for info in pkgutil.walk_packages(chess_engine.__path__, prefix=f"{chess_engine.__name__}.")
    )


def test_version_is_set() -> None:
    assert chess_engine.__version__


def test_expected_subpackages_exist() -> None:
    found = {info.name for info in pkgutil.iter_modules(chess_engine.__path__) if info.ispkg}
    assert EXPECTED_SUBPACKAGES <= found


@pytest.mark.parametrize("module_name", _all_module_names())
def test_module_imports(module_name: str) -> None:
    importlib.import_module(module_name)


@pytest.mark.parametrize("package", ["chess", "numpy", "pandas", "pyarrow", "zstandard", "yaml", "torch", "fastapi"])
def test_third_party_dependencies_import(package: str) -> None:
    importlib.import_module(package)
