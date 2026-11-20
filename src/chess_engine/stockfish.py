"""Find and verify the Stockfish engine binary.

Search order used by :func:`find_stockfish` (first match wins):

1. The ``STOCKFISH_PATH`` environment variable (handy on Colab / clusters).
2. ``stockfish.path`` from the config file.
3. A ``stockfish`` executable on your ``PATH``.
4. A few common install locations (apt, Homebrew, winget).

If an *explicit* path (1 or 2) is set but wrong, we raise instead of silently
falling back, because that almost always means a typo you want to know about.

Usage::

    from chess_engine.config import load_config
    from chess_engine.stockfish import get_stockfish

    info = get_stockfish(load_config())   # find + verify, raises StockfishError
    print(info.name, info.path)
"""

from __future__ import annotations

import glob
import os
import platform
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import chess
import chess.engine

from chess_engine.config import Config

ENV_VAR = "STOCKFISH_PATH"

# Places Stockfish commonly ends up that are not always on PATH (glob patterns allowed).
COMMON_LOCATIONS: tuple[str, ...] = (
    "/usr/games/stockfish",  # Debian / Ubuntu / Colab (apt install stockfish)
    "/usr/local/bin/stockfish",  # Homebrew on Intel macs, manual installs
    "/opt/homebrew/bin/stockfish",  # Homebrew on Apple Silicon
    "~/AppData/Local/Microsoft/WinGet/Links/stockfish*.exe",  # winget shortcut
    "~/AppData/Local/Microsoft/WinGet/Packages/Stockfish.Stockfish_*/**/stockfish*.exe",  # winget package
)


class StockfishError(RuntimeError):
    """Stockfish could not be found or did not respond correctly."""


@dataclass(frozen=True)
class EngineInfo:
    """Result of a successful engine check."""

    path: str
    name: str
    test_move: str


def install_instructions(system: str | None = None) -> str:
    """Return OS-specific instructions for installing Stockfish and pointing the project at it."""
    system = system or platform.system()
    if system == "Windows":
        install = (
            "  Option A: winget install --id Stockfish.Stockfish --source winget   (then open a NEW terminal)\n"
            "  Option B: download the Windows build from https://stockfishchess.org/download/ ,\n"
            "            unzip it (e.g. to C:\\Tools\\stockfish\\) and note the path of the .exe inside."
        )
        set_path = '  setx STOCKFISH_PATH "C:\\Tools\\stockfish\\stockfish-windows-x86-64-avx2.exe"'
    elif system == "Darwin":
        install = "  brew install stockfish"
        set_path = "  export STOCKFISH_PATH=/opt/homebrew/bin/stockfish   # add to ~/.zshrc to keep it"
    else:
        install = (
            "  sudo apt install stockfish        (Debian/Ubuntu; installs to /usr/games/stockfish)\n"
            "  !apt-get install -y stockfish     (Google Colab)"
        )
        set_path = "  export STOCKFISH_PATH=/usr/games/stockfish   # add to ~/.bashrc to keep it"
    return (
        "Install Stockfish:\n"
        f"{install}\n"
        "If it is not found automatically, tell the project where it is, either with an env variable:\n"
        f"{set_path}\n"
        "or by setting `stockfish.path` in your config YAML (e.g. configs/default.yaml)."
    )


def find_stockfish(configured_path: str | Path | None = None) -> Path:
    """Locate the Stockfish executable without running it.

    Args:
        configured_path: ``stockfish.path`` from the config (``None`` = auto-detect).

    Returns:
        Absolute path to an existing file.

    Raises:
        StockfishError: if an explicit path does not exist, or nothing is found.
    """
    env_value = os.environ.get(ENV_VAR, "").strip()
    explicit: list[tuple[str, str]] = []
    if env_value:
        explicit.append((f"the {ENV_VAR} environment variable", env_value))
    if configured_path:
        explicit.append(("`stockfish.path` in the config", str(configured_path)))

    if explicit:
        source, value = explicit[0]
        path = Path(value).expanduser()
        if path.is_file():
            return path.resolve()
        # A bare command name like "stockfish" is allowed and looked up on PATH.
        on_path = shutil.which(value)
        if on_path:
            return Path(on_path).resolve()
        raise StockfishError(
            f"Stockfish path from {source} does not exist: {value}\n\n{install_instructions()}"
        )

    on_path = shutil.which("stockfish")
    if on_path:
        return Path(on_path).resolve()

    for pattern in COMMON_LOCATIONS:
        for match in sorted(glob.glob(os.path.expanduser(pattern), recursive=True)):
            if Path(match).is_file():
                return Path(match).resolve()

    raise StockfishError(
        "Stockfish was not found (checked STOCKFISH_PATH, the config, your PATH and common "
        f"install locations).\n\n{install_instructions()}"
    )


def check_uci_engine(command: str | Sequence[str], timeout: float = 10.0) -> EngineInfo:
    """Start a UCI engine, ask it for one move from the starting position, then shut it down.

    Args:
        command: Executable path, or a full command line as a list of arguments.
        timeout: Seconds to wait for the engine to start and answer.

    Returns:
        The engine's reported name and the move it played.

    Raises:
        StockfishError: if the engine cannot be started or misbehaves.
    """
    argv = [command] if isinstance(command, str) else list(command)
    shown = " ".join(argv)
    board = chess.Board()
    try:
        with chess.engine.SimpleEngine.popen_uci(argv, timeout=timeout) as engine:
            result = engine.play(board, chess.engine.Limit(depth=1, time=timeout))
            name = engine.id.get("name", "unknown engine")
    except chess.engine.EngineError as exc:
        # The engine started but broke the UCI protocol (e.g. answered an illegal move).
        raise StockfishError(f"Engine `{shown}` did not behave like a UCI engine: {exc}") from exc
    except (OSError, TimeoutError) as exc:
        # OSError covers "file not found", "permission denied" and "wrong CPU build".
        raise StockfishError(
            f"Could not run the engine `{shown}`: {type(exc).__name__}: {exc}\n"
            "If you downloaded Stockfish manually, make sure the path points to the executable itself "
            "and that the build matches your CPU (try the plain 'x86-64' build if AVX2 fails)."
        ) from exc

    if result.move is None or result.move not in board.legal_moves:
        raise StockfishError(f"Engine `{shown}` did not return a legal move (got {result.move}).")
    return EngineInfo(path=shown, name=name, test_move=result.move.uci())


def verify_stockfish(path: str | Path, timeout: float = 10.0) -> EngineInfo:
    """Run the Stockfish binary at ``path`` and confirm it speaks UCI and plays a legal move."""
    return check_uci_engine(str(path), timeout=timeout)


def get_stockfish(config: Config) -> EngineInfo:
    """Find Stockfish using ``config`` and verify that it works."""
    path = find_stockfish(config.stockfish.path)
    return verify_stockfish(path, timeout=config.stockfish.startup_timeout_seconds)
