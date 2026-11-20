# Human-Like Chess Engine

A chess engine that plays like a **human at a given rating**, not like a perfect machine.
Given a position and a target rating (800–2200+), a neural network trained on millions of real
Lichess blitz games predicts a probability for each legal move: *what would a 1400 actually play here?*
On top of that model sit practical training tools: a bot you can dial to any strength, rating
estimation, free game review, and a "your level vs. next level" coach.

> **Status:** early development. See [PROGRESS.md](PROGRESS.md) for what's done.

## Features (planned)

| | Feature | What it does |
|---|---|---|
| 1 | Human-like move model | Probability for every legal move, conditioned on rating |
| 2 | Custom Elo bot | Play the bot at any rating from 800 to 2200+ |
| 3 | Rating estimation | Estimate a player's rating (with a range) from their games |
| 4 | Game review | Best / Good / Inaccuracy / Mistake / Blunder, accuracy, key moments (Stockfish) |
| 5 | Your level vs. next level | What players at your rating play next to what players +200/+400 play, with evals |
| 6 | Tiered hints | Nudge → point → show move; hinted games are unrated |
| 7 | Adaptive difficulty | Bot rating adjusts between games or live (capped) |
| 8 | Style imitation | Bot plays like a specific player |
| 9 | Ghost mode | Play a past version of yourself, trained on your games up to a date |

## Tech stack

- **Python 3.11**, python-chess, PyTorch, NumPy, pandas, pyarrow (Parquet)
- **Stockfish** for analysis and baselines (installed separately)
- **FastAPI** backend and a lightweight HTML/JS frontend
- **pytest** for tests
- **Data:** the [Lichess open database](https://database.lichess.org/) (CC0), streamed directly from
  compressed `.pgn.zst` files, with a sample mode (first 50k games) for laptops

## Setup

### 1. Python environment

Requires Python 3.11+.

```bash
python3.11 -m venv .venv
# Windows (PowerShell):  .venv\Scripts\Activate.ps1
# macOS / Linux:         source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"
```

### 2. Stockfish

| OS | Install |
|---|---|
| Windows | `winget install --id Stockfish.Stockfish --source winget`, or download from [stockfishchess.org](https://stockfishchess.org/download/) and unzip |
| macOS | `brew install stockfish` |
| Ubuntu / Debian | `sudo apt install stockfish` (installs to `/usr/games/stockfish`) |
| Google Colab | `!apt-get install -y stockfish` |

The project looks for Stockfish in this order: the `STOCKFISH_PATH` environment variable,
`stockfish.path` in the config, your `PATH`, then common install locations. If it is not found
automatically, set the environment variable:

```bash
# Windows (persists for new terminals)
setx STOCKFISH_PATH "C:\Tools\stockfish\stockfish-windows-x86-64-avx2.exe"
# macOS / Linux
export STOCKFISH_PATH=/usr/games/stockfish
```

### 3. Check everything works

```bash
python scripts/check_setup.py   # Python, packages, config, Stockfish
pytest                          # Stockfish tests are skipped (with instructions) if it isn't installed
```

## Configuration

All settings live in YAML files in [`configs/`](configs/):

- [`default.yaml`](configs/default.yaml): every setting, laptop-friendly (sample mode on, small model)
- [`full.yaml`](configs/full.yaml): full-data GPU run; lists only what differs from the default

Config files are validated into typed objects, so a typo or a bad value fails immediately:

```python
from chess_engine.config import load_config

cfg = load_config("configs/full.yaml")
cfg.ratings.bucket_labels()   # ['800-999', '1000-1199', ..., '2200+']
cfg.data.sample.enabled       # False
```

## Project structure

```
configs/              YAML configs
src/chess_engine/
  config.py           typed config loading
  stockfish.py        find + verify Stockfish
  data/               streaming Lichess parser, dataset builder, game import
  encoding/           board -> tensor, move <-> index
  model/              rating-conditioned policy network
  training/           training loop
  eval/               accuracy, baselines, Elo validation, rating estimation
  bot/                move selection, hints, adaptive difficulty, styles
  review/             Stockfish game review, level comparison
  api/                FastAPI backend
web/                  frontend
scripts/              command-line entry points
tests/                pytest suite
reports/              generated plots and result tables
data/, checkpoints/   local only (git-ignored)
```

## Roadmap

The full 20-step build plan is in [PLAN.md](PLAN.md); progress and design decisions are in [PROGRESS.md](PROGRESS.md).

## Data & license notes

Game data comes from the Lichess open database, released under CC0. Stockfish is GPL-3.0 and is
installed separately (it is not bundled with this repo).
