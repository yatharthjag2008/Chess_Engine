# Progress

See [PLAN.md](PLAN.md) for details on each step.

## Checklist

- [x] 1. Project setup
- [ ] 2. Data download + streaming parser
- [ ] 3. Board + move encoding
- [ ] 4. Dataset builder
- [ ] 5. Model
- [ ] 6. Training pipeline
- [ ] 7. Evaluation + baselines
- [ ] 8. Bot move selection + CLI
- [ ] 9. Backend API
- [ ] 10. Web UI for playing
- [ ] 11. Elo dial validation
- [ ] 12. Rating estimation
- [ ] 13. Game import
- [ ] 14. Game review engine
- [ ] 15. "Your level vs. next level" + mistake coach
- [ ] 16. Review UI
- [ ] 17. Tiered hints
- [ ] 18. Adaptive difficulty
- [ ] 19. Style imitation + ghost mode
- [ ] 20. Polish

## Step 1: Project setup (done 2026-11-20)

**Built**
- `src/` layout package `chess_engine` with subpackages `data`, `encoding`, `model`, `training`,
  `eval`, `bot`, `review`, `api`. Placeholder modules (docstring only) name the step that fills them in.
- `pyproject.toml` is the single dependency list (installed with `pip install -e ".[dev]"`).
- Config system (`chess_engine.config`): YAML → validated, immutable pydantic models.
- Stockfish utility (`chess_engine.stockfish`): find → verify → clear error with OS-specific install steps.
- `scripts/check_setup.py`: one command that reports Python, packages, config and Stockfish status.
- Tests: imports of every module, config loading and validation, python-chess behaviour, Stockfish
  detection (using a fake UCI engine) and real-Stockfish tests that skip with instructions if it is missing.

**Decisions**
- **pyproject.toml rather than requirements.txt.** One source of truth, and the editable install makes
  `import chess_engine` work from scripts, tests and notebooks without `sys.path` hacks.
  Major versions are pinned to what was installed and tested (`>=tested,<next major`).
- **pandas 3.1.1 excluded.** On this Windows machine, Smart App Control blocked a DLL inside the
  pandas 3.1.1 wheel. 3.1.0 works. Remove the `!=3.1.1` once a newer release installs cleanly.
- **Config = default.yaml + override file.** `default.yaml` holds every setting (laptop-friendly:
  sample mode ON, small model). Other files (e.g. `full.yaml` for Colab/GPU) list only what changes
  and are deep-merged on top. Unknown keys are errors, so typos fail loudly.
- **pydantic for config validation** (rather than bare dataclasses): type checking, range checks and
  readable errors for free, and FastAPI (step 9) uses it anyway.
- **Relative paths resolve against the project root**, not the current directory, so scripts behave
  the same wherever they are launched from.
- **Rating buckets** are lower edges `[800, 1000, ..., 2200]`, giving 8 buckets:
  800–999, …, 2000–2199, 2200+. Games with a player under 800 are filtered out (`min_rating: 800`).
- **Time control** is filtered by Lichess speed category (`speeds: [blitz]`). Lichess defines the
  speed from base + 40 × increment: blitz is 180–479 s. The parser itself is step 2.
- **Excluded terminations**: `Abandoned`, `Rules infraction`, `Unterminated`.
- **Stockfish lookup order**: `STOCKFISH_PATH` env var → `stockfish.path` in config → `PATH` →
  common install locations. A wrong *explicit* path is an error rather than a silent fallback.
  The env var is the recommended way to set a machine-specific path, so committed configs stay portable.
- **Stockfish checks are tested with a fake UCI engine** (`tests/fixtures/fake_uci_engine.py`), so
  the verify logic is covered even on machines without Stockfish.
- Default Lichess month `2024-01` is a placeholder; the actual month(s) get chosen in step 2.
