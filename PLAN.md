# Build Plan

A human-like chess engine that predicts and plays moves the way real players at a given
rating do, plus training tools built on top of that model. Progress and decisions are
tracked in [PROGRESS.md](PROGRESS.md).

## Project features (full scope)

1. **Human-like move model**: given a position and a target rating, predict a probability for each legal move (trained on real Lichess games).
2. **Customizable Elo bot**: play against the bot at any rating you choose (e.g. 800–2200).
3. **Rating estimation**: estimate a player's rating from their game(s), with a likely range.
4. **Free unlimited game review / mistake coach**: classify each move (Best / Good / Inaccuracy / Mistake / Blunder) using Stockfish, and show accuracy and key moments.
5. **"Your level vs. next level"**: for each position, show the top moves players at your rating usually play next to what players 200–400 points higher play, with Stockfish evals for each.
6. **Tiered hints** (optional, toggleable): Nudge → Point (highlight piece/square) → Show move with a one-line reason. Games with hints count as unrated.
7. **Adaptive difficulty** (optional): the bot's rating adjusts between games (Elo-style update) or live within a game (capped).
8. **Style imitation**: the bot plays like a specific player (fine-tune / player embedding).
9. **Ghost mode**: play against a past version of yourself, trained on your own games up to a cutoff date.

## Tech stack

- Python 3.11, python-chess, PyTorch, NumPy, pandas, pyarrow (Parquet)
- Stockfish (installed separately, path in config)
- FastAPI backend, simple web frontend (HTML/JS with an open-source chessboard library)
- pytest for tests
- Data: Lichess open database (database.lichess.org, CC0). Monthly files are huge, so the pipeline
  **streams** `.pgn.zst` files and supports a small sample mode (e.g. the first 50k games) so everything
  runs on a laptop. Full training may run on Google Colab or a university GPU cluster, so training
  scripts are runnable from the command line with config files.
- Blitz games only to start. Rating buckets every 200 points from 800 to 2200+.

## Steps

1. **Project setup**: repo structure, dependencies, config system, Stockfish path detection, .gitignore, README skeleton, pytest smoke tests.
2. **Data download + streaming parser**: stream Lichess `.pgn.zst`, filter by time control (blitz), rating range, minimum game length, and no abandoned games; sample mode.
3. **Board + move encoding**: position → tensor (piece planes, side to move, castling, en passant), move ↔ index mapping, legal-move mask. Thoroughly unit tested.
4. **Dataset builder**: turn games into (position, move played, player rating bucket, clock time) samples; save as Parquet/NumPy shards; split train/val/test **by game** (no leakage).
5. **Model**: rating-conditioned neural net (CNN/ResNet-style to start, with a rating embedding added to the board features); output = move logits with illegal moves masked.
6. **Training pipeline**: config-driven training loop, checkpointing, resume, metrics logging; works on CPU (small) and GPU.
7. **Evaluation + baselines**: move-match accuracy per rating bucket compared with baselines (Stockfish top move, most-common move); calibration chart; plots saved to `/reports`.
8. **Bot move selection**: sample from the model at a chosen rating (with temperature), plus a terminal CLI to play the bot.
9. **Backend API (FastAPI)**: endpoints for new game, bot move, legal moves, game state.
10. **Web UI for playing**: board, rating selector, move list, game-over handling.
11. **Elo dial validation**: play the bot at different ratings against Stockfish skill levels and against itself, and check that strength increases with the dial; report results.
12. **Rating estimation**: combine per-move likelihoods across rating buckets over a game → estimate + range; evaluate average error on held-out games.
13. **Game import**: paste a PGN, or fetch recent games by username from the Lichess and Chess.com public APIs.
14. **Game review engine**: Stockfish analysis per move at a fixed depth, centipawn loss, move classification, accuracy score, top 3–5 key moments.
15. **"Your level vs. next level" + mistake coach**: run the model at your rating and at +200/+400, compare top moves, and flag only the moments where the levels disagree AND the eval difference matters; template-based explanations.
16. **Review UI**: step through a reviewed game with classifications, eval graph, level-comparison panel, and summary (accuracy, estimated performance rating, "next-level match %").
17. **Tiered hints**: nudge/point/show using Stockfish plus simple move features (captures, checks, hanging pieces); vagueness based on how often players at your rating find the move; hint-limit setting; hinted games marked unrated.
18. **Adaptive difficulty**: off / between games / live (capped) modes; the bot's current rating shown on screen.
19. **Style imitation + ghost mode**: player embeddings or small fine-tuned adapters; train on a famous player's games or on your own games up to a cutoff date; blend with the base rating model when data is small (fewer than ~200 games); select styles/ghosts in the UI.
20. **Polish**: full README with results tables and charts, test-coverage pass, cleanup, simple deployment instructions, and an optional Lichess bot account integration.

## Where each step lives in the code

| Step(s) | Location |
|---|---|
| 1 | `pyproject.toml`, `configs/`, `src/chess_engine/config.py`, `src/chess_engine/stockfish.py`, `scripts/check_setup.py` |
| 2 | `src/chess_engine/data/pgn_stream.py`, `sources.py`, `game_store.py`, `scripts/prepare_games.py`, `scripts/download_lichess.py` |
| 3 | `src/chess_engine/encoding/board.py`, `moves.py`, `scripts/benchmark_encoding.py` |
| 4 | `src/chess_engine/data/splits.py`, `samples.py`, `shards.py`, `dataset_builder.py`, `dataset.py`, `src/chess_engine/encoding/compact.py`, `scripts/build_dataset.py`, `scripts/inspect_dataset.py` |
| 13 | `src/chess_engine/data/game_import.py`, `scripts/import_games.py`, import + rating-estimate endpoints in `src/chess_engine/api/app.py` |
| 5 | `src/chess_engine/model/` (`features.py`, `layers.py`, `policy_map.py`, `heads.py`, `network.py`, `masking.py`, `inference.py`), `scripts/model_summary.py` |
| 6 | `src/chess_engine/training/`, `src/chess_engine/plotting.py`, `scripts/train.py`, `scripts/plot_training.py`, `configs/laptop.yaml` |
| 7, 11, 12 | `src/chess_engine/eval/`, `scripts/evaluate.py`, `scripts/compare_models.py`, `scripts/elo_validation.py`, `scripts/estimate_rating.py`, `scripts/evaluate_rating_estimation.py`, `reports/eval/`, `reports/elo/`, `reports/rating/` |
| 8, 17, 18, 19 | `src/chess_engine/bot/`, `scripts/play.py`, `scripts/bot_match.py` |
| 14, 15 | `src/chess_engine/review/`, `scripts/review_game.py`, review endpoints in `src/chess_engine/api/app.py` |
| 9 | `src/chess_engine/api/` (`schemas`, `store`, `service`, `app`), `scripts/serve.py` |
| 10, 16 | `web/` (`index.html`, `css/`, `js/board.js`, `js/api.js`, `js/app.js`, `tests/`), `tests/test_web_js.py` |
