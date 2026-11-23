# Progress

See [PLAN.md](PLAN.md) for details on each step.

## Checklist

- [x] 1. Project setup
- [x] 2. Data download + streaming parser
- [x] 3. Board + move encoding
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
- Default Lichess month `2024-01` was a placeholder; step 2 chose `2026-10`.

## Step 2: Data download + streaming parser (done 2026-11-23)

**Built**
- `chess_engine.data.pgn_stream`: lazy PGN splitter (`iter_raw_games`), movetext tokenizer
  (`parse_movetext` → SAN moves, per-move `[%clk]` seconds, result), `TimeControl`/speed
  classification, `GameFilter` (built from config), `GameRecord`, `filter_games` and `FilterStats`.
- `chess_engine.data.sources`: open local `.pgn`/`.pgn.zst` or an HTTP(S) URL as a text stream
  decompressed on the fly; `HTTPRangeReader` resumes dropped connections with `Range` requests;
  `download_file` / `download_lichess_month` for full, resumable, SHA-256-verified downloads.
- `chess_engine.data.game_store`: `GameWriter` (Parquet, batched row groups, written to `.tmp`
  and renamed only when complete) and `prepare_games` (source → filter → Parquet + `.stats.json`).
- Scripts: `scripts/prepare_games.py`, `scripts/download_lichess.py`.
- Tests: an anonymized fixture of 16 real games from the 2026-10 dump (every speed + BOT games),
  plus a local HTTP test server with Range support and fault injection (dropped connections,
  503s, servers that ignore Range).

**Real sample run** (`python scripts/prepare_games.py`, default config, streamed from Lichess)

| | |
|---|---|
| Games read | 116,102 (first 36 MB of the ~31 GB 2026-10 file) |
| Games kept | 50,000 (43.1%), 3.54M positions, 10.1 MB Parquet |
| Time | 30 s on a laptop (parser alone: ~4,800 games/s) |
| Rejected: speed (not blitz) | 62,928 |
| Rejected: too short (< 20 plies) | 1,635 |
| Rejected: a player under 800 | 992 |
| Rejected: BOT account | 286 |
| Rejected: abandoned / rules infraction | 208 |
| Rejected: not rated | 53 |
| Time controls kept | 3+0 43%, 5+0 26%, 3+2 19%, 5+3 9% |
| Player-games per bucket | 800: 4.1k · 1000: 8.8k · 1200: 13.4k · 1400: 19.2k · 1600: 21.1k · 1800: 18.7k · 2000: 10.6k · 2200+: 4.2k |

Replaying the first 5,000 kept games with python-chess found 0 illegal moves.

**Decisions**
- **Month 2026-10** (latest complete month). Each file is ~30 GB, but sample mode only streams the
  start, so the month choice costs nothing on a laptop. More months can be listed in config.
- **Sample mode streams straight from the Lichess server** instead of downloading: 50k games need
  ~36 MB of a 31 GB file. Full runs use `download_lichess.py` first (resumable, checksum-verified),
  and `prepare_games.py` automatically uses a downloaded file in `data/raw/` when there is one.
- **Fast header-first filtering, no python-chess in step 2.** Headers are checked before the movetext
  is even tokenized, and moves are kept as SAN without replaying them (python-chess replay runs at
  ~550 games/s, ~9× slower than the parser). Step 4 has to replay every game anyway to get positions,
  so legality is checked there, once.
- **Moves stored as SAN lists, clocks as float seconds**, one row per game. SAN is what the file
  contains (no conversion cost) and is human-readable; step 4 converts while replaying.
- **BOT games excluded** (`exclude_titles: [BOT]`): the model should learn human moves only.
- **Clocks required** (`require_clocks: true`) and all-or-nothing per game: either every move has a
  `[%clk]` or the game is dropped, so step 4 can rely on a clock for every position.
- **"Rated" check**: Swiss events are named `Blitz swiss https://...` with no "Rated" prefix, so a game
  counts as rated if the event starts with "Rated" or it has a `RatingDiff` header. (The first real run
  wrongly rejected 589 rated Swiss games; now 53 remain, Swiss games without rating changes.)
- **Ratings**: both players must be ≥ `min_rating` (and ≤ `max_rating` if set); a `?` rating rejects.
- **Robust streaming**: dropped connections and 5xx errors are retried with exponential backoff and
  resumed at the exact byte, invalid UTF-8 is replaced instead of crashing, outputs are written to
  temporary files and renamed only when complete. Only a generic `User-Agent` is sent.
- **Test fixture is real data, anonymized**: player names and game ids replaced; the data is CC0.

## Step 3: Board + move encoding (done 2026-11-23)

**Built**
- `chess_engine.encoding.moves`: a fixed 1858-move vocabulary (`MOVE_VOCABULARY`), `move_to_index`,
  `index_to_move`, `legal_move_indices`, `legal_move_mask`, `masked_moves`.
- `chess_engine.encoding.board`: `encode_board` → `(20, 8, 8)` planes, `encode_boards` (batch),
  `decode_board` (inverse, for debugging and tests), `PLANE_NAMES` / `PLANE_INDEX`.
- `scripts/benchmark_encoding.py`: times each encoder on positions replayed from the step-2 output.
- Tests run on 312 positions: hand-picked edge cases (Kiwipete for both colours, promotions
  and underpromotions for both colours, en passant for both colours, partial castling rights,
  check, checkmate, stalemate, a 218-move position) plus seeded random games.

**Input planes** (all from the side to move's view)

| Planes | Content |
|---|---|
| 0–5 | my pawns, knights, bishops, rooks, queens, king |
| 6–11 | their pawns … king |
| 12 | white to move (all 1s or all 0s) |
| 13–16 | castling rights: my kingside, my queenside, their kingside, their queenside |
| 17 | en passant target square (only if the capture is legal) |
| 18 | fifty-move counter / 100 (capped at 1) |
| 19 | constant 1s |

**Speed** (`python scripts/benchmark_encoding.py --games 500`, 36,530 real positions, one laptop core):
`encode_board` 18 µs, `legal_move_mask` 96 µs (53 µs of it is python-chess generating legal moves),
`move_to_index` 1 µs, replaying a SAN move 26 µs. About 8,700 positions/s, so the 3.54M positions in
the 50k-game sample take ~7 minutes to encode on one core (step 4 can use several processes).

**Decisions**
- **Side-to-move perspective for both board and moves.** When Black is to move, the board is mirrored
  vertically and the colours swap roles, and Black's moves are mirrored the same way, so the network
  only ever learns "my move" and its policy head is shared between colours. Tested by checking that
  `board` and `board.mirror()` give identical piece planes and identical legal-move indices.
- **A white-to-move plane is still included.** Mirroring makes the colours look the same, but humans
  don't play them the same (opening repertoires, White's extra tempo), so the model can use it.
- **1858-move flat policy** (from→to pairs reachable by a queen or knight move, plus 66 underpromotions),
  the same size as Leela Chess Zero's, rather than AlphaZero's 8×8×73 = 4672 (most of which can never be
  legal). Queen promotion shares the plain from→to index: in any position only one piece can stand on
  the from-square, so there is no ambiguity, and `index_to_move` re-adds the queen promotion when a pawn
  reaches the last rank.
- **Castling = king moves two squares** (`e1g1`), python-chess's standard-chess convention. Chess960
  boards raise an error instead of being silently mis-encoded.
- **En passant only when the capture is legal.** python-chess sets `ep_square` after every double pawn
  push; encoding it only when a capture is possible makes identical positions encode identically.
- **No move history planes for now.** The position itself is enough to start with; history can be added
  as extra planes later without changing the move encoding.
- **Bitboards → planes with `numpy.unpackbits`** instead of looping over squares in Python.
- **`legal_move_mask` vectorized** with numpy lookup tables (121 → 96 µs per position); the rest is
  python-chess's move generator.
- **`encode_board` returns float32 by default**; `fifty_move_clock` is the only fractional plane, so the
  other 19 planes can be stored as bits/uint8 in step 4.
