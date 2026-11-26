# Running the bot for yourself and others

Everything runs on one machine: a Python process (FastAPI + the model) that also serves the web pages. There is no
database and no account system. Games live in memory, and the few files it writes (your adaptive rating, reviews,
styles, cached imports) go to `data/`.

## 0. The public website (GitHub Pages)

[yatharthjag2008.github.io/Chess_Engine](https://yatharthjag2008.github.io/Chess_Engine/) lets anyone play the bot
in their browser: the network is recomputed in JavaScript from an exported copy of the model, and chess.js knows the
rules. It is built from `site/` by `.github/workflows/pages.yml` on every push to `main`.

- **After training a new model**, export it and commit the result: `python scripts/build_site.py --export` (writes
  `site/model/`). The site checks itself against PyTorch's numbers in every browser that loads it.
- **Opening names** come from `third_party/chess-openings/` (the Lichess data set, CC0); after updating those files
  run `python scripts/build_openings.py` (writes `site/data/openings.json`; a test checks it is up to date).
- **To look before pushing**: `python scripts/build_site.py`, then `python -m http.server -d _site 8001` and open
  `http://localhost:8001`.
- **One-time setup on GitHub**: Settings → Pages → Build and deployment → Source: **GitHub Actions** (as for any
  Pages site published by a workflow).

Only playing is on the website. Reviews, hints, the coach, adaptive difficulty and styles need Stockfish and the
Python server, so they run on your computer (below).

## 1. On your own computer

```bash
python -m venv .venv
.venv\Scripts\activate                 # Windows   (macOS / Linux: source .venv/bin/activate)
pip install -e ".[dev]"
python scripts/check_setup.py          # Python, PyTorch, Stockfish, data folders
python scripts/serve.py                # then open http://127.0.0.1:8000
```

- **The model.** `bot.model` in `configs/default.yaml` names the trained model file
  (`checkpoints/laptop_64x6/best_model.pt` after step 6). Use another one with `--model path/to/model.pt`.
  Model files are not in git: train one (`python scripts/train.py --config configs/laptop.yaml`) or copy one in.
- **Stockfish** is needed for reviews, hints and the coach (playing works without it). Install it (`winget install
  Stockfish.Stockfish`, `brew install stockfish`, `sudo apt install stockfish`) or set `STOCKFISH_PATH`.
- Without a GPU everything runs on the CPU. The 64×6 model answers a move in a few milliseconds.

## 2. For other devices on your network

```bash
python scripts/serve.py --host 0.0.0.0 --port 8000
```

Then open `http://<this computer's address>:8000` on a phone or another computer on the same network. The web
pages call the API on the same address, so CORS needs no change. Add an origin to `api.cors_origins` only when a
page served from somewhere else calls the API.

Your operating system may ask to allow incoming connections for Python. Allow them on private networks only.

## 3. On a server

The app has **no login**: anyone who can reach it can start games, read reviews and train styles. Keep it on a
private network, or put it behind a reverse proxy that adds authentication and HTTPS (for example Caddy or nginx
with basic auth).

A minimal setup on a Linux server:

```bash
git clone <your copy of this repository> && cd Human_Chess_Engine
python3.11 -m venv .venv && . .venv/bin/activate && pip install -e .
sudo apt install stockfish
# copy your model file to checkpoints/laptop_64x6/best_model.pt (or pass --model)
python scripts/serve.py --host 127.0.0.1 --port 8000      # behind the proxy
```

To keep it running, use a service manager (for example a systemd unit whose `ExecStart` is the last line above,
with `WorkingDirectory` set to the project). One process serves everyone, and games are kept in memory, so a
restart ends the games in progress (`api.idle_minutes` and `api.max_games` limit memory use).

These server steps follow from how the app works. They were not tested on a real server during development; the
local and network setups above were.

## 4. On Lichess (optional)

`scripts/lichess_bot.py` plays on a Lichess **BOT** account. You create the account and its token yourself:

1. Create a new Lichess account for the bot (it must never have played a game) and upgrade it to a BOT account
   ([Lichess Bot API](https://lichess.org/api#tag/Bot)).
2. Create an API token for it with the `bot:play` scope.
3. Put the token in the environment, not on the command line:
   `set LICHESS_BOT_TOKEN=lip_...` (Windows) or `export LICHESS_BOT_TOKEN=lip_...` (macOS / Linux).
4. `python scripts/lichess_bot.py --rating 1500`, or `--match-opponent --offset -100`, or `--style some-player`.

It accepts standard challenges at the speeds you allow and plays one game at a time until you press Ctrl+C. The
token is only sent to Lichess, in the request header, and is never printed. Lichess's rules for bots apply (the
account must be marked as a bot and must not play on behalf of a person). This integration is tested against a
fake Lichess only; it has not been run on a real account.
