// The play page: start a game, move on the board, show the bot's replies.
//
// All game state lives on the server; the page shows the latest state it received
// (render) and sends what the player does (moves, buttons) to the API.

import { api, ApiError } from "./api.js";
import { Board } from "./board.js";

const $ = (id) => document.getElementById(id);

export const ui = {
  game: null,        // the latest game state from the server
  busy: false,       // a request is in flight (the board is locked meanwhile)
  board: null,
};

export function setStatus(text, isError = false) {
  const el = $("status");
  el.textContent = text;
  el.style.color = isError ? "var(--danger)" : "";
}

function colorWord(color) {
  return color === "white" ? "White" : "Black";
}

/** The move list as numbered pairs; the latest move is marked. */
export function renderMoves(state) {
  const list = $("move-list");
  const items = [];
  const blackStarts = state.fen && state.moves_san.length % 2 === (state.turn === "white" ? 1 : 0);
  // When the game started from a Black-to-move position, the first row has an empty White cell.
  const sans = blackStarts ? [null, ...state.moves_san] : state.moves_san;
  for (let i = 0; i < sans.length; i += 2) {
    const li = document.createElement("li");
    const num = document.createElement("span");
    num.className = "num";
    num.textContent = `${i / 2 + 1}.`;
    li.append(num);
    for (const index of [i, i + 1]) {
      const cell = document.createElement("span");
      cell.className = "san";
      if (sans[index]) cell.textContent = sans[index];
      if (index === sans.length - 1) cell.classList.add("latest");
      li.append(cell);
    }
    items.push(li);
  }
  list.replaceChildren(...items);
  list.scrollTop = list.scrollHeight;
}

/** The bot's last move next to the moves players of its rating usually choose there. */
export function renderThinking(state) {
  const panel = $("thinking-panel");
  const move = state.last_bot_move;
  if (!move) {
    panel.hidden = true;
    return;
  }
  panel.hidden = false;
  $("thinking-caption").textContent =
    `In the position before ${move.san}, players rated about ${state.bot_rating} chose:`;
  const rows = move.candidates.map((c) => {
    const li = document.createElement("li");
    li.className = "candidate" + (c.uci === move.uci ? " played" : "");
    li.title = `${c.san}: ${(c.prob * 100).toFixed(1)}%`;
    const san = document.createElement("span");
    san.className = "san";
    san.textContent = c.san;
    const track = document.createElement("span");
    track.className = "track";
    const fill = document.createElement("span");
    fill.className = "fill";
    fill.style.width = `${Math.max(1, c.prob * 100)}%`;
    track.append(fill);
    const pct = document.createElement("span");
    pct.className = "pct";
    pct.textContent = `${Math.round(c.prob * 100)}%`;
    li.append(san, track, pct);
    return li;
  });
  if (!move.candidates.some((c) => c.uci === move.uci)) {
    const note = document.createElement("li");
    note.className = "hint";
    note.textContent = `It played ${move.san}, a rarer choice (${(move.prob * 100).toFixed(1)}%).`;
    rows.push(note);
  }
  $("candidates").replaceChildren(...rows);
}

/** Show a game state from the server. */
export function render(state) {
  ui.game = state;
  const board = ui.board;
  const humanToMove = state.status === "ongoing" && state.turn === state.human_color;
  board.setPosition(state.fen, { lastMove: state.last_move, check: state.check });
  board.setLegalMoves(humanToMove && !ui.busy ? state.legal_moves : [], humanToMove && !ui.busy ? state.turn : null);
  const bottomIsHuman = board.orientation === state.human_color;
  const bot = `Bot ${state.bot_rating}`;
  const you = state.your_rating ? `You (${state.your_rating})` : "You";
  $("top-name").textContent = bottomIsHuman ? bot : you;
  $("bottom-name").textContent = bottomIsHuman ? you : bot;
  $("game-panel").hidden = false;
  renderMoves(state);
  renderThinking(state);
  if (state.status === "over") {
    setStatus(state.outcome.description);
  } else if (ui.busy) {
    setStatus("The bot is thinking...");
  } else if (humanToMove) {
    setStatus(`Your move (${colorWord(state.turn)})${state.check ? ": you are in check" : ""}.`);
  } else {
    setStatus("Waiting for the bot.");
  }
}

function showError(err) {
  const message = err instanceof ApiError ? err.message : String(err);
  setStatus(message, true);
}

/** Send the player's move, show it, then ask for the bot's reply. */
export async function playMove(uci) {
  const game = ui.game;
  if (!game || ui.busy) return;
  ui.busy = true;
  let latest = game;
  let error = null;
  try {
    latest = (await api.play(game.id, uci, false)).state;
    render(latest);                                   // your move shows at once; the board stays locked
    if (latest.status === "ongoing") latest = (await api.botMove(game.id)).state;
  } catch (err) {
    error = err;
    latest = await api.state(game.id).catch(() => latest);  // resync with the server
  } finally {
    ui.busy = false;
  }
  render(latest);
  if (error) showError(error);
}

export function readForm(form) {
  const data = new FormData(form);
  const options = { rating: Number(data.get("rating")), color: data.get("color") };
  const time = data.get("time_control");
  if (time) options.time_control = time;
  return options;
}

export async function startGame(options) {
  if (ui.busy) return;
  ui.busy = true;
  setStatus("Starting a game...");
  try {
    const state = await api.newGame(options);
    ui.busy = false;
    ui.board.setOrientation(state.human_color);
    render(state);
  } catch (err) {
    ui.busy = false;
    showError(err);
  }
}

export function init() {
  ui.board = new Board($("board"), { promotionEl: $("promotion"), onMove: (uci) => playMove(uci) });
  ui.board.setLegalMoves([], null);
  const rating = $("rating");
  const output = $("rating-value");
  const showRating = () => { output.textContent = rating.value; };
  rating.addEventListener("input", showRating);
  showRating();
  $("new-game").addEventListener("submit", (event) => {
    event.preventDefault();
    startGame(readForm(event.target));
  });
  $("flip").addEventListener("click", () => {
    ui.board.flip();
    if (ui.game) render(ui.game);
  });
}

if (document.getElementById("board") && !window.__noAutoInit) init();
