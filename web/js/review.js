// The review page (step 16): choose a game, review it with Stockfish, step through it.
//
// Flow: source (pasted PGN / Lichess / Chess.com) -> list of games -> pick a game and your side ->
// review (and coach) from the API -> board, move list, eval graph, summary, coach notes.

import { api, ApiError } from "./api.js";
import { Board, START_FEN } from "./board.js";

const $ = (id) => document.getElementById(id);

export const ui = {
  board: null,
  games: [],          // imported games (summaries + pgn)
  username: null,     // set when the games came from a site, to find the player's side
  game: null,         // the chosen game
  side: "white",      // whose game this is ("your" moves for the coach)
  review: null,       // the review from the API
  coach: null,        // the coach report from the API
  ply: 0,             // 0 = start position, n = after move n
  pgn: null,          // the PGN that was reviewed
};

export function setStatus(text, isError = false, busy = false) {
  const el = $("status");
  el.replaceChildren();
  if (busy) {
    const spin = document.createElement("span");
    spin.className = "loading";
    el.append(spin);
  }
  el.append(text);
  el.style.color = isError ? "var(--danger)" : "";
}

export function showError(err) {
  setStatus(err instanceof ApiError ? err.message : String(err), true);
}

// ------------------------------------------------------------------ choosing a game

export function renderSourceForm(source) {
  const form = $("source-form");
  form.replaceChildren();
  form.dataset.source = source;
  if (source === "pgn") {
    const area = document.createElement("textarea");
    area.name = "pgn";
    area.placeholder = "Paste one or more games in PGN";
    area.setAttribute("aria-label", "PGN");
    form.append(area);
  } else {
    const input = document.createElement("input");
    input.type = "text";
    input.name = "username";
    input.placeholder = source === "lichess" ? "Lichess username" : "Chess.com username";
    input.setAttribute("aria-label", "Username");
    form.append(input);
  }
  const submit = document.createElement("button");
  submit.type = "submit";
  submit.className = "primary";
  submit.textContent = source === "pgn" ? "Load games" : "Fetch recent games";
  form.append(submit);
}

export async function loadGames(source, value) {
  setStatus(source === "pgn" ? "Reading the PGN..." : `Fetching games from ${source}...`, false, true);
  try {
    const result = source === "pgn" ? await api.importPgn(value) : await api.importUser(source, value.trim());
    ui.games = result.games;
    ui.username = result.username;
    renderGameList();
    setStatus(ui.games.length ? `${ui.games.length} game${ui.games.length === 1 ? "" : "s"} found: pick one to review.`
      : "No standard chess games found.", !ui.games.length);
  } catch (err) {
    showError(err);
  }
}

/** The side ``username`` played in ``game`` (case-insensitive), or null. */
export function sideOf(game, username) {
  if (!username) return null;
  const name = username.toLowerCase();
  if (game.white.toLowerCase() === name) return "white";
  if (game.black.toLowerCase() === name) return "black";
  return null;
}

function describeGame(game) {
  const elo = (r) => (r ? ` (${r})` : "");
  return [`${game.white}${elo(game.white_elo)} - ${game.black}${elo(game.black_elo)}`,
          `${game.result} · ${game.speed || "?"} · ${game.date || ""} · ${Math.ceil(game.plies / 2)} moves`];
}

export function renderGameList() {
  const list = $("game-list");
  list.replaceChildren();
  list.hidden = !ui.games.length;
  ui.games.forEach((game, index) => {
    const [players, meta] = describeGame(game);
    const button = document.createElement("button");
    button.type = "button";
    button.dataset.index = String(index);
    const top = document.createElement("span");
    top.textContent = players;
    const bottom = document.createElement("span");
    bottom.className = "meta";
    bottom.textContent = meta;
    button.append(top, bottom);
    button.addEventListener("click", () => chooseGame(index));
    list.append(button);
  });
}

/** Pick a game: your side is found from the username when the games came from a site. */
export function chooseGame(index, side = null) {
  ui.game = ui.games[index];
  ui.side = side || sideOf(ui.game, ui.username) || "white";
  for (const button of $("game-list").querySelectorAll("button[data-index]")) {
    button.setAttribute("aria-pressed", String(button.dataset.index === String(index)));
  }
  renderSideChoice();
  return ui.game;
}

/** "I played White / Black" and the Review button, under the game list. */
export function renderSideChoice() {
  let box = document.querySelector("#source-panel .side-choice");
  if (!box) {
    box = document.createElement("div");
    box.className = "side-choice";
    $("game-list").after(box);
  }
  box.replaceChildren();
  const label = document.createElement("span");
  label.textContent = "I played";
  box.append(label);
  const group = document.createElement("div");
  group.className = "segmented";
  for (const side of ["white", "black"]) {
    const option = document.createElement("label");
    const radio = document.createElement("input");
    radio.type = "radio";
    radio.name = "side";
    radio.value = side;
    radio.checked = ui.side === side;
    radio.addEventListener("change", () => { ui.side = side; });
    const text = document.createElement("span");
    text.textContent = `${side === "white" ? "White" : "Black"} (${ui.game[side]})`;
    option.append(radio, text);
    group.append(option);
  }
  const go = document.createElement("button");
  go.type = "button";
  go.className = "primary";
  go.id = "review-button";
  go.textContent = "Review";
  go.addEventListener("click", () => startReview());
  box.append(group, go);
}

// ------------------------------------------------------------------ the review

export const SYMBOLS = { inaccuracy: "?!", mistake: "?", blunder: "??" };

/** Review the chosen game (or ``pgn``) and show it from the start. */
export async function startReview(pgn = ui.game && ui.game.pgn) {
  setStatus("Stockfish is reviewing the game (this takes a few seconds per move)...", false, true);
  try {
    ui.review = await api.review(pgn);
  } catch (err) {
    showError(err);
    return null;
  }
  ui.pgn = pgn;
  showReview();
  return ui.review;
}

export function showReview() {
  const review = ui.review;
  ui.board.setOrientation(ui.side);
  const headers = review.headers || {};
  const name = (color) => `${headers[color === "white" ? "White" : "Black"] || (color === "white" ? "White" : "Black")}`;
  $("top-name").textContent = name(ui.side === "white" ? "black" : "white");
  $("bottom-name").textContent = name(ui.side);
  renderMoveList();
  $("moves-panel").hidden = false;
  $("stepper").hidden = false;
  setStatus(`Reviewed ${review.moves.length} moves at depth ${review.depth}.`);
  goTo(0);
}

// ------------------------------------------------------------------ stepping

/** The position after ``ply`` moves (0 = the start). */
export function fenAt(ply) {
  return ply === 0 ? ui.review.start_fen : ui.review.moves[ply - 1].fen_after;
}

/** Show the position after ``ply`` moves: board, highlighted move, label, and (later) the panels. */
export function goTo(ply) {
  if (!ui.review) return;
  const moves = ui.review.moves;
  ui.ply = Math.max(0, Math.min(moves.length, ply));
  const move = ui.ply ? moves[ui.ply - 1] : null;
  const fen = fenAt(ui.ply);
  const check = move && /[+#]$/.test(move.san);
  ui.board.setPosition(fen, { lastMove: move ? move.uci : null, check });
  for (const cell of $("move-list").querySelectorAll(".san[data-ply]")) {
    const current = cell.dataset.ply === String(ui.ply);
    cell.classList.toggle("current", current);
    if (current) cell.setAttribute("aria-current", "step"); else cell.removeAttribute("aria-current");
    if (current && cell.scrollIntoView) cell.scrollIntoView({ block: "nearest" });
  }
  $("ply-label").textContent = move ? `${move.move_number} ${move.san}` : "Start";
  $("first").disabled = $("prev").disabled = ui.ply === 0;
  $("next").disabled = $("last").disabled = ui.ply === moves.length;
  for (const listener of onStep) listener(ui.ply, move);
}

/** Called after every step with (ply, move or null); the panels register here. */
export const onStep = [];

export function step(delta) { goTo(ui.ply + delta); }

function onKey(event) {
  if (!ui.review) return;
  const target = event.target;
  if (target && target.closest && target.closest("input, textarea, select")) return;
  const actions = { ArrowLeft: () => step(-1), ArrowRight: () => step(1), Home: () => goTo(0),
                    End: () => goTo(ui.review.moves.length) };
  if (actions[event.key]) {
    event.preventDefault();
    actions[event.key]();
  }
}

/** The move list, one cell per move, each with its class (and symbol for the errors). */
export function renderMoveList() {
  const list = $("move-list");
  const moves = ui.review.moves;
  const items = [];
  let row = null;
  moves.forEach((move, index) => {
    if (!row || move.color === "white") {
      row = document.createElement("li");
      const num = document.createElement("span");
      num.className = "num";
      num.textContent = `${move.move_number.replace(/\.+$/, "")}.`;
      row.append(num);
      if (move.color === "black") row.append(document.createElement("span"));   // Black to move first
      items.push(row);
    }
    const cell = document.createElement("span");
    cell.className = `san ${move.classification}`;
    cell.dataset.ply = String(index + 1);
    cell.textContent = move.san;
    const symbol = SYMBOLS[move.classification];
    if (symbol) {
      const sym = document.createElement("span");
      sym.className = "sym";
      sym.textContent = symbol;
      cell.append(sym);
    }
    cell.title = move.classification;
    row.append(cell);
  });
  list.replaceChildren(...items);
}

export function init() {
  ui.board = new Board($("board"));
  ui.board.setPosition(START_FEN);
  for (const radio of document.querySelectorAll('input[name="source"]')) {
    radio.addEventListener("change", () => renderSourceForm(radio.value));
  }
  renderSourceForm("pgn");
  $("first").addEventListener("click", () => goTo(0));
  $("prev").addEventListener("click", () => step(-1));
  $("next").addEventListener("click", () => step(1));
  $("last").addEventListener("click", () => goTo(ui.review.moves.length));
  $("flip").addEventListener("click", () => ui.board.flip());
  $("move-list").addEventListener("click", (event) => {
    const cell = event.target.closest(".san[data-ply]");
    if (cell) goTo(Number(cell.dataset.ply));
  });
  document.addEventListener("keydown", onKey);
  $("source-form").addEventListener("submit", (event) => {
    event.preventDefault();
    const form = event.target;
    const value = form.dataset.source === "pgn" ? form.elements.pgn.value : form.elements.username.value;
    if (value.trim()) loadGames(form.dataset.source, value);
  });
}

if (document.getElementById("board") && !window.__noAutoInit) init();
