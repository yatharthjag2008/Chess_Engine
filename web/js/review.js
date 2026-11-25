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
  return ui.game;
}

export function init() {
  ui.board = new Board($("board"));
  ui.board.setPosition(START_FEN);
  for (const radio of document.querySelectorAll('input[name="source"]')) {
    radio.addEventListener("change", () => renderSourceForm(radio.value));
  }
  renderSourceForm("pgn");
  $("source-form").addEventListener("submit", (event) => {
    event.preventDefault();
    const form = event.target;
    const value = form.dataset.source === "pgn" ? form.elements.pgn.value : form.elements.username.value;
    if (value.trim()) loadGames(form.dataset.source, value);
  });
}

if (document.getElementById("board") && !window.__noAutoInit) init();
