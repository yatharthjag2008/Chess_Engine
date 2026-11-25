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
  busy: false,        // a request (import or review) is running
  coaching: null,     // the pending coach request (a promise)
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

/** Errors as plain sentences; the codes the server sends get a hint on what to do. */
export function errorText(err) {
  if (err instanceof ApiError) {
    if (err.code === "stockfish_unavailable") {
      return "Reviews need Stockfish, which the server cannot find. Install it (see the README) and restart the server.";
    }
    if (err.code === "network") return "Cannot reach the server. Is scripts/serve.py running?";
    return err.message;
  }
  return err && err.message ? err.message : String(err);
}

export function showError(err) {
  setStatus(errorText(err), true);
}

/** While a request runs: the buttons that start another one are disabled, and the page is marked busy. */
export function setBusy(busy) {
  ui.busy = busy;
  document.querySelector("main").setAttribute("aria-busy", String(busy));
  for (const el of document.querySelectorAll("#source-form button, #review-button, #game-list button")) {
    el.disabled = busy;
  }
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
  if (ui.busy) return;
  setStatus(source === "pgn" ? "Reading the PGN..." : `Fetching games from ${source}...`, false, true);
  setBusy(true);
  try {
    const result = source === "pgn" ? await api.importPgn(value) : await api.importUser(source, value.trim());
    ui.games = result.games;
    ui.username = result.username;
    renderGameList();
    setStatus(ui.games.length ? `${ui.games.length} game${ui.games.length === 1 ? "" : "s"} found: pick one to review.`
      : "No standard chess games found.", !ui.games.length);
  } catch (err) {
    showError(err);
  } finally {
    setBusy(false);
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

/** Review a game played against the bot on the play page: your side is the side you played. */
export async function reviewBotGame(id) {
  setStatus("Loading your game...", false, true);
  let state, pgn;
  try {
    [state, pgn] = await Promise.all([api.state(id), api.pgn(id)]);
  } catch (err) {
    showError(err.status === 404 ? new Error("That game is no longer on the server (it expired, or the server restarted).") : err);
    return null;
  }
  if (!state.moves_san.length) {
    setStatus("That game has no moves to review yet.", true);
    return null;
  }
  ui.side = state.human_color;
  ui.games = [];
  ui.username = null;
  renderGameList();
  return startReview(pgn);
}

/** The bot game named in the URL (``#game=<id>``, as the play page links it), or null. */
export function gameFromUrl() {
  const match = location.hash.match(/^#game=([\w-]+)$/);
  return match ? match[1] : null;
}

export const SYMBOLS = { inaccuracy: "?!", mistake: "?", blunder: "??" };

/** Review the chosen game (or ``pgn``) and show it from the start. */
export async function startReview(pgn = ui.game && ui.game.pgn) {
  if (ui.busy || !pgn) return null;
  setStatus("Stockfish is reviewing the game (this takes a few seconds per move)...", false, true);
  setBusy(true);
  try {
    ui.review = await api.review(pgn);
  } catch (err) {
    ui.review = null;
    hideReview();
    showError(err);
    return null;
  } finally {
    setBusy(false);
  }
  ui.pgn = pgn;
  ui.coach = null;
  showReview();
  ui.coaching = loadCoach();      // in the background: the board is usable meanwhile
  return ui.review;
}

/** Hide everything that belongs to a review (before a new one, or when it failed). */
export function hideReview() {
  for (const id of ["stepper", "graph", "summary-panel", "detail-panel", "coach-panel", "moves-panel"]) $(id).hidden = true;
  clearBestSquares();
}

export function showReview() {
  const review = ui.review;
  ui.board.setOrientation(ui.side);
  const headers = review.headers || {};
  const name = (color) => `${headers[color === "white" ? "White" : "Black"] || (color === "white" ? "White" : "Black")}`;
  $("top-name").textContent = name(ui.side === "white" ? "black" : "white");
  $("bottom-name").textContent = name(ui.side);
  renderMoveList();
  renderGraph();
  renderSummary();
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

// ------------------------------------------------------------------ the move's detail

const MATE_CP = 10000;
export const CLASS_NAMES = { best: "Best", good: "Good", inaccuracy: "Inaccuracy", mistake: "Mistake", blunder: "Blunder" };

/** An evaluation in centipawns as text: "+1.25", "-0.40", "#3" / "#-2" for mates, "0.00" level. */
export function formatEval(cp) {
  if (Math.abs(cp) >= MATE_CP - 1000) {
    const n = MATE_CP - Math.abs(cp);
    return n === 0 ? (cp > 0 ? "1-0" : "0-1") : `#${cp > 0 ? "" : "-"}${n}`;
  }
  const pawns = cp / 100;
  return `${pawns > 0 ? "+" : pawns < 0 ? "-" : ""}${Math.abs(pawns).toFixed(2)}`;
}

/** A centipawn value from the mover's view, turned to White's view. */
const whiteView = (cp, color) => (color === "white" ? cp : -cp);

function line(tag, className, text) {
  const el = document.createElement(tag);
  if (className) el.className = className;
  el.textContent = text;
  return el;
}

/** The panel describing the move just played: verdict, evaluations, the better move and other options. */
export function renderDetail(ply, move) {
  const panel = $("detail-panel");
  clearBestSquares();
  if (!move) {
    panel.hidden = true;
    return;
  }
  panel.hidden = false;
  const verdict = line("p", `verdict ${move.classification}`,
    `${move.move_number} ${move.san}${SYMBOLS[move.classification] || ""}: ${CLASS_NAMES[move.classification]}`);
  const evals = line("p", "detail-evals",
    `Eval ${formatEval(move.eval_before)} \u2192 ${formatEval(move.eval_after)} · ` +
    `win chance ${Math.round(move.win_before)}% \u2192 ${Math.round(move.win_after)}%` +
    (move.win_lost >= 1 ? ` (\u2212${Math.round(move.win_lost)}%)` : ""));
  const children = [line("h2", null, "This move"), verdict, evals];
  const betterExists = move.best_uci && move.best_uci !== move.uci;
  if (betterExists) {
    const best = document.createElement("p");
    best.append(`Best was ${move.best_san} (${formatEval(whiteView(move.best_cp, move.color))}). `);
    const show = document.createElement("button");
    show.type = "button";
    show.id = "show-best";
    show.textContent = "Show on board";
    show.setAttribute("aria-pressed", "false");
    show.addEventListener("click", () => toggleBest(move, show));
    best.append(show);
    children.push(best);
  } else if (move.best_uci) {
    children.push(line("p", null, "This was Stockfish's choice."));
  }
  const others = (move.alternatives || []).filter((a) => a.uci !== move.uci);
  if (others.length) {
    const list = document.createElement("ul");
    list.className = "alternatives";
    for (const alt of others) list.append(line("li", null, `${alt.san} ${formatEval(whiteView(alt.cp, move.color))}`));
    children.push(line("p", "coach-stats", "Stockfish's top moves:"), list);
  }
  panel.replaceChildren(...children);
}

/** Show the position before the move with the better move's squares marked, or go back. */
function toggleBest(move, button) {
  const showing = button.getAttribute("aria-pressed") === "true";
  if (showing) {
    clearBestSquares();
    ui.board.setPosition(fenAt(ui.ply), { lastMove: move.uci, check: /[+#]$/.test(move.san) });
    button.textContent = "Show on board";
    button.setAttribute("aria-pressed", "false");
    return;
  }
  ui.board.setPosition(move.fen_before);
  markBestSquares(move.best_uci);
  button.textContent = "Back to the game";
  button.setAttribute("aria-pressed", "true");
}

export function markBestSquares(uci) {
  ui.board.squareEl(uci.slice(0, 2)).classList.add("best-from");
  ui.board.squareEl(uci.slice(2, 4)).classList.add("best-to");
}

export function clearBestSquares() {
  if (!ui.board) return;
  for (const el of ui.board.el.querySelectorAll(".best-from, .best-to")) el.classList.remove("best-from", "best-to");
}

onStep.push(renderDetail);

// ------------------------------------------------------------------ the eval graph

const SVG = "http://www.w3.org/2000/svg";
export const GRAPH = { width: 600, height: 120 };

function svg(tag, attrs = {}) {
  const el = document.createElementNS(SVG, tag);
  for (const [key, value] of Object.entries(attrs)) el.setAttribute(key, String(value));
  return el;
}

/** White's win chance per ply as an area (white below the line, black above), with the errors marked.
 *  Each ply has a full-height hit column with a tooltip; clicking it jumps to that position. */
export function renderGraph() {
  const box = $("graph");
  const points = ui.review.graph;
  box.hidden = !points.length;
  if (!points.length) return;
  const { width, height } = GRAPH;
  const last = points.length - 1;
  const x = (ply) => (last ? (ply / last) * width : 0);
  const y = (win) => ((100 - win) / 100) * height;
  const root = svg("svg", { viewBox: `0 0 ${width} ${height}`, role: "img", preserveAspectRatio: "none",
                            "aria-label": "White's win chance after each move" });
  root.append(svg("rect", { class: "area-black", x: 0, y: 0, width, height }));
  const outline = points.map((p) => `${x(p.ply).toFixed(1)},${y(p.white_win).toFixed(1)}`);
  root.append(svg("polygon", { class: "area-white", points: `0,${height} ${outline.join(" ")} ${width},${height}` }));
  root.append(svg("line", { class: "midline", x1: 0, x2: width, y1: height / 2, y2: height / 2 }));
  const current = svg("line", { class: "current", y1: 0, y2: height, x1: 0, x2: 0 });
  root.append(current);
  const step = last ? width / last : width;
  for (const p of points) {
    const move = p.ply ? ui.review.moves[p.ply - 1] : null;
    const hit = svg("rect", { class: "hit", x: Math.max(0, x(p.ply) - step / 2), y: 0, width: step, height,
                              fill: "transparent", "data-ply": p.ply });
    const title = svg("title");
    title.textContent = `${move ? `${move.move_number} ${move.san}${SYMBOLS[move.classification] || ""}` : "Start"}`
      + ` · ${formatEval(p.eval)} · White ${Math.round(p.white_win)}%`;
    hit.append(title);
    root.append(hit);
  }
  ui.review.moves.forEach((move, index) => {
    if (!SYMBOLS[move.classification]) return;
    const p = points[index + 1];
    root.append(svg("circle", { class: `marker ${move.classification}`, cx: x(p.ply), cy: y(p.white_win), r: 4,
                                "data-ply": p.ply, "pointer-events": "none" }));
  });
  root.append(svg("rect", { class: "frame", x: 0.5, y: 0.5, width: width - 1, height: height - 1 }));
  root.addEventListener("click", (event) => {
    const hit = event.target.closest("[data-ply]");
    if (hit) goTo(Number(hit.getAttribute("data-ply")));
  });
  box.replaceChildren(root);
  moveGraphCursor(ui.ply);
}

/** Put the current-position line on ``ply``. */
export function moveGraphCursor(ply) {
  const line = document.querySelector("#graph .current");
  if (!line || !ui.review) return;
  const last = ui.review.graph.length - 1;
  const at = last ? (ply / last) * GRAPH.width : 0;
  line.setAttribute("x1", at);
  line.setAttribute("x2", at);
}

onStep.push((ply) => moveGraphCursor(ply));

// ------------------------------------------------------------------ the summary

const CLASS_ORDER = ["best", "good", "inaccuracy", "mistake", "blunder"];

function cell(text, className = "") {
  return line("span", className, text);
}

/** Accuracy, average loss and moves per class for both players, then the key moments (click to jump). */
export function renderSummary() {
  const panel = $("summary-panel");
  const { summary, key_moments: moments, headers = {} } = ui.review;
  const grid = document.createElement("div");
  grid.className = "summary-grid";
  grid.setAttribute("role", "table");
  grid.append(cell(""), cell(headers.White || "White", "head num"), cell(headers.Black || "Black", "head num"));
  grid.append(cell("Accuracy"), cell(`${summary.white.accuracy.toFixed(1)}%`, "num"),
              cell(`${summary.black.accuracy.toFixed(1)}%`, "num"));
  grid.append(cell("Average loss"), cell(`${Math.round(summary.white.average_cp_loss)} cp`, "num"),
              cell(`${Math.round(summary.black.average_cp_loss)} cp`, "num"));
  for (const name of CLASS_ORDER) {
    const label = cell(`${CLASS_NAMES[name]}${SYMBOLS[name] ? ` ${SYMBOLS[name]}` : ""}`, `class-name ${name}`);
    grid.append(label, cell(String(summary.white.counts[name] || 0), "num"),
                cell(String(summary.black.counts[name] || 0), "num"));
  }
  const children = [line("h2", null, "Summary"), grid];
  if (moments.length) {
    children.push(line("h2", null, "Key moments"));
    const list = document.createElement("ul");
    list.className = "moments";
    for (const moment of moments) {
      const item = document.createElement("li");
      const button = document.createElement("button");
      button.type = "button";
      button.dataset.ply = String(moment.ply + 1);
      // The text already starts with the move ("4. Qxf7+?? dropped White's chances..."): only the dot is added.
      button.append(cell(moment.text, `class-name ${moment.classification}`));
      button.addEventListener("click", () => goTo(moment.ply + 1));
      item.append(button);
      list.append(item);
    }
    children.push(list);
  } else {
    children.push(line("p", "coach-stats", "No inaccuracies, mistakes or blunders: a clean game."));
  }
  panel.replaceChildren(...children);
  panel.hidden = false;
}

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

// ------------------------------------------------------------------ the coach

const SOURCES = { given: "as you set it", pgn: "from the game", estimated: "estimated from your moves" };
const pct = (x) => `${Math.round(x * 100)}%`;

/** Ask the coach about your side of the game (``rating``: optional, else from the PGN or estimated). */
export async function loadCoach(rating = null) {
  const panel = $("coach-panel");
  panel.hidden = false;
  panel.replaceChildren(line("h2", null, "Coach"),
                        line("p", "coach-stats", "Comparing your moves with players of your level..."));
  try {
    ui.coach = await api.coach(ui.pgn, { color: ui.side }, rating);
  } catch (err) {
    ui.coach = null;
    panel.replaceChildren(line("h2", null, "Coach"), line("p", "coach-stats", `The coach is unavailable: ${err.message}`));
    return null;
  }
  renderCoach();
  return ui.coach;
}

/** The coach's note on the move that led to ``ply`` (only your moves get notes), or null. */
export function noteAt(ply) {
  if (!ui.coach || !ply) return null;
  const move = ui.review.moves[ply - 1];
  return ui.coach.report.notes.find((n) => n.ply === move.ply) || null;
}

/** What the coach says at the current position. */
export function coachText() {
  const report = ui.coach.report;
  const note = noteAt(ui.ply);
  if (note) return note.text;
  if (!report.notes.length) return "No mistakes, and nothing stronger players would do clearly better. Well played.";
  const move = ui.ply ? ui.review.moves[ui.ply - 1] : null;
  return move && move.color === report.color ? "Nothing to add about this move." : "Step to one of your moves for its note.";
}

function button(id, text, onClick) {
  const el = document.createElement("button");
  el.type = "button";
  if (id) el.id = id;
  el.textContent = text;
  el.addEventListener("click", onClick);
  return el;
}

export function renderCoach() {
  const panel = $("coach-panel");
  if (!ui.coach) return;
  const { report, rating_source: source, next_level: next } = ui.coach;
  const side = report.color === "white" ? "White" : "Black";
  const noteEl = line("p", "coach-note", coachText());
  noteEl.id = "coach-note";
  const children = [
    line("h2", null, "Coach"),
    line("p", "coach-stats", `Coaching ${side}, rated about ${report.rating} (${SOURCES[source] || source}).`),
    line("p", "coach-stats", `You played your level's favourite move ${pct(report.your_level_match)} of the time, ` +
                             `and the ${next}-level move ${pct(report.next_level_match)}.`),
    noteEl,
  ];

  const actions = document.createElement("div");
  actions.className = "side-choice";
  if (report.notes.length) actions.append(button("next-note", `Next note (${report.notes.length})`, jumpToNextNote));
  const ratingBox = document.createElement("input");
  Object.assign(ratingBox, { type: "number", id: "coach-rating", min: "400", max: "3000", step: "50",
                             placeholder: String(report.rating) });
  ratingBox.setAttribute("aria-label", "Your rating");
  actions.append(ratingBox, button(null, "Coach at this rating", () => {
    const value = Number(ratingBox.value);
    if (value) loadCoach(value);
  }));

  const estimate = document.createElement("div");
  estimate.className = "side-choice";
  estimate.id = "estimate";
  estimate.append(button("estimate-button", "Estimate my rating from this game", estimateRating));
  panel.replaceChildren(...children, actions, estimate);
  panel.hidden = false;
}

// Stepping only changes the note, so the rating box keeps what was typed.
onStep.push(() => {
  const el = $("coach-note");
  if (el && ui.coach) el.textContent = coachText();
});

/** Go to the position after your next move with a note (wrapping round to the first). */
export function jumpToNextNote() {
  const plies = ui.coach.report.notes.map((n) => n.ply + 1);
  goTo(plies.find((p) => p > ui.ply) || plies[0]);
}

/** Your rating from this game's moves (the model's estimate, with its range). */
export async function estimateRating() {
  const box = $("estimate");
  box.replaceChildren(line("span", "loading", ""), "Estimating...");
  try {
    const result = await api.ratingEstimate(ui.pgn, { color: ui.side });
    const o = result.overall;
    box.replaceChildren(line("p", "coach-stats",
      `Your moves in this game look like a ${Math.round(o.estimate)} player ` +
      `(likely ${Math.round(o.low)}\u2013${Math.round(o.high)}, from ${o.moves} moves).`));
    if (result.note) box.append(line("p", "coach-stats", result.note));
    return result;
  } catch (err) {
    box.replaceChildren(line("p", "coach-stats", `No estimate: ${err.message}`));
    return null;
  }
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
  const id = gameFromUrl();
  if (id) reviewBotGame(id);
  $("source-form").addEventListener("submit", (event) => {
    event.preventDefault();
    const form = event.target;
    const value = form.dataset.source === "pgn" ? form.elements.pgn.value : form.elements.username.value;
    if (value.trim()) loadGames(form.dataset.source, value);
  });
}

if (document.getElementById("board") && !window.__noAutoInit) init();
