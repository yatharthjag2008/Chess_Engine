// The Review page: a pasted game (or the one just played) judged by Stockfish in the browser, shown as the local
// app's review page shows it (web/js/review.js): board and stepper, eval graph, summary, key moments, move detail.

import { Chess } from "../vendor/chess.js";
import { Board } from "./board.js";
import { MATE_CP, StockfishAnalyzer } from "./stockfish.js";
import { CLASSES, SYMBOLS, evalGraph, keyMoments, reviewMoves, summary } from "./review.js";
import { splitPgn } from "./games.js";
import { stored } from "./storage.js";

const $ = (id) => document.getElementById(id);

export const ui = {
  board: null,
  review: null,        // { headers, startFen, depth, moves, summary, moments, graph }
  ply: 0,
  side: "white",       // the side at the bottom
  running: false,
  makeAnalyzer: (depth) => new StockfishAnalyzer({ depth, multipv: 3 }),   // replaceable in tests
  analyzer: null,
};

const handoff = stored("human-chess-site.review");    // a game handed over by the play page

export const CLASS_NAMES = { best: "Best", good: "Good", inaccuracy: "Inaccuracy", mistake: "Mistake", blunder: "Blunder" };
const ERROR_SYMBOLS = { inaccuracy: "?!", mistake: "?", blunder: "??" };

function setStatus(text) {
  $("status").textContent = text;
}

/** An evaluation in centipawns as text: "+1.25", "-0.40", "#3" / "#-2" for mates, "0.00" level. */
export function formatEval(cp) {
  if (Math.abs(cp) >= MATE_CP - 1000) {
    const n = MATE_CP - Math.abs(cp);
    return n === 0 ? (cp > 0 ? "1-0" : "0-1") : `#${cp > 0 ? "" : "-"}${n}`;
  }
  const pawns = cp / 100;
  return `${pawns > 0 ? "+" : pawns < 0 ? "-" : ""}${Math.abs(pawns).toFixed(2)}`;
}

const whiteView = (cp, color) => (color === "white" ? cp : -cp);

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

// ------------------------------------------------------------------ reading the game

/** The first game of a PGN text -> { startFen, ucis, headers }, or { error }. */
export function readFirstGame(text) {
  const games = splitPgn(text);
  if (!games.length) return { error: "Paste a game in PGN." };
  const chess = new Chess();
  try {
    chess.loadPgn(games[0]);
  } catch (err) {
    return { error: `That game could not be read (${String(err.message || err).split("\n")[0]}).` };
  }
  const history = chess.history({ verbose: true });
  if (!history.length) return { error: "That game has no moves to review." };
  return { startFen: history[0].before, ucis: history.map((m) => m.lan), headers: chess.getHeaders() };
}

/** The side to show at the bottom: the player called "You" (a game from the play page), else White. */
function sideFor(headers) {
  return headers.Black === "You" && headers.White !== "You" ? "black" : "white";
}

// ------------------------------------------------------------------ reviewing

/** Review the first game of `text` at `depth`; renders it when done. Returns the review (null on an error). */
export async function startReview(text = $("review-pgn").value, depth = Number($("depth").value) || 12) {
  if (ui.running) return null;
  $("review-error").hidden = true;
  const game = readFirstGame(text);
  if (game.error) {
    $("review-error").textContent = game.error;
    $("review-error").hidden = false;
    return null;
  }
  ui.running = true;
  $("review").disabled = true;
  hideReview();
  setStatus("Starting Stockfish...");
  try {
    if (!ui.analyzer || ui.analyzer.depth !== depth) {
      if (ui.analyzer && ui.analyzer.terminate) ui.analyzer.terminate();
      ui.analyzer = ui.makeAnalyzer(depth);
    }
    const review = await reviewMoves(game.startFen, game.ucis, ui.analyzer, {
      headers: game.headers, depth,
      onProgress: (done, total) => setStatus(`Stockfish is analysing position ${done} of ${total} (depth ${depth})...`),
    });
    review.summary = summary(review);
    review.moments = keyMoments(review);
    review.graph = evalGraph(review);
    ui.review = review;
    ui.side = sideFor(game.headers);
    showReview();
    return review;
  } catch (err) {
    $("review-error").textContent = `The review failed: ${err.message}`;
    $("review-error").hidden = false;
    setStatus("");
    return null;
  } finally {
    ui.running = false;
    $("review").disabled = false;
  }
}

export function hideReview() {
  for (const id of ["stepper", "graph", "summary-panel", "detail-panel", "moves-panel"]) $(id).hidden = true;
  clearBestSquares();
}

export function showReview() {
  const review = ui.review;
  ui.board.setOrientation(ui.side);
  renderNames();
  renderMoveList();
  renderGraph();
  renderSummary();
  $("moves-panel").hidden = false;
  $("stepper").hidden = false;
  setStatus(`Reviewed ${review.moves.length} moves at depth ${review.depth}.`);
  goTo(0);
}

function renderNames() {
  const headers = ui.review ? ui.review.headers : {};
  const name = (color) => headers[color === "white" ? "White" : "Black"] || (color === "white" ? "White" : "Black");
  $("top-name").textContent = name(ui.side === "white" ? "black" : "white");
  $("bottom-name").textContent = name(ui.side);
}

// ------------------------------------------------------------------ stepping

/** The position after `ply` moves (0 = the start). */
export function fenAt(ply) {
  return ply === 0 ? ui.review.startFen : ui.review.moves[ply - 1].fenAfter;
}

/** Show the position after `ply` moves: board, highlighted move, label, graph cursor and the move's detail. */
export function goTo(ply) {
  if (!ui.review) return;
  const moves = ui.review.moves;
  ui.ply = Math.max(0, Math.min(moves.length, ply));
  const move = ui.ply ? moves[ui.ply - 1] : null;
  ui.board.setPosition(fenAt(ui.ply), { lastMove: move ? move.uci : null, check: Boolean(move && /[+#]$/.test(move.san)) });
  for (const cell of $("move-list").querySelectorAll(".san[data-ply]")) {
    const current = cell.dataset.ply === String(ui.ply);
    cell.classList.toggle("current", current);
    if (current) cell.setAttribute("aria-current", "step"); else cell.removeAttribute("aria-current");
    if (current && cell.scrollIntoView) cell.scrollIntoView({ block: "nearest" });
  }
  $("ply-label").textContent = move ? `${move.moveNumber} ${move.san}` : "Start";
  $("first").disabled = $("prev").disabled = ui.ply === 0;
  $("next").disabled = $("last").disabled = ui.ply === moves.length;
  moveGraphCursor(ui.ply);
  renderDetail(move);
}

export function step(delta) {
  goTo(ui.ply + delta);
}

// ------------------------------------------------------------------ the move's detail

/** The panel describing the move just played: verdict, evaluations, the better move and other options. */
export function renderDetail(move) {
  const panel = $("detail-panel");
  clearBestSquares();
  if (!move) {
    panel.hidden = true;
    return;
  }
  panel.hidden = false;
  const verdict = el("p", `verdict ${move.classification}`,
    `${move.moveNumber} ${move.san}${ERROR_SYMBOLS[move.classification] || ""}: ${CLASS_NAMES[move.classification]}`);
  const evals = el("p", "detail-evals",
    `Eval ${formatEval(move.evalBefore)} → ${formatEval(move.evalAfter)} · `
    + `win chance ${Math.round(move.winBefore)}% → ${Math.round(move.winAfter)}%`
    + (move.winLost >= 1 ? ` (−${Math.round(move.winLost)}%)` : ""));
  const children = [el("h2", null, "This move"), verdict, evals];
  if (move.bestUci && move.bestUci !== move.uci) {
    const best = el("p");
    best.append(`Best was ${move.bestSan} (${formatEval(whiteView(move.bestCp, move.color))}). `);
    const show = el("button", null, "Show on board");
    show.type = "button";
    show.id = "show-best";
    show.setAttribute("aria-pressed", "false");
    show.addEventListener("click", () => toggleBest(move, show));
    best.append(show);
    children.push(best);
  } else if (move.bestUci) {
    children.push(el("p", null, "This was Stockfish's choice."));
  }
  const others = move.alternatives.filter((a) => a.uci !== move.uci);
  if (others.length) {
    const list = el("ul", "alternatives");
    for (const alt of others) {
      const line = alt.pvSan && alt.pvSan.length > 1 ? `  (${alt.pvSan.slice(0, 5).join(" ")})` : "";
      list.append(el("li", null, `${alt.san} ${formatEval(whiteView(alt.cp, move.color))}${line}`));
    }
    children.push(el("p", "coach-stats", "Stockfish's top moves:"), list);
  }
  panel.replaceChildren(...children);
}

/** Show the position before the move with the better move's squares marked, or go back. */
function toggleBest(move, button) {
  if (button.getAttribute("aria-pressed") === "true") {
    clearBestSquares();
    ui.board.setPosition(fenAt(ui.ply), { lastMove: move.uci, check: /[+#]$/.test(move.san) });
    button.textContent = "Show on board";
    button.setAttribute("aria-pressed", "false");
    return;
  }
  ui.board.setPosition(move.fenBefore);
  ui.board.setMarks({ [move.bestUci.slice(0, 2)]: "best-from", [move.bestUci.slice(2, 4)]: "best-to" });
  ui.board.setShapes([{ orig: move.bestUci.slice(0, 2), dest: move.bestUci.slice(2, 4), brush: "green" }]);
  button.textContent = "Back to the game";
  button.setAttribute("aria-pressed", "true");
}

function clearBestSquares() {
  if (!ui.board) return;
  if (Object.keys(ui.board.marks || {}).length) ui.board.setMarks({});
  ui.board.setShapes([]);
}

// ------------------------------------------------------------------ the eval graph

const SVG = "http://www.w3.org/2000/svg";
export const GRAPH = { width: 600, height: 120 };

function svg(tag, attrs = {}) {
  const node = document.createElementNS(SVG, tag);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, String(value));
  return node;
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
  const outline = points.map((p) => `${x(p.ply).toFixed(1)},${y(p.whiteWin).toFixed(1)}`);
  root.append(svg("polygon", { class: "area-white", points: `0,${height} ${outline.join(" ")} ${width},${height}` }));
  root.append(svg("line", { class: "midline", x1: 0, x2: width, y1: height / 2, y2: height / 2 }));
  root.append(svg("line", { class: "current", y1: 0, y2: height, x1: 0, x2: 0 }));
  const stepWidth = last ? width / last : width;
  for (const p of points) {
    const move = p.ply ? ui.review.moves[p.ply - 1] : null;
    const hit = svg("rect", { class: "hit", x: Math.max(0, x(p.ply) - stepWidth / 2), y: 0, width: stepWidth, height,
                              fill: "transparent", "data-ply": p.ply });
    const title = svg("title");
    title.textContent = `${move ? `${move.moveNumber} ${move.san}${ERROR_SYMBOLS[move.classification] || ""}` : "Start"}`
      + ` · ${formatEval(p.eval)} · White ${Math.round(p.whiteWin)}%`;
    hit.append(title);
    root.append(hit);
  }
  ui.review.moves.forEach((move, index) => {
    if (!ERROR_SYMBOLS[move.classification]) return;
    const p = points[index + 1];
    root.append(svg("circle", { class: `marker ${move.classification}`, cx: x(p.ply), cy: y(p.whiteWin), r: 4,
                                "data-ply": p.ply, "pointer-events": "none" }));
  });
  root.append(svg("rect", { class: "frame", x: 0.5, y: 0.5, width: width - 1, height: height - 1 }));
  root.addEventListener("click", (event) => {
    const hit = event.target.closest("[data-ply]");
    if (hit) goTo(Number(hit.getAttribute("data-ply")));
  });
  box.replaceChildren(root);
}

function moveGraphCursor(ply) {
  const line = document.querySelector("#graph .current");
  if (!line || !ui.review) return;
  const last = ui.review.graph.length - 1;
  const at = last ? (ply / last) * GRAPH.width : 0;
  line.setAttribute("x1", at);
  line.setAttribute("x2", at);
}

// ------------------------------------------------------------------ the summary and the moves

/** Accuracy, average loss and moves per class for both players, then the key moments (click to jump). */
export function renderSummary() {
  const panel = $("summary-panel");
  const { summary: sides, moments, headers = {} } = ui.review;
  const grid = el("div", "summary-grid");
  grid.setAttribute("role", "table");
  const pct = (side) => (side.accuracy === null ? "–" : `${side.accuracy.toFixed(1)}%`);
  const loss = (side) => (side.averageCpLoss === null ? "–" : `${Math.round(side.averageCpLoss)} cp`);
  grid.append(el("span", ""), el("span", "head num", headers.White || "White"), el("span", "head num", headers.Black || "Black"));
  grid.append(el("span", "", "Accuracy"), el("span", "num", pct(sides.white)), el("span", "num", pct(sides.black)));
  grid.append(el("span", "", "Average loss"), el("span", "num", loss(sides.white)), el("span", "num", loss(sides.black)));
  for (const name of CLASSES) {
    grid.append(el("span", `class-name ${name}`, `${CLASS_NAMES[name]}${ERROR_SYMBOLS[name] ? ` ${ERROR_SYMBOLS[name]}` : ""}`),
                el("span", "num", String(sides.white.counts[name] || 0)),
                el("span", "num", String(sides.black.counts[name] || 0)));
  }
  const children = [el("h2", null, "Summary"), grid];
  if (moments.length) {
    children.push(el("h2", null, "Key moments"));
    const list = el("ul", "moments");
    for (const moment of moments) {
      const item = el("li");
      const button = el("button");
      button.type = "button";
      button.dataset.ply = String(moment.ply + 1);
      button.append(el("span", `class-name ${moment.classification}`, moment.text));
      button.addEventListener("click", () => goTo(moment.ply + 1));
      item.append(button);
      list.append(item);
    }
    children.push(list);
  } else {
    children.push(el("p", "coach-stats", "No inaccuracies, mistakes or blunders: a clean game."));
  }
  panel.replaceChildren(...children);
  panel.hidden = false;
}

/** The move list, one cell per move, each with its class (and symbol for the errors). */
export function renderMoveList() {
  const items = [];
  let row = null;
  ui.review.moves.forEach((move, index) => {
    if (!row || move.color === "white") {
      row = el("li");
      row.append(el("span", "num", `${move.moveNumber.replace(/\.+$/, "")}.`));
      if (move.color === "black") row.append(el("span"));       // Black moved first
      items.push(row);
    }
    const cell = el("span", `san ${move.classification}`, move.san);
    cell.dataset.ply = String(index + 1);
    if (ERROR_SYMBOLS[move.classification]) cell.append(el("span", "sym", ERROR_SYMBOLS[move.classification]));
    cell.title = move.classification;
    row.append(cell);
  });
  $("move-list").replaceChildren(...items);
}

// ------------------------------------------------------------------ the page

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

/** A game from the play page ("Review this game"): put it in the box and review it. */
export async function openHandedGame() {
  const text = handoff.load(null);
  if (typeof text !== "string" || !text.trim()) return null;
  handoff.clear();
  $("review-pgn").value = text;
  return startReview(text);
}

export async function init() {
  ui.board = new Board($("board"));
  ui.board.setLegalMoves([], null);
  $("review-form").addEventListener("submit", (event) => {
    event.preventDefault();
    startReview();
  });
  $("first").addEventListener("click", () => goTo(0));
  $("prev").addEventListener("click", () => step(-1));
  $("next").addEventListener("click", () => step(1));
  $("last").addEventListener("click", () => goTo(ui.review ? ui.review.moves.length : 0));
  $("flip").addEventListener("click", () => {
    ui.side = ui.side === "white" ? "black" : "white";
    ui.board.setOrientation(ui.side);
    renderNames();
  });
  $("move-list").addEventListener("click", (event) => {
    const cell = event.target.closest(".san[data-ply]");
    if (cell) goTo(Number(cell.dataset.ply));
  });
  document.addEventListener("keydown", onKey);
  if (window.location.hash === "#game") await openHandedGame();
}

if (document.getElementById("review-form") && !window.__noAutoInit) init();
