// The public play page: the bot runs in the browser (engine.js), the rules come from chess.js.

import { Chess } from "../vendor/chess.js";
import { Board } from "./board.js";
import { loadModel, sampleMove } from "./engine.js";

const $ = (id) => document.getElementById(id);

export const ui = {
  model: null,
  board: null,
  chess: null,
  human: "w",          // "w" or "b"
  rating: 1500,
  busy: false,
  last: null,          // the bot's last move: { san, uci, candidates }
  over: false,
  random: Math.random, // replaceable in tests
  delay: 350,          // ms before the bot moves (so its move is visible as a move)
};

const colorName = (c) => (c === "w" ? "white" : "black");

/** chess.js moves in the board's format. */
function legalMoves() {
  return ui.chess.moves({ verbose: true }).map((m) => ({ uci: m.lan, from_square: m.from, to_square: m.to,
                                                         promotion: m.promotion || null }));
}

export function outcome() {
  const c = ui.chess;
  if (c.isCheckmate()) return { text: `${c.turn() === "w" ? "Black" : "White"} wins by checkmate`,
                                winner: c.turn() === "w" ? "b" : "w" };
  if (c.isStalemate()) return { text: "Draw by stalemate", winner: null };
  if (c.isInsufficientMaterial()) return { text: "Draw by insufficient material", winner: null };
  if (c.isThreefoldRepetition()) return { text: "Draw by threefold repetition", winner: null };
  if (c.isDrawByFiftyMoves()) return { text: "Draw by the fifty-move rule", winner: null };
  return null;
}

export function render() {
  const c = ui.chess;
  const history = c.history({ verbose: true });
  const last = history.length ? history[history.length - 1].lan : null;
  const humanToMove = !ui.over && !ui.busy && c.turn() === ui.human;
  ui.board.setPosition(c.fen(), { lastMove: last, check: c.inCheck() });
  const botToMove = !ui.over && c.turn() !== ui.human;           // meanwhile you may premove
  ui.board.setLegalMoves(humanToMove ? legalMoves() : [], humanToMove ? colorName(ui.human) : null,
                         { premover: botToMove ? colorName(ui.human) : null });
  const bottomIsHuman = ui.board.orientation === colorName(ui.human);
  $("top-name").textContent = bottomIsHuman ? `Bot ${ui.rating}` : "You";
  $("bottom-name").textContent = bottomIsHuman ? "You" : `Bot ${ui.rating}`;
  $("game-panel").hidden = false;
  renderMoves(history);
  renderThinking();
  const humanMoves = history.filter((m) => m.color === ui.human).length;
  $("takeback").disabled = ui.over || ui.busy || humanMoves === 0;
  $("resign").disabled = ui.over || ui.busy;
  const end = outcome();
  if (ui.over) setStatus(ui.overText);
  else if (ui.busy) setStatus("The bot is thinking...");
  else if (humanToMove) setStatus(`Your move${c.inCheck() ? ": you are in check" : ""}.`);
  if (end && !ui.over) finish(end.text, end.winner);
}

function setStatus(text) {
  $("status").textContent = text;
}

function renderMoves(history) {
  const items = [];
  const blackFirst = history.length && history[0].color === "b";
  const sans = blackFirst ? [null, ...history.map((m) => m.san)] : history.map((m) => m.san);
  for (let i = 0; i < sans.length; i += 2) {
    const li = document.createElement("li");
    const num = document.createElement("span");
    num.className = "num";
    num.textContent = `${i / 2 + 1}.`;
    li.append(num);
    for (const index of [i, i + 1]) {
      const cell = document.createElement("span");
      cell.className = "san" + (index === sans.length - 1 ? " latest" : "");
      if (sans[index]) cell.textContent = sans[index];
      li.append(cell);
    }
    items.push(li);
  }
  $("move-list").replaceChildren(...items);
}

function renderThinking() {
  const move = ui.last;
  $("thinking-panel").hidden = !move;
  if (!move) return;
  $("thinking-caption").textContent = `In the position before ${move.san}, players rated about ${ui.rating} chose:`;
  const rows = move.candidates.slice(0, 5).map((c) => {
    const li = document.createElement("li");
    li.className = "candidate" + (c.uci === move.uci ? " played" : "");
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
  if (!move.candidates.slice(0, 5).some((c) => c.uci === move.uci)) {
    const note = document.createElement("li");
    note.className = "hint";
    note.textContent = `It played ${move.san}, a rarer choice (${(move.prob * 100).toFixed(1)}%).`;
    rows.push(note);
  }
  $("candidates").replaceChildren(...rows);
}

function finish(text, winner) {
  ui.over = true;
  ui.overText = text;
  setStatus(text);
  const banner = $("result-banner");
  const heading = document.createElement("h2");
  heading.textContent = winner === null ? "Draw" : winner === ui.human ? "You won" : "The bot won";
  const p = document.createElement("p");
  p.textContent = text;
  const actions = document.createElement("div");
  actions.className = "actions";
  const again = document.createElement("button");
  again.type = "button";
  again.className = "primary";
  again.textContent = "New game";
  again.addEventListener("click", () => newGame(readForm()));
  const close = document.createElement("button");
  close.type = "button";
  close.textContent = "Close";
  close.addEventListener("click", () => { banner.hidden = true; });
  actions.append(again, close);
  banner.replaceChildren(heading, p, actions);
  banner.hidden = false;
  ui.board.setLegalMoves([], null);
  $("takeback").disabled = $("resign").disabled = true;
}

/** The bot picks a move as often as players of its rating play it. */
export function botMove() {
  const c = ui.chess;
  const { moves } = ui.model.predict(c, ui.rating);
  const choice = sampleMove(moves, 1, ui.random);
  c.move(choice.uci);
  ui.last = { san: choice.san, uci: choice.uci, prob: choice.prob, candidates: moves };
  return choice;
}

function botTurn() {
  ui.busy = true;
  render();
  return new Promise((resolve) => {
    setTimeout(() => {
      botMove();
      ui.busy = false;
      render();
      resolve();
    }, ui.delay);
  });
}

/** You play ``uci``; the bot answers. */
export async function play(uci) {
  if (ui.over || ui.busy || ui.chess.turn() !== ui.human) return;
  ui.chess.move(uci);
  render();
  if (!ui.over) await botTurn();
}

export function readForm() {
  const data = new FormData($("new-game"));
  let color = data.get("color");
  if (color === "random") color = ui.random() < 0.5 ? "white" : "black";
  return { rating: Number(data.get("rating")), color };
}

export async function newGame({ rating, color }) {
  ui.chess = new Chess();
  ui.rating = rating;
  ui.human = color === "black" ? "b" : "w";
  ui.over = false;
  ui.last = null;
  $("result-banner").hidden = true;
  ui.board.setOrientation(color === "black" ? "black" : "white");
  render();
  if (ui.human === "b") await botTurn();
}

export function takeback() {
  if (ui.over || ui.busy) return;
  ui.chess.undo();                                   // the bot's reply
  if (ui.chess.turn() !== ui.human) ui.chess.undo(); // and your move
  ui.last = null;
  render();
}

export function resign() {
  if (ui.over || !ui.chess) return;
  finish(`You resigned: ${ui.human === "w" ? "Black" : "White"} wins`, ui.human === "w" ? "b" : "w");
}

function download() {
  const c = ui.chess;
  c.setHeader("Event", "Game vs. human-like bot (in the browser)");
  c.setHeader(ui.human === "w" ? "White" : "Black", "You");
  c.setHeader(ui.human === "w" ? "Black" : "White", `Bot ${ui.rating}`);
  const blob = new Blob([c.pgn() + "\n"], { type: "application/x-chess-pgn" });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = `game-vs-bot-${ui.rating}.pgn`;
  link.click();
  URL.revokeObjectURL(link.href);
}

export async function init(modelBase = "model/") {
  ui.board = new Board($("board"), { promotionEl: $("promotion"), onMove: (uci) => play(uci) });
  ui.board.setLegalMoves([], null);
  const rating = $("rating");
  const show = () => { $("rating-value").textContent = rating.value; };
  rating.addEventListener("input", show);
  show();
  $("new-game").addEventListener("submit", (event) => {
    event.preventDefault();
    newGame(readForm());
  });
  $("flip").addEventListener("click", () => { ui.board.flip(); if (ui.chess) render(); });
  $("takeback").addEventListener("click", takeback);
  $("resign").addEventListener("click", resign);
  $("download").addEventListener("click", download);
  document.addEventListener("keydown", (event) => {
    const typing = event.target instanceof Element && event.target.closest("input, select, textarea, button");
    if (typing || event.ctrlKey || event.metaKey || event.altKey) return;
    if (event.key === "f") $("flip").click();
    if (event.key === "Escape") ui.board.cancelPremove();   // drops the selection and any premove
  });
  try {
    ui.model = await loadModel(modelBase);
  } catch (err) {
    setStatus(`The model could not be loaded (${err.message}).`);
    return;
  }
  const check = ui.model.selfTest(Chess);
  $("model-note").textContent = check.ok
    ? `Model ${ui.model.meta.name}: ${(ui.model.meta.floats / 1e6).toFixed(2)}M weights, checked against PyTorch in this browser.`
    : `Warning: this browser's results differ from PyTorch's (by ${check.worst.toExponential(1)}).`;
  $("play").disabled = false;
  setStatus("Choose a rating and press Play.");
}

if (document.getElementById("board") && !window.__noAutoInit) init();
