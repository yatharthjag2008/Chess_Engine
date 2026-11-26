// The public play page: the bot runs in the browser (engine.js), the rules come from chess.js.

import { Chess } from "../vendor/chess.js";
import { Board, pieceEl } from "./board.js";
import { loadModel, sampleMove } from "./engine.js";
import { Sounds, soundFor } from "./sounds.js";
import { START, captured, describeCaptures } from "./material.js";

const $ = (id) => document.getElementById(id);

export const ui = {
  model: null,
  board: null,
  chess: null,
  startFen: START,     // where the game began (captures are counted from here)
  view: null,          // null: the game as it stands; a number: looking at the position after that many moves
  thoughts: {},        // what the bot weighed, by the ply its move made: { san, uci, prob, candidates }
  human: "w",          // "w" or "b"
  rating: 1500,
  busy: false,
  last: null,          // the bot's last move: { san, uci, candidates }
  over: false,
  random: Math.random, // replaceable in tests
  delay: 350,          // ms before the bot moves (so its move is visible as a move)
  sounds: new Sounds(),
};

const SOUND_KEY = "human-chess-site.sound";
const VOLUME_KEY = "human-chess-site.volume";

/** Volume from 0 to 100 (remembered like the on/off switch). */
export function setVolume(percent, { save = true } = {}) {
  const value = Math.min(100, Math.max(0, Number(percent) || 0));
  ui.sounds.volume = value / 100;
  $("volume").value = String(value);
  if (save) {
    try { localStorage.setItem(VOLUME_KEY, String(value)); } catch { /* not remembered */ }
  }
}

function savedVolume() {
  try { return localStorage.getItem(VOLUME_KEY) ?? 70; } catch { return 70; }
}

/** Sound on or off (remembered in this browser when it allows). */
export function setSound(on, { save = true } = {}) {
  ui.sounds.enabled = on;
  const button = $("sound");
  button.textContent = `Sound: ${on ? "on" : "off"}`;
  button.setAttribute("aria-pressed", String(on));
  if (!save) return;
  try { localStorage.setItem(SOUND_KEY, on ? "on" : "off"); } catch { /* private mode: not remembered */ }
}

function savedSound() {
  try { return localStorage.getItem(SOUND_KEY) !== "off"; } catch { return true; }
}

/** Make a move on the game (UCI or SAN) and play its sound; returns chess.js's move. */
function makeMove(move) {
  const made = ui.chess.move(move);
  ui.sounds.play(soundFor(made, { check: ui.chess.inCheck() }));
  return made;
}

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

/** The position after `ply` half-moves of `history` (chess.js verbose moves), and the move that led there. */
function positionAt(history, ply) {
  if (ply === 0) return { fen: history.length ? history[0].before : ui.chess.fen(), move: null };
  const move = history[ply - 1];
  return { fen: move.after, move };
}

/** "Looking at the position after 12...Nf6." for a chess.js move (null: the start). */
function lookingAt(move) {
  if (!move) return "Looking at the starting position.";
  const number = Number(move.before.split(" ")[5]);          // the full-move number before the move
  return `Looking at the position after ${number}${move.color === "w" ? "." : "..."}${move.san}.`;
}

/** Look at an earlier position (`ply` half-moves in); null, or the last ply, returns to the game. */
export function showPly(ply) {
  if (!ui.chess) return;
  const plies = ui.chess.history().length;
  ui.view = ply === null || ply >= plies ? null : Math.max(0, ply);
  render();
}

/** Step `delta` moves back (negative) or forward from the position shown. */
export function step(delta) {
  if (!ui.chess) return;
  const plies = ui.chess.history().length;
  showPly((ui.view ?? plies) + delta);
}

const NAVIGATION = {
  ArrowLeft: () => step(-1), ArrowRight: () => step(1),
  Home: () => showPly(0), ArrowUp: () => showPly(0), End: () => showPly(null), ArrowDown: () => showPly(null),
};

export function render() {
  const c = ui.chess;
  const history = c.history({ verbose: true });
  const viewing = ui.view !== null && ui.view < history.length;
  if (!viewing) ui.view = null;
  const shown = positionAt(history, viewing ? ui.view : history.length);
  const humanToMove = !viewing && !ui.over && !ui.busy && c.turn() === ui.human;
  const check = viewing ? Boolean(shown.move && /[+#]$/.test(shown.move.san)) : c.inCheck();
  ui.board.setPosition(shown.fen, { lastMove: shown.move ? shown.move.lan : null, check });
  const botToMove = !viewing && !ui.over && c.turn() !== ui.human;   // meanwhile you may premove
  ui.board.setLegalMoves(humanToMove ? legalMoves() : [], humanToMove ? colorName(ui.human) : null,
                         { premover: botToMove ? colorName(ui.human) : null });
  const bottomIsHuman = ui.board.orientation === colorName(ui.human);
  $("top-name").textContent = bottomIsHuman ? `Bot ${ui.rating}` : "You";
  $("bottom-name").textContent = bottomIsHuman ? "You" : `Bot ${ui.rating}`;
  $("game-panel").hidden = false;
  renderMoves(history, viewing ? ui.view : history.length);
  const shownPly = viewing ? ui.view : history.length;
  $("first").disabled = $("prev").disabled = shownPly === 0;
  $("next").disabled = $("last").disabled = !viewing;
  renderCaptured(shown.fen);
  renderThinking(viewing ? ui.thoughts[ui.view] || null : ui.last);
  const humanMoves = history.filter((m) => m.color === ui.human).length;
  $("takeback").disabled = ui.over || ui.busy || humanMoves === 0;
  $("resign").disabled = ui.over || ui.busy;
  const end = outcome();
  ui.board.el.classList.toggle("viewing", viewing);
  if (viewing) setStatus(`${lookingAt(shown.move)} End, or a click on the board, returns to the game.`);
  else if (ui.over) setStatus(ui.overText);
  else if (ui.busy) setStatus("The bot is thinking...");
  else if (humanToMove) setStatus(`Your move${c.inCheck() ? ": you are in check" : ""}.`);
  if (end && !ui.over) finish(end.text, end.winner);
}

function setStatus(text) {
  $("status").textContent = text;
}

/** The move list; the move at `shownPly` is highlighted, and every move can be clicked to look at it. */
function renderMoves(history, shownPly) {
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
      const ply = blackFirst ? index : index + 1;           // the position after this move
      cell.className = "san" + (ply === shownPly && sans[index] ? " latest" : "");
      if (sans[index]) {
        cell.textContent = sans[index];
        cell.dataset.ply = String(ply);
      }
      li.append(cell);
    }
    items.push(li);
  }
  $("move-list").replaceChildren(...items);
}

/** Next to each player, the pieces they have captured. */
function renderCaptured(fen) {
  const taken = captured(ui.startFen, fen);
  const top = ui.board.orientation === "white" ? "b" : "w";
  for (const [id, side] of [["top-captured", top], ["bottom-captured", top === "w" ? "b" : "w"]]) {
    const opponent = side === "w" ? "black" : "white";
    const groups = [];                                   // one group per kind of piece, overlapped in CSS
    for (const type of taken[side]) {
      if (!groups.length || groups[groups.length - 1].dataset.type !== type) {
        const group = document.createElement("span");
        group.className = "group";
        group.dataset.type = type;
        groups.push(group);
      }
      groups[groups.length - 1].append(pieceEl(opponent, type));
    }
    const lead = side === "w" ? taken.diff : -taken.diff;   // material ahead, shown on the side ahead
    if (lead > 0) {
      const score = document.createElement("span");
      score.className = "score";
      score.textContent = `+${lead}`;
      groups.push(score);
    }
    $(id).replaceChildren(...groups);
    $(id).setAttribute("aria-label", describeCaptures(taken[side], lead));
  }
}

/** "What players at this rating play" for one bot move (null hides the panel). */
function renderThinking(move) {
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
  ui.sounds.play(winner === null ? "draw" : winner === ui.human ? "win" : "lose");
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
  makeMove(choice.uci);
  ui.last = { san: choice.san, uci: choice.uci, prob: choice.prob, candidates: moves };
  ui.thoughts[c.history().length] = ui.last;
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
  makeMove(uci);
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
  ui.startFen = ui.chess.fen();
  ui.view = null;
  ui.thoughts = {};
  ui.rating = rating;
  ui.human = color === "black" ? "b" : "w";
  ui.over = false;
  ui.last = null;
  $("result-banner").hidden = true;
  ui.board.setOrientation(color === "black" ? "black" : "white");
  ui.sounds.play("start");
  render();
  if (ui.human === "b") await botTurn();
}

export function takeback() {
  if (ui.over || ui.busy) return;
  ui.chess.undo();                                   // the bot's reply
  if (ui.chess.turn() !== ui.human) ui.chess.undo(); // and your move
  ui.last = null;
  ui.view = null;
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
  ui.board = new Board($("board"), {
    promotionEl: $("promotion"),
    onMove: (uci) => play(uci),
    onPremove: (event) => ui.sounds.play(event === "set" ? "premove" : "premove-dropped"),
  });
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
  $("board").addEventListener("click", (event) => {         // a click on the board returns to the game
    if (ui.view === null) return;
    event.stopPropagation();
    showPly(null);
  }, true);
  $("first").addEventListener("click", () => showPly(0));
  $("prev").addEventListener("click", () => step(-1));
  $("next").addEventListener("click", () => step(1));
  $("last").addEventListener("click", () => showPly(null));
  $("move-list").addEventListener("click", (event) => {
    const cell = event.target.closest("[data-ply]");
    if (cell) showPly(Number(cell.dataset.ply));
  });
  $("takeback").addEventListener("click", takeback);
  $("resign").addEventListener("click", resign);
  $("download").addEventListener("click", download);
  $("sound").addEventListener("click", () => setSound(!ui.sounds.enabled));
  const unlock = () => {                                   // audio may only start after a click or key
    ui.sounds.unlock();
    for (const type of ["pointerdown", "keydown"]) document.removeEventListener(type, unlock, true);
  };
  for (const type of ["pointerdown", "keydown"]) document.addEventListener(type, unlock, true);
  setSound(savedSound(), { save: false });
  $("volume").addEventListener("input", () => setVolume($("volume").value));
  $("volume").addEventListener("change", () => ui.sounds.play("move"));   // a sample at the new volume
  setVolume(savedVolume(), { save: false });
  document.addEventListener("keydown", (event) => {
    const typing = event.target instanceof Element && event.target.closest("input, select, textarea, button");
    if (typing || event.ctrlKey || event.metaKey || event.altKey) return;
    if (event.key === "f") $("flip").click();
    if (event.key === "m") $("sound").click();
    if (NAVIGATION[event.key] && ui.chess) {
      event.preventDefault();                                // not scrolling the page
      NAVIGATION[event.key]();
    }
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
