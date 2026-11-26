// The public play page: the bot runs in the browser (engine.js), the rules come from chess.js.

import { Chess } from "../vendor/chess.js";
import { Board, pieceEl } from "./board.js";
import { loadModel, sampleMove } from "./engine.js";
import { Sounds, soundFor } from "./sounds.js";
import { START, canMate, captured, describeCaptures } from "./material.js";
import { stored } from "./storage.js";
import { Clock, baseSeconds, formatClock, thinkTime } from "./clock.js";
import { loadOpenings, openingOf } from "./openings.js";
import { PRESETS, parsePosition } from "./position.js";
import { ADAPTIVE, LiveAdjuster, botRatingFor, readProfile, recordGame, scoreFor } from "./adaptive.js";
import { HINTS, HintBudget, HintEngine } from "./hints.js";
import { StockfishAnalyzer } from "./stockfish.js";

const $ = (id) => document.getElementById(id);

export const ui = {
  model: null,
  board: null,
  chess: null,
  startFen: START,     // where the game began (captures are counted from here)
  view: null,          // null: the game as it stands; a number: looking at the position after that many moves
  openings: null,      // the opening-name table, once loaded
  thoughts: {},        // what the bot weighed, by the ply its move made: { san, uci, prob, candidates }
  human: "w",          // "w" or "b"
  rating: 1500,
  busy: false,
  last: null,          // the bot's last move: { san, uci, candidates }
  over: false,
  random: Math.random, // replaceable in tests
  delay: 350,          // ms before the bot moves in untimed games (so its move is visible as a move)
  thinkScale: 1,       // timed games: the bot's human-like thinking time is multiplied by this (tests shorten it)
  sounds: new Sounds(),
  profile: null,       // your rating for adaptive difficulty: { rating, games, history }
  adaptive: { mode: "off", start: 1500, rated: true, recorded: null },   // this game's adaptive difficulty
  hints: new HintBudget(HINTS.limit),   // this game's hints
  hint: null,          // the latest hint: { fen, data }, shown while that position is on the board
  hintEngine: null,    // Stockfish and the model, started on the first hint
  makeHintEngine: () => new HintEngine(ui.model, new StockfishAnalyzer({ depth: HINTS.depth, multipv: HINTS.multipv })),
};

const saved = {
  sound: stored("human-chess-site.sound"),      // "on" / "off"
  volume: stored("human-chess-site.volume"),    // 0-100
  game: stored("human-chess-site.game"),        // the game in progress (or just finished), see snapshot()
  settings: stored("human-chess-site.settings"), // the new-game form: { rating, color, time, difficulty }
  profile: stored("human-chess-site.adaptive"),  // your adaptive-difficulty rating and recent games
  handoff: stored("human-chess-site.analyse"),   // a game handed to the Analyse page
  review: stored("human-chess-site.review"),     // a game handed to the Review page
};

/** Your rating for adaptive difficulty (read once from this browser). */
export function profile() {
  if (!ui.profile) ui.profile = readProfile(saved.profile.load());
  return ui.profile;
}

/** Under the Difficulty choice: what "Follow my level" will do (the bot's rating then comes from your profile). */
export function renderAdaptiveNote() {
  const mode = $("new-game").elements.difficulty.value;
  const following = mode === "between";
  const note = $("adaptive-note");
  note.hidden = mode === "off";
  $("rating").disabled = following;
  const p = profile();
  renderAdaptiveHistory(p, mode);
  if (!following) {
    $("rating-value").textContent = $("rating").value;
    note.textContent = `The bot starts at the rating you choose, then plays a step stronger while it is losing or `
      + `weaker while it is winning (at most ${ADAPTIVE.liveCap} either way). Your rating: ${Math.round(p.rating)}.`;
    return;
  }
  const next = botRatingFor(p.rating);
  $("rating-value").textContent = String(next);
  note.textContent = p.games
    ? `Your rating: ${Math.round(p.rating)} after ${p.games} game${p.games === 1 ? "" : "s"}. The bot plays at ${next}.`
    : `Your first game starts at ${next}; after each game the bot moves towards your level.`;
}

const RESULT_WORDS = { 1: "won", 0.5: "drew", 0: "lost" };

/** Your latest rated games, newest first, with how each moved your rating. */
function renderAdaptiveHistory(p, mode) {
  $("adaptive-history").hidden = mode === "off" || !p.games;
  const rows = p.history.slice(-10).reverse().map((g) => {
    const li = document.createElement("li");
    const change = Math.round(g.ratingAfter) - Math.round(g.ratingBefore);
    li.textContent = `${g.date}: ${RESULT_WORDS[g.score] || "played"} vs. bot ${g.botRating}, `
      + `${Math.round(g.ratingBefore)} \u2192 ${Math.round(g.ratingAfter)} (${change >= 0 ? "+" : ""}${change})`
      + (g.mode === "live" ? ", live" : "");
    return li;
  });
  $("adaptive-games").replaceChildren(...rows);
}

/** Start your adaptive-difficulty rating again (at `rating`, else the usual start). */
export function resetProfile(rating) {
  ui.profile = readProfile(rating === undefined ? null : { rating, games: 0, history: [] });
  saved.profile.save(ui.profile);
  renderAdaptiveNote();
  return ui.profile;
}

/** A finished game counts for adaptive difficulty (once; not for games from a position). */
function recordResult(winner) {
  const a = ui.adaptive;
  if (a.mode === "off" || !a.rated || a.recorded || !ui.hints.rated) return;
  const before = profile().rating;
  const botRating = a.live ? Math.round(a.live.meanRating) : a.start;   // live: the rating it played at on average
  ui.profile = recordGame(profile(), botRating, scoreFor(winner, ui.human), a.mode);
  saved.profile.save(ui.profile);
  a.recorded = { before, after: ui.profile.rating, next: botRatingFor(ui.profile.rating) };
  renderAdaptiveNote();
}

/** Fill the new-game form from saved settings (unknown values are left alone). */
export function applySettings(settings) {
  if (!settings) return;
  const rating = Number(settings.rating);
  if (rating >= 800 && rating <= 2400) {
    $("rating").value = String(rating);
    $("rating-value").textContent = String(rating);
  }
  if (settings.hints !== undefined && [...$("hint-limit").options].some((o) => o.value === String(settings.hints))) {
    $("hint-limit").value = String(settings.hints);
  }
  for (const name of ["color", "time", "difficulty"]) {
    const radio = $("new-game").querySelector(`input[name="${name}"][value="${settings[name]}"]`);
    if (radio) radio.checked = true;
  }
}

/** Everything needed to carry on with the game after a reload. */
export function snapshot() {
  const c = ui.chess;
  const history = c.history({ verbose: true });
  const trim = (t) => t && { ...t, candidates: t.candidates.slice(0, 8) };
  return {
    version: 1,
    start: history.length ? history[0].before : c.fen(),   // the position the moves are played from
    startFen: ui.startFen,
    moves: history.map((m) => m.lan),
    human: ui.human,
    rating: ui.rating,
    orientation: ui.board.orientation,
    over: ui.over ? { text: ui.overText, winner: ui.winner ?? null } : null,
    thoughts: Object.fromEntries(Object.entries(ui.thoughts).map(([ply, t]) => [ply, trim(t)])),
    time: ui.time || "none",
    adaptive: ui.adaptive,
    hints: ui.hints.toJSON(),
    clock: ui.clock ? ui.clock.toJSON() : null,           // carries on after a reload, time running as on a server
    savedAt: Date.now(),
  };
}

/** Volume from 0 to 100 (remembered like the on/off switch). */
export function setVolume(percent, { save = true } = {}) {
  const value = Math.min(100, Math.max(0, Number(percent) || 0));
  ui.sounds.volume = value / 100;
  $("volume").value = String(value);
  if (save) saved.volume.save(value);
}

/** Sound on or off (remembered in this browser when it allows). */
export function setSound(on, { save = true } = {}) {
  ui.sounds.enabled = on;
  const button = $("sound");
  button.textContent = `Sound: ${on ? "on" : "off"}`;
  button.setAttribute("aria-pressed", String(on));
  if (save) saved.sound.save(on ? "on" : "off");
}

/** Make a move on the game (UCI or SAN) and play its sound; returns chess.js's move. */
function makeMove(move) {
  const made = ui.chess.move(move);
  if (ui.clock) ui.clock.start(ui.chess.turn());              // the other side's time now runs
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
  ui.board.marks = hintMarks(viewing);                       // drawn by setPosition below
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
  $("play-from-here").hidden = !viewing;
  renderCaptured(shown.fen);
  renderOpening(history, viewing ? ui.view : history.length);
  updateClocks();
  renderThinking(viewing ? ui.thoughts[ui.view] || null : ui.last);
  const humanMoves = history.filter((m) => m.color === ui.human).length;
  $("takeback").disabled = ui.over || ui.busy || humanMoves === 0 || Boolean(ui.clock);   // not in timed games
  $("resign").disabled = ui.over || ui.busy;
  renderHintControls(humanToMove);
  const end = outcome();
  ui.board.el.classList.toggle("viewing", viewing);
  if (viewing) setStatus(`${lookingAt(shown.move)} End, or a click on the board, returns to the game.`);
  else if (ui.over) setStatus(ui.overText);
  else if (ui.busy) setStatus("The bot is thinking...");
  else if (humanToMove) setStatus(`Your move${c.inCheck() ? ": you are in check" : ""}.`
                                  + (ui.adaptive.notice ? ` ${ui.adaptive.notice}` : ""));
  if (end && !ui.over) finish(end.text, end.winner);
  saved.game.save(snapshot());
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

/** The clock faces: time left, whose clock runs, and red when it is low. */
export function updateClocks() {
  const clock = ui.clock;
  const top = ui.board.orientation === "white" ? "b" : "w";
  for (const [id, color] of [["top-clock", top], ["bottom-clock", top === "w" ? "b" : "w"]]) {
    const el = $(id);
    el.hidden = !clock;
    if (!clock) continue;
    const left = clock.left(color);
    el.textContent = formatClock(left);
    el.classList.toggle("running", clock.running === color);
    el.classList.toggle("low", left < Math.min(20000, (clock.base * 1000) / 6));
  }
}

/** Every tenth of a second in a timed game: the clock faces, and the end of the game if a flag fell. */
export function tickClocks() {
  if (!ui.clock || !ui.chess) return;
  updateClocks();
  const flagged = ui.over ? null : ui.clock.flagged();
  if (!flagged) return;
  const name = (c) => (c === "w" ? "White" : "Black");
  const other = flagged === "w" ? "b" : "w";
  ui.busy = false;
  if (canMate(ui.chess.fen(), other)) finish(`${name(other)} wins on time`, other);
  else finish(`${name(flagged)} ran out of time, but ${name(other)} cannot checkmate: a draw`, null);
  render();
}

/** The game's opening after `plies` half-moves (all of them by default), or null. */
export function openingAt(plies = Infinity) {
  const history = ui.chess.history({ verbose: true });
  const fens = [history.length ? history[0].before : ui.chess.fen(), ...history.map((m) => m.after)];
  return openingOf(ui.openings, fens.slice(0, plies + 1));
}

/** The opening's name and ECO code above the move list, as of the position shown (hidden until there is one). */
function renderOpening(history, shownPly) {
  const opening = openingAt(shownPly);
  $("opening").hidden = !opening;
  if (!opening) return;
  $("opening-eco").textContent = opening.eco;
  $("opening-name").textContent = opening.name;
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

// ------------------------------------------------------------------ hints

const TIER_NAMES = { nudge: "1 of 3: a nudge", point: "2 of 3: the piece", show: "3 of 3: the move" };
const MORE_LABELS = { point: "Which piece?", show: "Show the move" };

/** The hint for the position in the game, or null (hints belong to one position). */
function currentHint(viewing = false) {
  return ui.hint && !viewing && !ui.over && ui.chess && ui.hint.fen === ui.chess.fen() ? ui.hint.data : null;
}

function hintMarks(viewing) {
  const hint = currentHint(viewing);
  const marks = {};
  if (hint && hint.from) marks[hint.from] = "hint-from";
  if (hint && hint.to) marks[hint.to] = "hint-to";
  return marks;
}

/** The Hint button (with how many are left), the Unrated badge and the hint panel. */
function renderHintControls(humanToMove) {
  const budget = ui.hints;
  const button = $("hint");
  const fen = ui.chess.fen();
  button.hidden = budget.limit === 0;
  const shownAll = Boolean(currentHint()) && currentHint().tier === "show";
  button.disabled = !humanToMove || ui.hintBusy || !budget.canTake(fen) || shownAll;
  button.textContent = `Hint (${budget.left} left)`;
  $("unrated").hidden = budget.rated;
  const hint = currentHint(ui.view !== null);
  $("hint-panel").hidden = !hint;
  if (!hint) return;
  $("hint-tier").textContent = TIER_NAMES[hint.tier];
  $("hint-text").textContent = hint.text;
  const next = hint.tier === "show" ? null : budget.nextTier(fen);
  $("hint-more").hidden = !next;
  if (next) $("hint-more").textContent = MORE_LABELS[next];
}

/** A hint about the best move for you here: the next tier for this position (nudge, then the piece, then the move).
 *  Stockfish starts on the first hint. */
export async function askHint() {
  if (!ui.chess || ui.over || ui.busy || ui.hintBusy || ui.view !== null || ui.chess.turn() !== ui.human) return null;
  const fen = ui.chess.fen();
  if (!ui.hints.canTake(fen)) {
    setStatus("No hints left in this game.");
    return null;
  }
  ui.hintBusy = true;
  setStatus("Looking for a hint...");
  try {
    if (!ui.hintEngine) ui.hintEngine = ui.makeHintEngine();
    const tier = ui.hints.nextTier(fen);
    const data = await ui.hintEngine.hint(fen, tier, ui.adaptive.mode === "off" ? ui.rating : Math.round(profile().rating),
                                          clockContext());
    if (ui.chess.fen() !== fen) return null;                 // the position changed meanwhile
    ui.hints.take(fen);
    ui.hint = { fen, data };
    return data;
  } catch (err) {
    setStatus(`No hint: ${err.message}`);
    return null;
  } finally {
    ui.hintBusy = false;
    render();
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
  ui.winner = winner;
  if (ui.clock) ui.clock.stop();
  setStatus(text);
  ui.sounds.play(winner === null ? "draw" : winner === ui.human ? "win" : "lose");
  recordResult(winner);
  const banner = $("result-banner");
  const heading = document.createElement("h2");
  heading.className = "result-title";
  heading.textContent = winner === null ? "Draw" : winner === ui.human ? "You won" : "The bot won";
  const p = document.createElement("p");
  p.textContent = text;
  const lines = [heading, p];
  const r = ui.adaptive.recorded;
  if (!r && ui.adaptive.mode !== "off" && !ui.hints.rated) {
    lines.push(Object.assign(document.createElement("p"), { className: "adaptive-result",
      textContent: "Hints were used, so your rating is unchanged." }));
  }
  if (r) {
    const rating = document.createElement("p");
    rating.className = "adaptive-result";
    rating.textContent = `Your rating: ${Math.round(r.before)} \u2192 ${Math.round(r.after)}. Next bot: ${r.next}.`;
    lines.push(rating);
  }
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
  const reviewIt = document.createElement("button");
  reviewIt.type = "button";
  reviewIt.textContent = "Review";
  reviewIt.addEventListener("click", () => reviewGame());
  actions.append(again, reviewIt, close);
  banner.replaceChildren(...lines, actions);
  banner.hidden = false;
  ui.board.setLegalMoves([], null);
  $("takeback").disabled = $("resign").disabled = true;
  saved.game.save(snapshot());
}

/** The clocks as the network was trained on them (seconds left for the side to move and the other, the
 *  time control); untimed games use the model's default of three minutes each. */
export function clockContext() {
  if (!ui.clock) return {};
  const mover = ui.chess.turn();
  const other = mover === "w" ? "b" : "w";
  return { clock: ui.clock.left(mover) / 1000, oppClock: ui.clock.left(other) / 1000, base: ui.clock.base, increment: 0 };
}

/** The bot picks a move as often as players of its rating play it. */
export function botMove() {
  const c = ui.chess;
  const { moves, wdl } = ui.model.predict(c, ui.rating, clockContext());
  const choice = sampleMove(moves, 1, ui.random);
  makeMove(choice.uci);
  ui.last = { san: choice.san, uci: choice.uci, prob: choice.prob, candidates: moves };
  ui.thoughts[c.history().length] = ui.last;
  const live = ui.adaptive.live;
  if (live) {                                               // live: its own chances in the position it moved from
    const before = ui.rating;
    const changed = live.observe(wdl);
    if (changed !== null) {
      ui.rating = changed;
      ui.adaptive.notice = changed > before
        ? `The bot was losing, so it now plays stronger: ${changed}.`
        : `The bot was winning, so it now plays weaker: ${changed}.`;
    }
  }
  return choice;
}

/** How long the bot takes: a fixed pause untimed, a human-like share of its clock in timed games. */
function botDelay() {
  if (!ui.clock) return ui.delay;
  const color = ui.chess.turn();
  return thinkTime(ui.clock.left(color), ui.chess.history().length, ui.random) * ui.thinkScale;
}

function botTurn() {
  ui.busy = true;
  render();
  return new Promise((resolve) => {
    setTimeout(() => {
      if (ui.over) {                                         // its time ran out while it was thinking
        ui.busy = false;
        resolve();
        return;
      }
      botMove();
      ui.busy = false;
      render();
      resolve();
    }, botDelay());
  });
}

/** You play ``uci``; the bot answers. */
export async function play(uci) {
  if (ui.over || ui.busy || ui.chess.turn() !== ui.human) return;
  ui.adaptive.notice = null;
  makeMove(uci);
  render();
  if (!ui.over) await botTurn();
}

export function readForm() {
  const data = new FormData($("new-game"));
  let color = data.get("color");
  if (color === "random") color = ui.random() < 0.5 ? "white" : "black";
  return { rating: Number(data.get("rating")), color, time: data.get("time") || "none",
           mode: data.get("mode") || "standard", position: data.get("position") || "",
           difficulty: data.get("difficulty") || "off", hints: Number(data.get("hints") ?? HINTS.limit) };
}

/** A new game: `start` (a chess.js game, see position.js) to carry on from a position, else the usual start. */
export async function newGame({ rating, color, time = "none", start = null, difficulty = "off", hints = HINTS.limit }) {
  ui.chess = start || new Chess();
  const mode = ["between", "live"].includes(difficulty) ? difficulty : "off";
  if (mode === "between") rating = botRatingFor(profile().rating);   // the bot plays at your level
  ui.time = baseSeconds(time) ? String(time) : "none";
  ui.clock = baseSeconds(time) ? new Clock(baseSeconds(time)) : null;   // null: an untimed game
  const history = ui.chess.history({ verbose: true });
  ui.startFen = history.length ? history[0].before : ui.chess.fen();
  // Games from a position do not count: a won position says little about your level.
  ui.adaptive = { mode, start: rating, rated: ui.startFen === START, recorded: null, notice: null,
                  live: mode === "live" ? new LiveAdjuster(rating) : null };
  ui.view = null;
  ui.thoughts = {};
  ui.hints = new HintBudget(Math.max(0, Number(hints) || 0));
  ui.hint = null;
  ui.rating = rating;
  ui.human = color === "black" ? "b" : "w";
  ui.over = false;
  ui.last = null;
  $("result-banner").hidden = true;
  ui.board.setOrientation(color === "black" ? "black" : "white");
  ui.sounds.play("start");
  render();
  if (ui.chess.turn() !== ui.human) await botTurn();          // the bot's move first (Black, or its turn in a position)
}

/** A new game from the position looked at, with the moves that led there (same rating, side and clock). */
export async function playFromHere() {
  if (ui.view === null) return;
  const history = ui.chess.history({ verbose: true });
  const start = new Chess(history.length ? history[0].before : ui.chess.fen());
  for (const move of history.slice(0, ui.view)) start.move(move.lan);
  await newGame({ rating: ui.adaptive.start, color: colorName(ui.human), time: ui.time || "none", start,
                 difficulty: ui.adaptive.mode, hints: ui.hints.limit });
}

/** What the Play button does: a standard game, or one from the pasted position (an error shown if it cannot be). */
export async function startFromForm() {
  const form = readForm();
  const error = $("position-error");
  error.hidden = true;
  if (form.mode !== "position") return newGame(form);
  const { chess, error: problem } = parsePosition(form.position);
  if (problem) {
    error.textContent = problem;
    error.hidden = false;
    return undefined;
  }
  return newGame({ ...form, start: chess });
}

export function takeback() {
  if (ui.over || ui.busy || ui.clock) return;
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

/** The game in PGN, with who played whom and, once it is over, the result. */
export function pgn() {
  const c = ui.chess;
  if (ui.over) c.setHeader("Result", ui.winner === "w" ? "1-0" : ui.winner === "b" ? "0-1" : "1/2-1/2");
  const opening = openingAt();
  if (opening) {
    c.setHeader("ECO", opening.eco);
    c.setHeader("Opening", opening.name);
  }
  const fromPosition = ui.startFen.split(" ")[0] !== START.split(" ")[0];
  c.setHeader("Event", `Game vs. human-like bot (in the browser${fromPosition ? ", from a position" : ""})`);
  if (fromPosition) {                                         // so other programs can replay it
    c.setHeader("SetUp", "1");
    c.setHeader("FEN", ui.startFen);
  }
  c.setHeader(ui.human === "w" ? "White" : "Black", "You");
  if (!ui.hints.rated) {
    c.setHeader("Hints", String(ui.hints.used));
    c.setHeader("Annotator", "unrated: hints used");
  }
  const a = ui.adaptive;
  const changed = a.live && a.live.changes.length;
  c.setHeader(ui.human === "w" ? "Black" : "White", changed ? `Bot ${a.start}-${ui.rating}` : `Bot ${ui.rating}`);
  if (a.mode !== "off") {
    c.setHeader("Adaptive", a.mode);                          // as the local app writes it
    if (changed) c.setHeader("AdaptiveChanges", a.live.changes.map((x) => `${x.botMove}:${x.rating}`).join(" "));
  }
  return c.pgn() + "\n";
}

/** The FEN of the position on the board (the one looked at, when looking back). */
export function shownFen() {
  return ui.board.fen;
}

/** Copy the position shown as a FEN, to paste into "From a position" here or elsewhere. */
export async function copyFen() {
  const fen = shownFen();
  try {
    await navigator.clipboard.writeText(fen);
    setStatus(`Copied: ${fen}`);
  } catch {
    setStatus(`Your browser did not allow copying. The position: ${fen}`);
  }
  return fen;
}

const openPage = (url) => { window.location.href = url; };

/** Hand the game to another page (through this browser's storage) and open it there. */
function handOver(store, url, open) {
  if (!ui.chess || !ui.chess.history().length) return false;
  if (!store.save(pgn())) {
    setStatus("Your browser did not allow handing the game over: download the PGN and paste it there.");
    return false;
  }
  open(url);
  return true;
}

/** The Analyse page estimates how strong you played in this game. */
export function analyseGame({ open = openPage } = {}) {
  return handOver(saved.handoff, "analyse.html#game", open);
}

/** The Review page judges every move of this game with Stockfish. */
export function reviewGame({ open = openPage } = {}) {
  return handOver(saved.review, "review.html#game", open);
}

function download() {
  const blob = new Blob([pgn()], { type: "application/x-chess-pgn" });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = `game-vs-bot-${ui.rating}.pgn`;
  link.click();
  URL.revokeObjectURL(link.href);
}

export async function init(modelBase = "model/", dataBase = modelBase.replace(/model\/$/, "data/")) {
  loadOpenings(dataBase).then((table) => {                 // names appear once loaded; the game does not wait
    ui.openings = table;
    if (ui.chess) render();
  }).catch(() => {});
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
  applySettings(saved.settings.load());
  for (const radio of $("new-game").elements.difficulty) radio.addEventListener("change", renderAdaptiveNote);
  $("adaptive-reset").addEventListener("click", () => {
    if (window.confirm("Start your rating again from 1200? Your rated games here are forgotten.")) resetProfile();
  });
  renderAdaptiveNote();
  const showMode = () => {
    $("position-fields").hidden = $("new-game").elements.mode.value !== "position";
  };
  for (const radio of $("new-game").elements.mode) radio.addEventListener("change", showMode);
  showMode();
  for (const preset of PRESETS) $("preset").append(new Option(preset.name, preset.fen));
  $("preset").addEventListener("change", () => {             // a practice position fills the box
    if ($("preset").value) $("position-text").value = $("preset").value;
    $("position-error").hidden = true;
  });
  $("new-game").addEventListener("submit", (event) => {
    event.preventDefault();
    const data = new FormData($("new-game"));
    saved.settings.save({ rating: Number($("rating").value), color: data.get("color"), time: data.get("time"),
                          difficulty: data.get("difficulty") || "off", hints: data.get("hints") });
    startFromForm();
  });
  $("flip").addEventListener("click", () => { ui.board.flip(); if (ui.chess) render(); });
  $("board").addEventListener("click", (event) => {         // a click on the board returns to the game
    if (ui.view === null) return;
    event.stopPropagation();
    showPly(null);
  }, true);
  $("play-from-here").addEventListener("click", playFromHere);
  $("first").addEventListener("click", () => showPly(0));
  $("prev").addEventListener("click", () => step(-1));
  $("next").addEventListener("click", () => step(1));
  $("last").addEventListener("click", () => showPly(null));
  $("move-list").addEventListener("click", (event) => {
    const cell = event.target.closest("[data-ply]");
    if (cell) showPly(Number(cell.dataset.ply));
  });
  $("takeback").addEventListener("click", takeback);
  $("hint").addEventListener("click", () => askHint());
  $("hint-more").addEventListener("click", () => askHint());
  $("resign").addEventListener("click", resign);
  $("download").addEventListener("click", download);
  $("copy-fen").addEventListener("click", copyFen);
  $("analyse-game").addEventListener("click", () => analyseGame());
  $("review-game").addEventListener("click", () => reviewGame());
  $("sound").addEventListener("click", () => setSound(!ui.sounds.enabled));
  const unlock = () => {                                   // audio may only start after a click or key
    ui.sounds.unlock();
    for (const type of ["pointerdown", "keydown"]) document.removeEventListener(type, unlock, true);
  };
  for (const type of ["pointerdown", "keydown"]) document.addEventListener(type, unlock, true);
  setSound(saved.sound.load("on") !== "off", { save: false });
  $("volume").addEventListener("input", () => setVolume($("volume").value));
  $("volume").addEventListener("change", () => ui.sounds.play("move"));   // a sample at the new volume
  setVolume(saved.volume.load(70), { save: false });
  document.addEventListener("keydown", (event) => {
    const typing = event.target instanceof Element && event.target.closest("input, select, textarea, button");
    if (typing || event.ctrlKey || event.metaKey || event.altKey) return;
    if (event.key === "f") $("flip").click();
    if (event.key === "m") $("sound").click();
    if (event.key === "h" && !$("hint").hidden && !$("hint").disabled) askHint();
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
  setInterval(tickClocks, 100);
  restore();
}

/** Carry on with a saved game (by default the one saved in this browser); false if there is none or it is broken. */
export function restore(snap = saved.game.load()) {
  if (!snap || snap.version !== 1 || !Array.isArray(snap.moves)) return false;
  let chess;
  try {
    chess = new Chess(snap.start);
    for (const move of snap.moves) chess.move(move);
  } catch {
    saved.game.clear();                                     // unreadable: start afresh
    return false;
  }
  ui.chess = chess;
  ui.startFen = snap.startFen || snap.start;
  ui.human = snap.human === "b" ? "b" : "w";
  ui.rating = Number(snap.rating) || 1500;
  ui.time = snap.time || "none";
  ui.clock = snap.clock ? Clock.fromJSON(snap.clock) : null;
  ui.thoughts = snap.thoughts || {};
  ui.hints = snap.hints ? HintBudget.fromJSON(snap.hints) : new HintBudget(0);
  ui.hint = null;
  const a = snap.adaptive || {};
  const mode = ["between", "live"].includes(a.mode) ? a.mode : "off";
  ui.adaptive = { mode, start: Number(a.start) || ui.rating, rated: a.rated !== false, recorded: a.recorded || null,
                  notice: null, live: mode === "live" && a.live ? LiveAdjuster.fromJSON(a.live) : null };
  const plies = snap.moves.length;
  ui.last = ui.thoughts[plies] || null;
  ui.view = null;
  ui.busy = false;
  ui.over = Boolean(snap.over);
  ui.overText = snap.over ? snap.over.text : null;
  ui.winner = snap.over ? snap.over.winner : null;
  $("result-banner").hidden = true;
  ui.board.setOrientation(snap.orientation === "black" ? "black" : "white");
  render();
  if (!ui.over && chess.turn() !== ui.human) {
    botTurn();                                              // reloaded while the bot was thinking
  } else if (ui.over) {
    setStatus(`Your last game: ${ui.overText}. Press Play for a new one.`);
  } else {
    setStatus(`Welcome back: your game continues. Your move${chess.inCheck() ? ": you are in check" : ""}.`);
  }
  return true;
}

if (document.getElementById("board") && !window.__noAutoInit) init();
