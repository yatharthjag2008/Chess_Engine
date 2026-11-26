// The play page: start a game, move on the board, show the bot's replies.
//
// All game state lives on the server; the page shows the latest state it received
// (render) and sends what the player does (moves, buttons) to the API. The game id
// is kept in the URL (#game=<id>), so reloading the page continues the game.

import { api, ApiError } from "./api.js";
import { Board } from "./board.js";

const $ = (id) => document.getElementById(id);

export const ui = {
  game: null,        // the latest game state from the server
  busy: false,       // a request is in flight (the board is locked meanwhile)
  board: null,
  clock: null,       // { white, black, running, at }: the server's clock and when it was received
  flagCheck: false,  // a state refresh after a flag fall is in flight
  hint: null,        // { id, fen, data }: the latest hint, shown while that game and position are on the board
  adaptive: null,    // your adaptive-difficulty profile from the server
  adaptiveFor: null, // the finished game whose result was last loaded into it
  styles: [],        // trained styles and ghosts the bot can play in
  savedStyle: "",    // the style chosen last time (from this browser's settings)
};

const LOW_TIME = 20; // seconds: the clock turns red below this

/** "m:ss", or "0:ss.t" under 10 seconds (never negative). */
export function formatClock(seconds) {
  const s = Math.max(0, seconds);
  if (s < 10) return `0:${s.toFixed(1).padStart(4, "0")}`;
  const whole = Math.floor(s);
  return `${Math.floor(whole / 60)}:${String(whole % 60).padStart(2, "0")}`;
}

/** Both clocks `now` (ms, performance.now()), counting down the running one since it was received. */
export function clockTimes(clock, now) {
  const elapsed = (now - clock.at) / 1000;
  return {
    white: clock.white - (clock.running === "white" ? elapsed : 0),
    black: clock.black - (clock.running === "black" ? elapsed : 0),
  };
}

function updateClocks() {
  const clock = ui.clock;
  const top = $("top-clock"), bottom = $("bottom-clock");
  top.hidden = bottom.hidden = !clock;
  if (!clock) return;
  const times = clockTimes(clock, performance.now());
  const bottomColor = ui.board.orientation;
  const topColor = bottomColor === "white" ? "black" : "white";
  for (const [el, color] of [[top, topColor], [bottom, bottomColor]]) {
    el.textContent = formatClock(times[color]);
    el.classList.toggle("running", clock.running === color);
    el.classList.toggle("low", times[color] < LOW_TIME);
  }
  const flagged = clock.running && times[clock.running] <= 0;
  if (flagged && !ui.flagCheck && ui.game && ui.game.status === "ongoing") {
    ui.flagCheck = true;      // the server decides: it ends the game by time forfeit
    api.state(ui.game.id).then((state) => render(state)).catch(() => {}).finally(() => { ui.flagCheck = false; });
  }
}

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

function humanMoves(state) {
  // Moves you made = all moves minus the bot's (the bot moved first if it had White).
  const botFirst = state.human_color === "black";
  const n = state.moves_san.length;
  return botFirst ? Math.floor(n / 2) : Math.ceil(n / 2);
}

function renderButtons(state) {
  const over = state.status === "over";
  const hint = $("hint");
  const hints = state.hints || { limit: 0, left: 0, rated: true };
  const shownHere = Boolean(currentHint(state));
  hint.hidden = hints.limit === 0;
  hint.disabled = over || ui.busy || state.turn !== state.human_color || (hints.left === 0 && !shownHere);
  hint.textContent = `Hint (${hints.left} left)`;
  $("unrated").hidden = hints.rated;
  const resign = $("resign");
  resign.disabled = over || ui.busy;
  if (over) resetResign();
  $("takeback").disabled = over || ui.busy || Boolean(state.clock) || humanMoves(state) === 0;
  $("takeback").title = state.clock ? "No takebacks in timed games" : "";
}

function resetResign() {
  const resign = $("resign");
  resign.dataset.confirm = "";
  resign.textContent = "Resign";
}

// ------------------------------------------------------------------ hints

const TIER_NAMES = { nudge: "1 of 3: a nudge", point: "2 of 3: the piece", show: "3 of 3: the move" };
const MORE_LABELS = { point: "Which piece?", show: "Show the move" };

/** The hint for the position on the board (hints belong to one position), or null. */
function currentHint(state) {
  return ui.hint && state && ui.hint.id === state.id && ui.hint.fen === state.fen && state.status === "ongoing"
    ? ui.hint.data : null;
}

function hintMarks(state) {
  const hint = currentHint(state);
  const marks = {};
  if (hint && hint.from_square) marks[hint.from_square] = "hint-from";
  if (hint && hint.to_square) marks[hint.to_square] = "hint-to";
  return marks;
}

export function renderHint(state) {
  const panel = $("hint-panel");
  const hint = currentHint(state);
  panel.hidden = !hint;
  if (!hint) return;
  $("hint-tier").textContent = TIER_NAMES[hint.tier];
  $("hint-text").textContent = hint.text;
  const more = $("hint-more");
  more.hidden = !hint.next_tier;
  if (hint.next_tier) more.textContent = MORE_LABELS[hint.next_tier];
}

/** Ask for a hint (the next tier for this position, or ``tier``); it is drawn on the board and in the panel. */
export async function askHint(tier = null) {
  const game = ui.game;
  if (!game || ui.busy || game.status !== "ongoing" || game.turn !== game.human_color) return null;
  $("hint").disabled = true;
  try {
    const data = await api.hint(game.id, tier);
    ui.hint = { id: game.id, fen: game.fen, data };
    if (ui.game === game) {
      ui.game = { ...game, hints: data.hints };
      ui.board.setMarks(hintMarks(ui.game));
      renderButtons(ui.game);
      renderHint(ui.game);
    }
    return data;
  } catch (err) {
    showError(err);
    renderButtons(ui.game);
    return null;
  }
}

/** The result banner over the board when the game has ended. */
export function renderResult(state) {
  const banner = $("result-banner");
  if (state.status !== "over") {
    banner.hidden = true;
    return;
  }
  const outcome = state.outcome;
  const won = outcome.winner === state.human_color;
  const title = outcome.winner === null ? "Draw" : won ? "You won" : "The bot won";
  const heading = document.createElement("h2");
  heading.textContent = title;
  const text = document.createElement("p");
  text.textContent = outcome.description;
  const rating = adaptiveSummary(state);
  if (rating) {
    const line = document.createElement("span");
    line.className = "adaptive-result";
    line.textContent = rating;
    text.append(document.createElement("br"), line);
    if (ui.adaptiveFor !== state.id) {      // once per game: the form shows your new rating for the next one
      ui.adaptiveFor = state.id;
      ui.adaptiveLoading = loadAdaptive();
    }
  }
  const actions = document.createElement("div");
  actions.className = "actions";
  const again = button("New game", () => startGame(readForm($("new-game"))), "primary");
  again.style.width = "auto";
  const download = document.createElement("a");
  download.href = api.pgnUrl(state.id);
  download.download = `game-vs-bot-${state.bot_rating}.pgn`;
  download.textContent = "Download PGN";
  download.className = "button-link";
  const copy = button("Copy PGN", () => copyPgn(state.id, copy));
  const review = document.createElement("a");
  review.href = `review.html#game=${encodeURIComponent(state.id)}`;
  review.textContent = "Review this game";
  review.className = "button-link review-link";
  const close = button("Close", () => { banner.hidden = true; });
  actions.append(again, download, review, copy, close);
  banner.replaceChildren(heading, text, actions);
  banner.hidden = false;
}

function button(label, onClick, className = "") {
  const b = document.createElement("button");
  b.type = "button";
  b.textContent = label;
  if (className) b.className = className;
  b.addEventListener("click", onClick);
  return b;
}

async function copyPgn(id, el) {
  try {
    const pgn = await (await fetch(api.pgnUrl(id))).text();
    await navigator.clipboard.writeText(pgn);
    el.textContent = "Copied";
  } catch {
    el.textContent = "Copy failed";
  }
}

/** Show a game state from the server. */
export function render(state) {
  ui.game = state;
  ui.clock = state.clock ? { ...state.clock, running: state.status === "ongoing" ? state.clock.running : null,
                              at: performance.now() } : null;
  const board = ui.board;
  const humanToMove = state.status === "ongoing" && state.turn === state.human_color;
  board.marks = hintMarks(state);
  board.setPosition(state.fen, { lastMove: state.last_move, check: state.check });
  const botToMove = state.status === "ongoing" && state.turn !== state.human_color;   // meanwhile you may premove
  board.setLegalMoves(humanToMove && !ui.busy ? state.legal_moves : [], humanToMove && !ui.busy ? state.turn : null,
                      { premover: botToMove ? state.human_color : null });
  const bottomIsHuman = board.orientation === state.human_color;
  const bot = botLabel(state);
  const you = state.your_rating ? `You (${state.your_rating})` : "You";
  $("top-name").textContent = bottomIsHuman ? bot : you;
  $("bottom-name").textContent = bottomIsHuman ? you : bot;
  $("game-panel").hidden = false;
  renderMoves(state);
  renderThinking(state);
  updateClocks();
  renderButtons(state);
  renderHint(state);
  renderResult(state);
  if (location.hash !== `#game=${state.id}`) history.replaceState(null, "", `#game=${state.id}`);
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

const SETTINGS_KEY = "human-chess-bot.settings";

/** Remember the new-game settings in this browser (best effort: storage may be unavailable). */
export function saveSettings(options) {
  try { localStorage.setItem(SETTINGS_KEY, JSON.stringify(options)); } catch { /* private mode etc. */ }
}

export function loadSettings(form) {
  let saved = null;
  try { saved = JSON.parse(localStorage.getItem(SETTINGS_KEY) || "null"); } catch { saved = null; }
  if (!saved || typeof saved !== "object") return;
  if (Number.isFinite(saved.rating)) form.elements.rating.value = String(saved.rating);
  const color = form.querySelector(`input[name="color"][value="${saved.color}"]`);
  if (color) color.checked = true;
  const select = form.elements.time_control;
  if ([...select.options].some((o) => o.value === (saved.time_control || ""))) select.value = saved.time_control || "";
  ui.savedStyle = typeof saved.style === "string" ? saved.style : "";   // applied once the styles are loaded
  const adaptive = form.querySelector(`input[name="adaptive"][value="${saved.adaptive}"]`);
  if (adaptive) adaptive.checked = true;
  const hints = form.elements.hint_limit;
  if (hints && [...hints.options].some((o) => o.value === String(saved.hint_limit))) hints.value = String(saved.hint_limit);
}

export function readForm(form) {
  const data = new FormData(form);
  // The slider is disabled in "between games" mode, and FormData skips disabled fields.
  const options = { rating: Number(form.elements.rating.value), color: data.get("color") };
  const time = data.get("time_control");
  if (time) options.time_control = time;
  if (data.get("hint_limit") !== null) options.hint_limit = Number(data.get("hint_limit"));
  if (data.get("adaptive")) options.adaptive = data.get("adaptive");
  if (data.get("style")) options.style = data.get("style");
  return options;
}

// ------------------------------------------------------------------ adaptive difficulty

/** "Bot 1550", with an arrow when live adaptation has moved it from where the game started. */
export function botLabel(state) {
  const a = state.adaptive;
  const name = `Bot ${state.bot_rating}` + (state.style ? ` (${state.style.player})` : "");
  if (!a || a.mode !== "live" || state.bot_rating === a.start_rating) return name;
  return `${name} ${state.bot_rating > a.start_rating ? "\u2191" : "\u2193"}`;
}

// ------------------------------------------------------------------ styles (step 19)

/** "Ann: style, 120 games" / "me: ghost, games up to 2025-05-31". */
export function styleLabel(style) {
  const what = style.kind === "ghost" ? `ghost, games up to ${style.until}` : `style, ${style.games} games`;
  return `${style.player}: ${what}`;
}

/** Fill the Style choice from the server (hidden when no style has been trained). */
export async function loadStyles() {
  let styles = [];
  try {
    styles = await api.styles();
  } catch {
    styles = [];
  }
  ui.styles = styles;
  const select = $("style");
  const keep = select.value || ui.savedStyle || "";
  select.replaceChildren(select.options[0]);
  for (const style of styles) {
    const option = document.createElement("option");
    option.value = style.name;
    option.textContent = styleLabel(style);
    select.append(option);
  }
  if (styles.some((s) => s.name === keep)) select.value = keep;
  $("style-field").hidden = styles.length === 0;
  renderStyleNote();
  return styles;
}

export function renderStyleNote() {
  const style = (ui.styles || []).find((s) => s.name === $("style").value);
  $("style-note").textContent = !style ? "" : style.weight < 1
    ? `Few games, so it is blended with the rating model (weight ${style.weight.toFixed(2)}).`
    : `Imitates ${style.player}'s choices fully.`;
}

/** The sentence about your adaptive rating once a game is over. */
export function adaptiveSummary(state) {
  const a = state.adaptive;
  if (!a || a.mode === "off" || state.status !== "over") return "";
  if (!a.result) return state.hints && !state.hints.rated ? "Hints were used, so your adaptive rating is unchanged." : "";
  const r = a.result;
  return `Your rating: ${Math.round(r.rating_before)} \u2192 ${Math.round(r.rating_after)}. Next bot: ${r.next_bot_rating}.`;
}

const ADAPTIVE_NOTES = {
  off: "The bot plays at the rating you choose.",
  between: "The bot plays at your level: your rating, updated after every game without hints.",
  live: "The bot starts at the rating you choose and adapts during the game (at most 200 points).",
};

/** Your adaptive rating, from the server (null if it cannot be read). */
export async function loadAdaptive() {
  try {
    ui.adaptive = await api.adaptive();
  } catch {
    ui.adaptive = null;
  }
  renderDifficulty();
  return ui.adaptive;
}

/** The difficulty note, and the rating slider: in "between games" it shows your rating and is not used. */
export function renderDifficulty() {
  const form = $("new-game");
  const mode = new FormData(form).get("adaptive") || "off";
  const rating = $("rating");
  const between = mode === "between";
  rating.disabled = between;
  let note = ADAPTIVE_NOTES[mode];
  if (between && ui.adaptive) {
    rating.value = String(ui.adaptive.next_bot_rating);
    $("rating-value").textContent = String(ui.adaptive.next_bot_rating);
    note += ` Your rating: ${Math.round(ui.adaptive.rating)} (${ui.adaptive.games} rated game${ui.adaptive.games === 1 ? "" : "s"}).`;
  }
  $("adaptive-note").textContent = note;
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

export async function resign() {
  const el = $("resign");
  if (!ui.game || ui.busy) return;
  if (el.dataset.confirm !== "yes") {        // first click asks for confirmation
    el.dataset.confirm = "yes";
    el.textContent = "Click again to resign";
    setTimeout(resetResign, 4000);
    return;
  }
  resetResign();
  await act(() => api.resign(ui.game.id));
}

export async function takeback() {
  if (!ui.game || ui.busy) return;
  await act(() => api.takeback(ui.game.id));
}

async function act(call) {
  ui.busy = true;
  try {
    const state = await call();
    ui.busy = false;
    render(state);
  } catch (err) {
    ui.busy = false;
    render(ui.game);
    showError(err);
  }
}

/** Continue the game named in the URL (#game=<id>), if the server still has it. */
export async function resumeFromUrl() {
  const match = location.hash.match(/^#game=([\w-]+)$/);
  if (!match) return false;
  try {
    const state = await api.state(match[1]);
    ui.board.setOrientation(state.human_color);
    render(state);
    return true;
  } catch {
    history.replaceState(null, "", location.pathname + location.search);
    setStatus("That game is no longer on the server. Start a new one.");
    return false;
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
  loadSettings($("new-game"));
  showRating();
  for (const radio of document.querySelectorAll('#new-game input[name="adaptive"]')) {
    radio.addEventListener("change", () => renderDifficulty());
  }
  renderDifficulty();
  loadAdaptive();
  $("style").addEventListener("change", renderStyleNote);
  loadStyles();
  $("new-game").addEventListener("submit", (event) => {
    event.preventDefault();
    const options = readForm(event.target);
    saveSettings(options);
    startGame(options);
  });
  document.addEventListener("keydown", (event) => {
    const typing = event.target instanceof Element && event.target.closest("input, select, textarea, button");
    if (typing || event.ctrlKey || event.metaKey || event.altKey) return;
    if (event.key === "f") $("flip").click();
    if (event.key === "h" && !$("hint").hidden && !$("hint").disabled) askHint();
    if (event.key === "Escape") ui.board.cancelPremove();   // drops the selection and any premove
  });
  setInterval(updateClocks, 100);
  $("resign").addEventListener("click", resign);
  $("takeback").addEventListener("click", takeback);
  $("hint").addEventListener("click", () => askHint());
  $("hint-more").addEventListener("click", () => askHint());
  $("flip").addEventListener("click", () => {
    ui.board.flip();
    if (ui.game) render(ui.game);
  });
}

if (document.getElementById("board") && !window.__noAutoInit) {
  init();
  resumeFromUrl();
}
