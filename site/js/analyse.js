// The Analyse page: estimate how strong each player in pasted games played, with the model in the browser.

import { Chess } from "../vendor/chess.js";
import { loadModel } from "./engine.js";
import { playerGames } from "./games.js";
import { SOURCES, fetchGames } from "./imports.js";
import { stored } from "./storage.js";
import { ScoringPool, scoreSide } from "./pool.js";

const handoff = stored("human-chess-site.analyse");   // a game handed over by the play page
import { describeEstimate, estimateFromBucketLL, loadRatingData, moveBucketLL, opponentFits } from "./rating.js";

const $ = (id) => document.getElementById(id);

export const page = { model: null, data: null, running: false, fetchImpl: (...args) => fetch(...args),
                      pool: null };   // background workers that score the moves (null: on the page)

const pause = () => new Promise((resolve) => setTimeout(resolve, 0));

/**
 * Per-move bucket log-likelihoods of one player's side of a game (the opponent's rating from the PGN when known).
 * Yields to the page between moves so it stays responsive; `onMove(done)` reports progress.
 */
export async function sideLikelihoods(model, side, onMove = () => {}) {
  const rows = [];
  for (const move of side.moves) {
    rows.push(moveBucketLL(model, { fen: move.fen, uci: move.uci, oppRating: side.oppRating, clock: move.clock,
                                    oppClock: move.oppClock }, side.base, side.increment));
    onMove(rows.length);
    await pause();
  }
  return rows;
}

const COLOR = { w: "White", b: "Black" };

/** "Ann (White, rated 1520 in the game, against 1488)". */
export function sideLabel(side) {
  const facts = [COLOR[side.color]];
  if (side.rating) facts.push(`rated ${side.rating} in the game`);
  if (side.oppRating) facts.push(`against ${side.oppRating}`);
  return `${side.name} (${facts.join(", ")})`;
}

/** Estimate every side in `text` (or only `player`'s); returns { results, overall, unreadable }. */
export async function estimateText(text, { player = "", onProgress = () => {} } = {}) {
  const { games, unreadable } = playerGames(text, { player });
  if (!games.length) {
    const why = unreadable ? `${unreadable} game${unreadable === 1 ? "" : "s"} could not be read, and none could.`
      : player ? `No game of "${player}" was found (the name is matched on the White and Black headers).`
        : "Paste at least one game in PGN.";
    return { results: [], overall: null, unreadable, error: why };
  }
  const total = games.reduce((n, g) => n + g.moves.length, 0);
  let done = 0;
  const results = [];
  const all = [];
  for (const side of games) {
    const before = done;
    const rows = await scoreSide(page.pool, page.model, side, (n) => onProgress(before + n, total));
    done += rows.length;
    all.push(...rows);
    const movesOnly = estimateFromBucketLL(rows, page.data);
    const paired = side.oppRating ? estimateFromBucketLL(rows, page.data, { oppRatings: [side.oppRating] }) : null;
    const fits = Boolean(paired && paired.withOpponent && opponentFits(movesOnly, side.oppRating));
    results.push({ side, estimate: paired || movesOnly, movesOnly, fits });
  }
  let overall = null;
  if (player && results.length > 1) {
    // Every game's moves count; only the opponents whose ratings fit the moves count as well.
    const opps = results.filter((r) => r.fits).map((r) => r.side.oppRating);
    overall = estimateFromBucketLL(all, page.data, { oppRatings: opps.length ? opps : null });
    overall.opponentsUsed = opps.length;
    overall.games = results.length;
  }
  return { results, overall, unreadable, error: null };
}

function render({ results, overall, unreadable }, player) {
  $("estimate-results").hidden = false;
  $("overall").hidden = !overall;
  if (overall) {
    const used = overall.opponentsUsed;
    const left = overall.games - used;
    $("overall").textContent = `All ${results.length} games of ${player}: ${describeEstimate(overall)}`
      + (overall.withOpponent ? `, with the opponents' ratings${left ? ` (${left} game${left === 1 ? "" : "s"} left out `
        + `where the opponent's rating did not fit the moves)` : ""}` : ", from the moves alone") + ".";
  }
  const items = results.map(({ side, estimate, movesOnly, fits }) => {
    const li = document.createElement("li");
    const who = document.createElement("strong");
    who.textContent = sideLabel(side);
    const what = document.createElement("span");
    what.textContent = estimate.withOpponent
      ? `: ${describeEstimate(estimate)}, with the opponent's rating. From the moves alone: ${describeEstimate(movesOnly)}.`
      : `: ${describeEstimate(estimate)}, from the moves alone.`;
    li.append(who, what);
    if (estimate.withOpponent && !fits) {
      const warning = document.createElement("p");
      warning.className = "form-error misfit";
      warning.textContent = `These disagree: the opponent's rating (${side.oppRating}) is far outside what the moves `
        + `suggest. If this was not a game between similarly rated players (a casual challenge, say), go by the moves.`;
      li.append(warning);
    }
    const opponent = side.color === "w" ? side.headers.Black : side.headers.White;
    if (opponent) {
      const game = document.createElement("span");
      game.className = "hint";
      game.textContent = ` vs. ${opponent}${side.headers.Date ? `, ${side.headers.Date}` : ""}, ${side.result}`;
      li.append(game);
    }
    return li;
  });
  $("estimate-list").replaceChildren(...items);
  if (unreadable) {
    const li = document.createElement("li");
    li.className = "hint";
    li.textContent = `${unreadable} game${unreadable === 1 ? "" : "s"} could not be read and ${unreadable === 1 ? "was" : "were"} left out.`;
    $("estimate-list").append(li);
  }
}

/** What the Fetch button does: the player's recent games into the box, and their name as the player. */
export async function fetchRecent() {
  const source = $("source").value;
  const username = $("username").value.trim();
  $("estimate-error").hidden = true;
  $("fetch").disabled = true;
  $("estimate-status").textContent = `Fetching ${username}'s games from ${SOURCES[source].name}...`;
  try {
    const text = await fetchGames(source, username, { max: 10, fetchImpl: page.fetchImpl });
    const { games } = playerGames(text, { player: username });
    if (!games.length) throw new Error(`No standard games of ${username} were found on ${SOURCES[source].name}.`);
    $("pgn-text").value = text;
    $("player-name").value = games[0].name;              // as the site spells it
    $("estimate-status").textContent = `Fetched ${games.length} game${games.length === 1 ? "" : "s"} of `
      + `${games[0].name} from ${SOURCES[source].name}. Press Estimate.`;
    return games.length;
  } catch (err) {
    $("estimate-error").textContent = err.message;
    $("estimate-error").hidden = false;
    $("estimate-status").textContent = "";
    return 0;
  } finally {
    $("fetch").disabled = false;
  }
}

/** What the Estimate button does. */
export async function estimate() {
  if (page.running || !page.model) return null;
  page.running = true;
  $("estimate").disabled = true;
  $("estimate-error").hidden = true;
  const player = $("player-name").value.trim();
  try {
    const result = await estimateText($("pgn-text").value, {
      player,
      onProgress: (done, total) => { $("estimate-status").textContent = `Scoring move ${done} of ${total}...`; },
    });
    if (result.error) {
      $("estimate-error").textContent = result.error;
      $("estimate-error").hidden = false;
      $("estimate-status").textContent = "";
    } else {
      render(result, player);
      const moves = result.results.reduce((n, r) => n + r.estimate.moves, 0);
      $("estimate-status").textContent = `Done: ${moves} moves scored at every rating.`;
    }
    return result;
  } finally {
    page.running = false;
    $("estimate").disabled = false;
  }
}

export async function init(modelBase = "model/", dataBase = modelBase.replace(/model\/$/, "data/")) {
  $("estimate-form").addEventListener("submit", (event) => {
    event.preventDefault();
    estimate();
  });
  $("fetch").addEventListener("click", fetchRecent);
  $("username").addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      fetchRecent();
    }
  });
  try {
    [page.model, page.data] = await Promise.all([loadModel(modelBase), loadRatingData(dataBase)]);
    page.pool = typeof Worker === "undefined" ? null : new ScoringPool({ modelBase });
    if (page.pool) page.pool.start().catch(() => { page.pool = null; });   // warm up while you paste
  } catch (err) {
    $("estimate-status").textContent = `The model could not be loaded (${err.message}).`;
    return;
  }
  if (!page.model.selfTest(Chess).ok) {
    $("estimate-status").textContent = "Warning: this browser's results differ from PyTorch's.";
  } else {
    $("estimate-status").textContent = "Paste games and press Estimate.";
  }
  $("estimate").disabled = false;
  if (window.location.hash === "#game") await openHandedGame();
}

/** A game from the play page ("Analyse this game"): estimate how strong you played in it. */
export async function openHandedGame() {
  const text = handoff.load(null);
  if (typeof text !== "string" || !text.trim()) return null;
  handoff.clear();
  $("pgn-text").value = text;
  $("player-name").value = "You";
  return estimate();
}

if (document.getElementById("estimate-form") && !window.__noAutoInit) init();
