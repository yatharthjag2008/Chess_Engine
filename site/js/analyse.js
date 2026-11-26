// The Analyse page: estimate how strong each player in pasted games played, with the model in the browser.

import { Chess } from "../vendor/chess.js";
import { loadModel } from "./engine.js";
import { playerGames } from "./games.js";
import { describeEstimate, estimateFromBucketLL, loadRatingData, moveBucketLL } from "./rating.js";

const $ = (id) => document.getElementById(id);

export const page = { model: null, data: null, running: false };

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
    const rows = await sideLikelihoods(page.model, side, (n) => onProgress(before + n, total));
    done += rows.length;
    all.push(...rows);
    results.push({ side, estimate: estimateFromBucketLL(rows, page.data) });
  }
  const overall = player && results.length > 1 ? estimateFromBucketLL(all, page.data) : null;
  return { results, overall, unreadable, error: null };
}

function render({ results, overall, unreadable }, player) {
  $("estimate-results").hidden = false;
  $("overall").hidden = !overall;
  if (overall) $("overall").textContent = `All ${results.length} games of ${player}: ${describeEstimate(overall)}.`;
  const items = results.map(({ side, estimate }) => {
    const li = document.createElement("li");
    const who = document.createElement("strong");
    who.textContent = sideLabel(side);
    const what = document.createElement("span");
    what.textContent = `: ${describeEstimate(estimate)}`;
    li.append(who, what);
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
  try {
    [page.model, page.data] = await Promise.all([loadModel(modelBase), loadRatingData(dataBase)]);
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
}

if (document.getElementById("estimate-form") && !window.__noAutoInit) init();
