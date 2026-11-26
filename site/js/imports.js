// A player's recent games from Lichess or Chess.com, fetched by the browser from their public APIs (the browser's
// version of src/chess_engine/data/game_import.py). Both return PGN text for games.js.

export const SOURCES = {
  lichess: { name: "Lichess", pattern: /^[A-Za-z0-9](?:[A-Za-z0-9_-]{0,28}[A-Za-z0-9])$/ },
  "chess.com": { name: "Chess.com", pattern: /^[A-Za-z0-9_-]{3,25}$/ },
};

const LICHESS = "https://lichess.org";
const CHESSCOM = "https://api.chess.com";

export class ImportError extends Error {}

/** The username, trimmed, if it can be one on `source`; otherwise an ImportError saying so. */
export function checkUsername(source, username) {
  const info = SOURCES[source];
  if (!info) throw new ImportError(`Unknown source ${source}.`);
  const name = String(username || "").trim();
  if (!info.pattern.test(name)) throw new ImportError(`"${name}" is not a valid ${info.name} username.`);
  return name;
}

async function get(url, { accept, what, fetchImpl }) {
  let response;
  try {
    response = await fetchImpl(url, accept ? { headers: { Accept: accept } } : {});
  } catch {
    throw new ImportError(`${what} could not be reached (offline, or the browser blocked the request). `
      + "You can paste the games\u2019 PGN instead.");
  }
  if (response.status === 404) throw new ImportError(`${what} has no player of that name.`);
  if (response.status === 429) throw new ImportError(`${what} asks us to slow down: try again in a minute.`);
  if (!response.ok) throw new ImportError(`${what} answered with an error (${response.status}).`);
  return response;
}

/**
 * Why Lichess answered 404 to a games request. Its games export answers an unknown player with an HTML page, and so
 * it does when it refuses a request (it blocks some automated browsers), so the user endpoint (JSON) decides.
 */
async function lichessNotFound(name, fetchImpl) {
  const refused = new ImportError("Lichess refused to send the games (it blocks some automated browsers and "
    + "privacy-protected requests). You can paste the games\u2019 PGN from lichess.org instead.");
  let response;
  try {
    response = await fetchImpl(`${LICHESS}/api/user/${encodeURIComponent(name)}`, {});
  } catch {
    return refused;
  }
  let data = null;
  try {
    data = await response.json();
  } catch {
    return refused;                                   // not the API's JSON: the request was turned away
  }
  if (response.status === 404 && data && data.error) return new ImportError("Lichess has no player of that name.");
  if (response.ok && data && data.username) {
    return new ImportError(`Lichess has ${data.username}, but refused to send their games (it blocks some automated `
      + "browsers). You can paste the games\u2019 PGN from lichess.org instead.");
  }
  return refused;
}

/** The player's latest `max` games on Lichess as PGN, with clocks, newest first. */
export async function fetchLichess(username, { max = 10, fetchImpl = fetch } = {}) {
  const name = checkUsername("lichess", username);
  const query = new URLSearchParams({ max: String(max), clocks: "true", opening: "true", moves: "true",
                                      tags: "true", evals: "false" });
  const url = `${LICHESS}/api/games/user/${encodeURIComponent(name)}?${query}`;
  let response;
  try {
    response = await get(url, { accept: "application/x-chess-pgn", what: "Lichess", fetchImpl });
  } catch (err) {
    if (err instanceof ImportError && err.message === "Lichess has no player of that name.") {
      throw await lichessNotFound(name, fetchImpl);
    }
    throw err;
  }
  return (await response.text()).trim();
}

/** The player's latest `max` standard-chess games on Chess.com as PGN, newest first (from the monthly archives). */
export async function fetchChessCom(username, { max = 10, maxArchives = 6, fetchImpl = fetch } = {}) {
  const name = checkUsername("chess.com", username).toLowerCase();
  const what = "Chess.com";
  const list = await (await get(`${CHESSCOM}/pub/player/${encodeURIComponent(name)}/games/archives`,
                                { what, fetchImpl })).json();
  const archives = Array.isArray(list.archives) ? list.archives : [];
  const pgns = [];
  for (const url of archives.slice(-maxArchives).reverse()) {
    if (typeof url !== "string" || !url.startsWith(`${CHESSCOM}/`)) continue;   // never another host
    const month = await (await get(url, { what, fetchImpl })).json();
    for (const entry of (Array.isArray(month.games) ? month.games : []).slice().reverse()) {
      if ((entry.rules || "chess") !== "chess" || !entry.pgn) continue;
      pgns.push(entry.pgn.trim());
      if (pgns.length >= max) return pgns.join("\n\n");
    }
  }
  return pgns.join("\n\n");
}

/** Recent games of `username` on `source` ("lichess" or "chess.com") as PGN text. */
export function fetchGames(source, username, options = {}) {
  if (source === "chess.com") return fetchChessCom(username, options);
  return fetchLichess(username, options);
}
