// Calls to the game API (see src/chess_engine/api/app.py). Every call returns parsed JSON or throws ApiError.

export class ApiError extends Error {
  constructor(status, code, detail) {
    super(detail || code || `HTTP ${status}`);
    this.status = status;
    this.code = code;
  }
}

async function request(method, path, body) {
  let response;
  try {
    response = await fetch(path, {
      method,
      headers: body === undefined ? {} : { "Content-Type": "application/json" },
      body: body === undefined ? undefined : JSON.stringify(body),
    });
  } catch (err) {
    throw new ApiError(0, "network", "cannot reach the server (is scripts/serve.py running?)");
  }
  const text = await response.text();
  let data = null;
  try { data = text ? JSON.parse(text) : null; } catch { data = text; }
  if (!response.ok) {
    throw new ApiError(response.status, data && data.error, (data && data.detail) || text);
  }
  return data;
}

export const api = {
  health: () => request("GET", "/api/health"),
  newGame: (options) => request("POST", "/api/games", options),
  state: (id) => request("GET", `/api/games/${encodeURIComponent(id)}`),
  play: (id, move, reply = true) => request("POST", `/api/games/${encodeURIComponent(id)}/moves`, { move, reply }),
  botMove: (id) => request("POST", `/api/games/${encodeURIComponent(id)}/bot-move`),
  resign: (id) => request("POST", `/api/games/${encodeURIComponent(id)}/resign`),
  takeback: (id) => request("POST", `/api/games/${encodeURIComponent(id)}/takeback`),
  pgnUrl: (id) => `/api/games/${encodeURIComponent(id)}/pgn`,
};
