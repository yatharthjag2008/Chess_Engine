// Remembering things in this browser (localStorage), without ever breaking the page: private windows,
// blocked site data or a full quota just mean nothing is remembered.
//
//   const saved = stored("human-chess-site.game");
//   saved.save({ moves: ["e2e4"] });  saved.load();  // -> { moves: ["e2e4"] }, or null
//   saved.clear();

function defaultStorage() {
  try {
    return window.localStorage;
  } catch {
    return null;
  }
}

/** A JSON value kept under `key`. */
export function stored(key, storage = defaultStorage()) {
  return {
    key,
    load(fallback = null) {
      try {
        const text = storage ? storage.getItem(key) : null;
        return text === null ? fallback : JSON.parse(text);
      } catch {
        return fallback;                // unreadable or not JSON: as if nothing were saved
      }
    },
    save(value) {
      try {
        if (!storage) return false;
        storage.setItem(key, JSON.stringify(value));
        return true;
      } catch {
        return false;
      }
    },
    clear() {
      try {
        if (storage) storage.removeItem(key);
      } catch {
        // nothing to do
      }
    },
  };
}
