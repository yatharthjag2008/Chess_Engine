// Opening names (the Lichess data set, built into data/openings.json by scripts/build_openings.py).
//
//   const table = await loadOpenings();
//   openingOf(table, fens);   // fens: the positions of the game in order -> { eco, name, ply } or null
//
// Names are filed under positions (placement, side to move, castling), so transpositions are found. The
// game's opening is its latest named position: once the game leaves the book the name stays.

let loading = null;

/** The name table, fetched once (later calls share the same promise). */
export function loadOpenings(base = "data/") {
  if (!loading) {
    loading = fetch(base + "openings.json")
      .then((response) => {
        if (!response.ok) throw new Error("openings.json not found");
        return response.json();
      })
      .catch((err) => {
        loading = null;                     // try again next time
        throw err;
      });
  }
  return loading;
}

/** The key a position is filed under: the first three FEN fields. */
export function positionKey(fen) {
  return fen.split(" ").slice(0, 3).join(" ");
}

/** The latest named position among `fens` (in game order): { eco, name, ply }, or null. */
export function openingOf(table, fens) {
  if (!table) return null;
  for (let ply = fens.length - 1; ply >= 0; ply--) {
    const entry = table[positionKey(fens[ply])];
    if (entry) return { eco: entry[0], name: entry[1], ply };
  }
  return null;
}
