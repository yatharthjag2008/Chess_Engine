// Chess clocks: each side's time left, the side whose clock runs, and who ran out.
//
//   const clock = new Clock(180);          // 3 minutes each
//   clock.start("b");                      // after White's first move Black's time runs
//   clock.left("b");                       // milliseconds left
//   clock.flagged();                       // "w" / "b" once a side has no time left, else null
//
// Time is measured on the wall clock (Date.now), so a saved clock (toJSON) carries on after a reload as
// it would on a server: time passes for the side to move while the page is closed.

export const TIME_CONTROLS = [
  { id: "none", label: "No clock", base: null },
  { id: "1", label: "1 min", base: 60 },
  { id: "3", label: "3 min", base: 180 },
  { id: "5", label: "5 min", base: 300 },
  { id: "10", label: "10 min", base: 600 },
];

/** The base time in seconds of a time-control id ("3" -> 180), or null for no clock. */
export function baseSeconds(id) {
  const control = TIME_CONTROLS.find((c) => c.id === String(id));
  return control ? control.base : null;
}

export class Clock {
  constructor(base, { now = () => Date.now() } = {}) {
    this.base = base;                                   // seconds per side
    this.now = now;
    this.remaining = { w: base * 1000, b: base * 1000 };   // ms, as of `since` for the running side
    this.running = null;                                // "w", "b" or null
    this.since = null;
  }

  /** Milliseconds left for `color` (never below zero). */
  left(color) {
    const spent = this.running === color ? this.now() - this.since : 0;
    return Math.max(0, this.remaining[color] - spent);
  }

  /** Stop the running clock, charging it the time since it started. */
  stop() {
    if (this.running) this.remaining[this.running] = this.left(this.running);
    this.running = null;
    this.since = null;
  }

  /** Start `color`'s clock (stopping the other). */
  start(color) {
    this.stop();
    this.running = color;
    this.since = this.now();
  }

  /** The side that has run out of time, or null. */
  flagged() {
    for (const color of ["w", "b"]) if (this.left(color) <= 0) return color;
    return null;
  }

  toJSON() {
    return { base: this.base, remaining: { w: this.left("w"), b: this.left("b") }, running: this.running, at: this.now() };
  }

  /** A clock saved with toJSON; the running side is charged for the time since it was saved. */
  static fromJSON(data, options = {}) {
    const clock = new Clock(data.base, options);
    clock.remaining = { w: Number(data.remaining.w), b: Number(data.remaining.b) };
    if (data.running === "w" || data.running === "b") {
      clock.running = data.running;
      clock.since = Number(data.at);
    }
    return clock;
  }
}

/** "4:59" from milliseconds; under 10 seconds with tenths ("9.4"), as in fast games. */
export function formatClock(ms) {
  const value = Math.max(0, ms);
  if (value < 10000) return (Math.floor(value / 100) / 10).toFixed(1);
  const seconds = Math.ceil(value / 1000);
  return `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, "0")}`;
}

/** How long the bot "thinks" (ms) with `leftMs` on its clock, `ply` half-moves into the game: quick in the
 *  opening, about a thirtieth of its time later on, hurried when short of time; `random` in [0, 1) spreads it. */
export function thinkTime(leftMs, ply, random = Math.random) {
  const share = ply < 10 ? 80 : 30;                        // opening moves are played from memory
  const spread = 0.35 + 1.3 * random();
  const wanted = (leftMs / share) * spread;
  return Math.round(Math.max(150, Math.min(wanted, leftMs * 0.2, 30000)));
}
