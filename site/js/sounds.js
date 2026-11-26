// Sound effects, synthesized with the Web Audio API: no audio files (and no licences to track).
//
//   const sounds = new Sounds();
//   sounds.play(soundFor(move, { check: chess.inCheck() }));   // "move", "capture", "castle", ...
//
// Every sound is a few milliseconds of filtered noise (the wooden click of a piece) and/or short
// sine tones. Browsers keep audio silent until the page has been clicked: call unlock() from the
// first click or key press (site.js does), and nothing breaks before that.

/** Which sound a chess.js move makes (`check`: the move gives check). */
export function soundFor(move, { check = false } = {}) {
  if (check) return "check";
  if (move.promotion) return "promote";
  if (move.flags.includes("k") || move.flags.includes("q")) return "castle";
  if (move.captured) return "capture";
  return "move";
}

/** A burst of noise through a band-pass filter: the knock of a piece on the board. */
function knock(ctx, at, { freq = 1800, q = 1.2, length = 0.035, gain = 0.6 } = {}) {
  const frames = Math.ceil(ctx.sampleRate * length);
  const buffer = ctx.createBuffer(1, frames, ctx.sampleRate);
  const data = buffer.getChannelData(0);
  for (let i = 0; i < frames; i++) data[i] = (Math.random() * 2 - 1) * (1 - i / frames) ** 2;
  const source = ctx.createBufferSource();
  source.buffer = buffer;
  const filter = ctx.createBiquadFilter();
  filter.type = "bandpass";
  filter.frequency.value = freq;
  filter.Q.value = q;
  const amp = ctx.createGain();
  amp.gain.value = gain;
  source.connect(filter).connect(amp).connect(ctx.destination);
  source.start(at);
}

/** A sine tone with a quick attack and an exponential fade. */
function tone(ctx, at, { freq = 440, length = 0.12, gain = 0.2, type = "sine" } = {}) {
  const osc = ctx.createOscillator();
  osc.type = type;
  osc.frequency.value = freq;
  const amp = ctx.createGain();
  amp.gain.setValueAtTime(0.0001, at);
  amp.gain.exponentialRampToValueAtTime(gain, at + 0.008);
  amp.gain.exponentialRampToValueAtTime(0.0001, at + length);
  osc.connect(amp).connect(ctx.destination);
  osc.start(at);
  osc.stop(at + length + 0.02);
}

const notes = (ctx, at, freqs, step, volume, length = 0.16) =>
  freqs.forEach((freq, i) => tone(ctx, at + i * step, { freq, length, gain: 0.18 * volume }));

export const RECIPES = {
  move: (ctx, at, v) => { knock(ctx, at, { gain: 0.7 * v }); tone(ctx, at, { freq: 190, length: 0.05, gain: 0.12 * v }); },
  capture: (ctx, at, v) => {
    knock(ctx, at, { freq: 1100, length: 0.06, gain: 0.9 * v });
    knock(ctx, at + 0.03, { freq: 2400, length: 0.03, gain: 0.5 * v });
    tone(ctx, at, { freq: 130, length: 0.09, gain: 0.18 * v });
  },
  castle: (ctx, at, v) => { RECIPES.move(ctx, at, v); RECIPES.move(ctx, at + 0.09, v); },
  check: (ctx, at, v) => { RECIPES.move(ctx, at, v); notes(ctx, at + 0.02, [880, 1175], 0.09, v, 0.12); },
  promote: (ctx, at, v) => { RECIPES.move(ctx, at, v); notes(ctx, at + 0.02, [660, 880, 1100], 0.07, v); },
  start: (ctx, at, v) => notes(ctx, at, [523, 784], 0.1, v, 0.22),
  win: (ctx, at, v) => notes(ctx, at, [523, 659, 784, 1047], 0.1, v, 0.3),
  lose: (ctx, at, v) => notes(ctx, at, [392, 330, 262], 0.14, v, 0.32),
  draw: (ctx, at, v) => notes(ctx, at, [440, 440], 0.16, v, 0.22),
};

function defaultContext() {
  const Context = window.AudioContext || window.webkitAudioContext;
  return Context ? new Context() : null;
}

export class Sounds {
  constructor({ enabled = true, volume = 0.7, makeContext = defaultContext } = {}) {
    this.enabled = enabled;
    this.volume = volume;
    this.makeContext = makeContext;
    this.ctx = undefined;     // created on first use
    this.log = [];            // the names played, newest last (for tests and debugging)
  }

  get context() {
    if (this.ctx === undefined) {
      try { this.ctx = this.makeContext(); } catch { this.ctx = null; }
    }
    return this.ctx;
  }

  /** Browsers start audio suspended until the page is clicked: call this from a click or key press. */
  unlock() {
    const ctx = this.context;
    if (ctx && ctx.state === "suspended" && ctx.resume) ctx.resume().catch(() => {});
    return Boolean(ctx && ctx.state !== "closed");
  }

  /** Play a sound by name; returns whether it was played (false when muted or unknown). */
  play(name) {
    if (!this.enabled || !RECIPES[name]) return false;
    this.log.push(name);
    const ctx = this.context;
    if (!ctx) return true;
    try {
      RECIPES[name](ctx, ctx.currentTime + 0.005, this.volume);
    } catch {
      // A sound must never break the game.
    }
    return true;
  }
}
