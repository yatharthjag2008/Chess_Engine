// The honest rating dial (data/dial.json, from scripts/build_dial.py and chess_engine/eval/dial.py): a label is the
// bot's measured strength, anchored at 1500, and the bot plays at the model input that has that strength.
//
//   const dial = makeDial(await loadDial("data/"));
//   dial.min, dial.max                  // the strengths the bot can actually play (e.g. 1100-1750)
//   dial.inputFor(1600)                 // the rating the model is told to imitate for that strength
//   dial.labelFor(1400)                 // what a model input really plays like

/** numpy.interp on increasing xs, flat beyond the ends. */
function interp(x, xs, ys) {
  if (x <= xs[0]) return ys[0];
  const last = xs.length - 1;
  if (x >= xs[last]) return ys[last];
  let i = 1;
  while (xs[i] < x) i++;
  return ys[i - 1] + ((x - xs[i - 1]) / (xs[i] - xs[i - 1])) * (ys[i] - ys[i - 1]);
}

/** The dial from dial.json's contents; null gives the plain dial (800-2400, a label is the model's input). */
export function makeDial(data) {
  if (!data || !Array.isArray(data.points) || data.points.length < 2) {
    return { calibrated: false, min: 800, max: 2400, step: 50, anchor: 1500, points: [],
             inputFor: (label) => Math.round(label), labelFor: (input) => input };
  }
  const inputs = data.points.map((p) => p.input);
  const labels = data.points.map((p) => p.label);
  return {
    calibrated: true, min: data.min, max: data.max, step: data.step || 50, anchor: data.anchor, points: data.points,
    labelFor: (input) => interp(input, inputs, labels),
    /** The lowest input that plays at `label` (as DialCalibration.input_for). */
    inputFor(label) {
      if (label <= labels[0]) return inputs[0];
      if (label >= labels[labels.length - 1]) return inputs[inputs.length - 1];
      let i = 0;
      while (labels[i] < label) i++;
      if (labels[i] === label) return inputs[i];
      const t = (label - labels[i - 1]) / (labels[i] - labels[i - 1]);
      return Math.round(inputs[i - 1] + t * (inputs[i] - inputs[i - 1]));
    },
  };
}

/** data/dial.json, or null if the site has none. */
export async function loadDial(base = "data/") {
  try {
    const response = await fetch(base + "dial.json");
    return response.ok ? await response.json() : null;
  } catch {
    return null;
  }
}
