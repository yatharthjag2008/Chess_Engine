// A background worker for rating estimates: it loads the model once, then scores the moves it is sent at every
// rating bucket (rating.js's moveBucketLL), reporting each move as it is done. See pool.js.

import { loadModel } from "./engine.js";
import { moveBucketLL } from "./rating.js";

let model = null;

self.onmessage = async (event) => {
  const message = event.data;
  try {
    if (message.type === "init") {
      model = await loadModel(message.modelBase);
      self.postMessage({ type: "ready" });
    } else if (message.type === "score") {
      const rows = [];
      for (const move of message.moves) {
        rows.push(moveBucketLL(model, move, message.base, message.increment));
        self.postMessage({ type: "move", job: message.job });
      }
      self.postMessage({ type: "done", job: message.job, rows });
    }
  } catch (err) {
    self.postMessage({ type: "error", job: message.job, message: String(err && err.message ? err.message : err) });
  }
};
