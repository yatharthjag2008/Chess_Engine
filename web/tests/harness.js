// A tiny test runner for the web UI: register tests with test(), then run() writes the results
// into #results, ending with "RESULT passed=N failed=M" (read back by tests/test_web_js.py).

const tests = [];

export function test(name, fn) { tests.push({ name, fn }); }

export function assert(condition, message) {
  if (!condition) throw new Error(message || "assertion failed");
}

export function equal(actual, expected, message) {
  const a = JSON.stringify(actual), e = JSON.stringify(expected);
  if (a !== e) throw new Error(`${message || "not equal"}: got ${a}, expected ${e}`);
}

export const tick = (ms = 0) => new Promise((resolve) => setTimeout(resolve, ms));

/** Wait until `condition()` is true (polling), or fail after `timeout` ms. */
export async function until(condition, message, timeout = 4000) {
  const start = performance.now();
  while (!condition()) {
    if (performance.now() - start > timeout) throw new Error(`timed out: ${message}`);
    await tick(20);
  }
}

export async function run() {
  const lines = [];
  let passed = 0, failed = 0;
  for (const { name, fn } of tests) {
    try {
      await fn();
      passed += 1;
      lines.push(`ok   ${name}`);
    } catch (err) {
      failed += 1;
      lines.push(`FAIL ${name}: ${err.message}`);
    }
  }
  lines.push(`RESULT passed=${passed} failed=${failed}`);
  document.getElementById("results").textContent = lines.join("\n");
}
