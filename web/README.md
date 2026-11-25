# web/

The browser frontend, served by the FastAPI app (`python scripts/serve.py`, then open http://127.0.0.1:8000).
Plain HTML, CSS and JavaScript modules: no build step, no third-party code.

| File | What it is |
|---|---|
| `index.html` | the play page |
| `css/style.css` | styles (colour tokens, dark mode, responsive layout) |
| `js/board.js` | the chessboard: rendering, click and drag moves, promotion choice |
| `js/api.js` | calls to the game API |
| `js/app.js` | the page logic: new game, moves, bot replies, clocks, result, resign/takeback, resume |
| `tests/` | the page's tests (`index.html`: board unit tests; `app.html`: end to end against the API) |

The tests run in headless Chrome / Edge from pytest (`tests/test_web_js.py`). To run them by hand, start the
server and open http://127.0.0.1:8000/tests/index.html or /tests/app.html.

- Step 10: play page (board, rating selector, move list, clocks, game-over handling)
- Step 16: review page (classifications, eval graph, level comparison, summary)
