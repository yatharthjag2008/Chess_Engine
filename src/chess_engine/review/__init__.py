"""Game review and coaching.

Step 14 modules:
    scoring        centipawns -> win %, move accuracy, game accuracy (Lichess formulas)
    classify       best / good / inaccuracy / mistake / blunder from the win % lost
    engine         Stockfish position analysis (multi-PV lines, cache)
    analyzer       review a game: every move judged, per-side summary, key moments, graph, JSON
    annotate       annotated PGN ([%eval], ?! ? ??, better moves)
    run            run_review: one call with the configured Stockfish and a disk cache

Step 15 modules:
    level_compare  the likeliest moves at your rating and +200 / +400, valued by Stockfish; the moments that matter
    features       one-move facts for explanations (captures, mate, hanging pieces)
    coach          template explanations of mistakes and level differences; the coach report
"""
