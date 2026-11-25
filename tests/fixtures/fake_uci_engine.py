"""A tiny fake UCI engine used to test the Stockfish checks without needing Stockfish.

It understands just enough of the UCI protocol for ``chess.engine`` to start it,
ask for a move, and shut it down. It always answers ``bestmove e2e4`` unless
started with ``--illegal``, in which case it answers an illegal move, or with
``--first-legal``, in which case it tracks the ``position`` command and answers
the alphabetically first legal move (UCI notation) of that position.
With ``--score-cp=N`` it reports ``info depth 1 score cp N`` (side to move's view) before each answer.
"""

import sys


def main() -> None:
    best_move = "e2e5" if "--illegal" in sys.argv else "e2e4"
    first_legal = "--first-legal" in sys.argv
    score = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--score-cp=")), None)
    board = None
    if first_legal:
        import chess

        board = chess.Board()
    for line in sys.stdin:
        command = line.strip()
        if command == "uci":
            print("id name FakeFish 1.0")
            print("id author tests")
            print("uciok")
        elif command == "isready":
            print("readyok")
        elif command.startswith("position") and first_legal:
            board = _parse_position(command)
        elif command.startswith("go"):
            if score is not None:
                print(f"info depth 1 score cp {score} nodes 1 pv {best_move}")
            if first_legal:
                best_move = min(move.uci() for move in board.legal_moves)
            print(f"bestmove {best_move}")
        elif command == "quit":
            break
        sys.stdout.flush()


def _parse_position(command: str):
    import chess

    parts = command.split()
    moves = parts.index("moves") if "moves" in parts else len(parts)
    if parts[1] == "startpos":
        board = chess.Board()
    else:
        board = chess.Board(" ".join(parts[2:moves]))
    for uci in parts[moves + 1:]:
        board.push_uci(uci)
    return board


if __name__ == "__main__":
    main()
