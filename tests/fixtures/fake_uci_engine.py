"""A tiny fake UCI engine used to test the Stockfish checks without needing Stockfish.

It understands just enough of the UCI protocol for ``chess.engine`` to start it,
ask for a move, and shut it down. It always answers ``bestmove e2e4`` unless
started with ``--illegal``, in which case it answers an illegal move.
"""

import sys


def main() -> None:
    best_move = "e2e5" if "--illegal" in sys.argv else "e2e4"
    for line in sys.stdin:
        command = line.strip()
        if command == "uci":
            print("id name FakeFish 1.0")
            print("id author tests")
            print("uciok")
        elif command == "isready":
            print("readyok")
        elif command.startswith("go"):
            print(f"bestmove {best_move}")
        elif command == "quit":
            break
        sys.stdout.flush()


if __name__ == "__main__":
    main()
