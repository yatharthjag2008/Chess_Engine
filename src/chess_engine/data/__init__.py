"""Data pipeline: downloading, streaming and filtering Lichess games, building datasets.

Planned modules:
    pgn_stream       (step 2)  stream .pgn.zst files and filter games
    dataset_builder  (step 4)  games -> training samples, sharding, train/val/test split by game
    game_import      (step 13) import games from pasted PGN or the Lichess / Chess.com APIs
"""
