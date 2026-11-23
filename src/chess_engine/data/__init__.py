"""Data pipeline: downloading, streaming and filtering Lichess games, building datasets.

Modules:
    pgn_stream       (step 2)  split PGN streams into games, parse movetext, filter games
    sources          (step 2)  open local/remote .pgn(.zst) as text streams; resumable downloads
    game_store       (step 2)  save filtered games to Parquet; the stream -> filter -> save pipeline
    dataset_builder  (step 4)  games -> training samples, sharding, train/val/test split by game
    game_import      (step 13) import games from pasted PGN or the Lichess / Chess.com APIs
"""
