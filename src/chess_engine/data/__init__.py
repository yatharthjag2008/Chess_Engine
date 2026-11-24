"""Data pipeline: downloading, streaming and filtering Lichess games, building datasets.

Modules:
    pgn_stream       (step 2)  split PGN streams into games, parse movetext, filter games
    sources          (step 2)  open local/remote .pgn(.zst) as text streams; resumable downloads
    game_store       (step 2)  save filtered games to Parquet; the stream -> filter -> save pipeline
    splits           (step 4)  deterministic train/val/test assignment by game id
    samples          (step 4)  replay a game into per-position samples (board, move, rating, clocks)
    shards           (step 4)  save/load samples as shuffled .npz shard files
    dataset_builder  (step 4)  games Parquet -> shards per split + games index + manifest (parallel)
    dataset          (step 4)  read a built split as batches of planes, legal masks and targets
    game_import      (step 13) import games from pasted PGN or the Lichess / Chess.com APIs
"""
