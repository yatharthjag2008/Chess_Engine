"""Evaluation: move-matching accuracy, baselines, calibration, Elo validation, rating estimation.

Step 7 modules:
    positions           64-bit position keys; boards rebuilt from dataset samples
    predictions         one pass of a model over a split -> per-position PredictionTable (.npz)
    move_matching       top-1/3/5 and log-likelihood per rating bucket; breakdowns by phase, clock, legal moves
    bootstrap           confidence intervals that resample whole games
    baselines           most common move (overall / same rating) and random legal move
    stockfish_baseline  Stockfish's best move on a sample of positions (cached)
    cross_rating        model rating x player rating accuracy matrix; fixed-rating baseline
    calibration         reliability of move probabilities and of the value head
    plots               evaluation charts (PNG)
    report              Markdown report
    evaluate            evaluate_model: everything above for one model file
    compare             several models on the same positions, paired confidence intervals

Planned:
    elo_validation     (step 11) check bot strength increases with the rating dial
    rating_estimation  (step 12) estimate a player rating (and range) from games
"""

from chess_engine.eval.evaluate import EvaluationError, EvaluationOutput, evaluate_model
from chess_engine.eval.move_matching import move_matching
from chess_engine.eval.predictions import PredictionTable, predict_split

__all__ = [
    "EvaluationError",
    "EvaluationOutput",
    "PredictionTable",
    "evaluate_model",
    "move_matching",
    "predict_split",
]
