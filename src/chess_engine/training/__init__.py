"""Training pipeline (step 6).

Modules:
    device      device choice (cuda > mps > cpu), seeding, CPU threads, mixed precision
    schedule    learning-rate warmup + constant / cosine / linear decay
    balancing   rating-bucket weights so low-data buckets are not under-learned
    order       deterministic, resumable epoch order (shuffle or resample by bucket)
    loader      dataset split -> TrainBatch tensors, decoded ahead on a background thread
    losses      weighted legal-move cross-entropy (+ label smoothing) + value loss
    metrics     running means, top-k move matching, per-bucket metrics
    validation  validation pass on fixed positions
    checkpoint  full training state for resuming; checkpoint rotation
    run_log     run directory layout and the JSON-lines metrics log
    train       Trainer: the training loop (validation, best model, checkpoints, resume, stopping)
"""

from chess_engine.training.checkpoint import TrainingState
from chess_engine.training.run_log import MetricsLog, RunDirectory
from chess_engine.training.train import Trainer, TrainingError, TrainResult, run_training
from chess_engine.training.validation import validate

__all__ = [
    "MetricsLog",
    "RunDirectory",
    "TrainResult",
    "Trainer",
    "TrainingError",
    "TrainingState",
    "run_training",
    "validate",
]
