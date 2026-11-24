"""Rating-conditioned move-prediction network (step 5).

Modules:
    features    dataset batches -> input tensors (board, rating buckets, clock features) and targets
    layers      condition encoder (rating/opponent/clock -> vector) and conditioned residual blocks
    policy_map  1858-move vocabulary <-> 73 move-type planes per square (convolutional policy head)
    heads       policy heads (conv / dense) and the win/draw/loss value head
    network     RatingConditionedNet, build_model, count_parameters
    masking     illegal-move masking, legal-move probabilities, masked cross-entropy
    inference   MovePredictor (board + rating -> move probabilities), save_model / load_model
"""

from chess_engine.model.features import ModelInputs, Targets, inputs_from_batch, targets_from_batch
from chess_engine.model.inference import LoadedModel, MovePredictor, Prediction, load_model, save_model
from chess_engine.model.masking import legal_log_probs, legal_probs, masked_logits, policy_loss
from chess_engine.model.network import ModelOutput, RatingConditionedNet, build_model, count_parameters

__all__ = [
    "LoadedModel",
    "ModelInputs",
    "ModelOutput",
    "MovePredictor",
    "Prediction",
    "RatingConditionedNet",
    "Targets",
    "build_model",
    "count_parameters",
    "inputs_from_batch",
    "legal_log_probs",
    "legal_probs",
    "load_model",
    "masked_logits",
    "policy_loss",
    "save_model",
    "targets_from_batch",
]
