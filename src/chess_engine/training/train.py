"""Config-driven training loop.

    trainer = Trainer(cfg, run_dir="checkpoints/my_run")
    result = trainer.train()
    result.final_model        # checkpoints/my_run/final_model.pt (load with chess_engine.model.load_model)

What one optimizer step does:

1. a batch of positions (decoded ahead on a background thread, see :mod:`.loader`),
2. forward pass under the configured precision (fp32 on CPU, bf16/fp16 on GPU),
3. loss = weighted legal-move cross-entropy + value loss (:mod:`.losses`),
4. backward (scaled for fp16), gradient-norm clipping, AdamW step, learning-rate schedule step.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from chess_engine.config import Config, TrainingConfig
from chess_engine.data.dataset import PositionDataset
from chess_engine.data.dataset_builder import dataset_output_dir
from chess_engine.model.inference import save_model
from chess_engine.model.network import build_model, count_parameters
from chess_engine.training.balancing import bucket_weights, format_balance
from chess_engine.training.checkpoint import TrainingState
from chess_engine.training.device import configure_threads, resolve_device, resolve_precision, seed_everything
from chess_engine.training.loader import BatchLoader, TrainBatch
from chess_engine.training.losses import compute_loss
from chess_engine.training.metrics import RunningMeans
from chess_engine.training.run_log import RunDirectory
from chess_engine.training.schedule import build_scheduler

Logger = Callable[[str], None]


class TrainingError(RuntimeError):
    """Training cannot start or cannot go on (bad dataset, diverged loss, ...)."""


def build_optimizer(model: nn.Module, t: TrainingConfig) -> torch.optim.Optimizer:
    """AdamW (or SGD with Nesterov momentum) with weight decay on conv/linear weights only.

    Biases, batch-norm parameters and the rating embeddings are not decayed:
    shrinking them towards zero only fights the normalization (BN) or erases what
    a rarely seen rating bucket has learned (embeddings).
    """
    decay, no_decay = [], []
    for module in model.modules():
        for name, param in module.named_parameters(recurse=False):
            if not param.requires_grad:
                continue
            is_weight = name == "weight" and isinstance(module, (nn.Conv2d, nn.Linear))
            (decay if is_weight else no_decay).append(param)
    groups = [{"params": decay, "weight_decay": t.weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
    if t.optimizer == "adamw":
        return torch.optim.AdamW(groups, lr=t.learning_rate)
    return torch.optim.SGD(groups, lr=t.learning_rate, momentum=0.9, nesterov=True)


def train_bucket_counts(dataset: PositionDataset, num_buckets: int) -> np.ndarray:
    """Positions per rating bucket in the training data actually used."""
    if dataset.in_memory:
        return np.bincount(dataset.samples["bucket"].astype(np.int64), minlength=num_buckets)
    # Streaming: the manifest's split totals, scaled to the shards in use.
    counts = np.asarray(dataset.manifest["splits"][dataset.split]["positions_per_bucket"], dtype=np.float64)
    total = dataset.manifest["splits"][dataset.split]["positions"]
    return np.rint(counts * len(dataset) / max(total, 1)).astype(np.int64)


@dataclass(frozen=True)
class TrainResult:
    """What a :meth:`Trainer.train` call did."""

    state: TrainingState
    stop_reason: str              # "finished" (all epochs) or "max_steps"
    final_model: Path
    last_train: dict[str, float]  # the last logged training means


class Trainer:
    """Trains the configured model on a built dataset, logging into ``run_dir``.

    Args:
        cfg: Project config (``training``, ``model``, ``ratings``).
        run_dir: Run directory (see :mod:`.run_log`); created if it does not exist.
        dataset_dir: Built dataset (step 4); default: the config's dataset directory.
        log: Where progress lines go (default ``print``).
    """

    def __init__(
        self,
        cfg: Config,
        run_dir: str | Path,
        dataset_dir: str | Path | None = None,
        log: Logger = print,
    ) -> None:
        t = cfg.training
        self.cfg = cfg
        self.log = log
        self.run = RunDirectory(run_dir)
        self.device = resolve_device(t.device)
        self.threads = configure_threads(t.threads) if self.device.type == "cpu" else torch.get_num_threads()
        seed_everything(t.seed)
        self.precision = resolve_precision(t.mixed_precision, self.device)

        self.dataset_dir = Path(dataset_dir) if dataset_dir is not None else dataset_output_dir(cfg)
        if not (self.dataset_dir / "manifest.json").is_file():
            raise TrainingError(f"no dataset at {self.dataset_dir} (build it with scripts/build_dataset.py)")
        self.train_data = PositionDataset(self.dataset_dir, "train", in_memory=t.in_memory,
                                          max_shards=t.max_train_shards)
        edges = self.train_data.manifest["ratings"]["bucket_edges"]
        if edges != cfg.ratings.bucket_edges:
            raise TrainingError(f"dataset was built with bucket edges {edges}, config has {cfg.ratings.bucket_edges}")
        if len(self.train_data) < t.batch_size:
            raise TrainingError(f"training split has {len(self.train_data)} positions, fewer than one batch "
                                f"({t.batch_size})")

        num_buckets = cfg.ratings.num_buckets
        self.bucket_counts = train_bucket_counts(self.train_data, num_buckets)
        self.bucket_weights = (bucket_weights(self.bucket_counts, t.balance_power)
                               if t.bucket_balancing != "none" else np.ones(num_buckets))
        self.loader = BatchLoader(
            self.train_data, cfg.ratings, t.batch_size, self.device, t.prefetch_batches,
            loss_weights=self.bucket_weights if t.bucket_balancing == "weight" else None,
            resample_weights=self.bucket_weights if t.bucket_balancing == "sample" else None,
        )

        self.model = build_model(cfg).to(self.device)
        self.optimizer = build_optimizer(self.model, t)
        self.steps_per_epoch = self.loader.batches_per_epoch(0, t.seed)
        planned = t.epochs * self.steps_per_epoch
        self.total_steps = min(planned, t.max_steps) if t.max_steps is not None else planned
        self.scheduler = build_scheduler(self.optimizer, self.total_steps, t.lr_schedule, t.warmup_steps,
                                         t.min_lr_fraction)
        self.scaler = torch.amp.GradScaler(self.device.type, enabled=self.precision.needs_grad_scaler)
        self.state = TrainingState()
        self._running = RunningMeans()
        self._last_train: dict[str, float] = {}

    # ------------------------------------------------------------------ one step

    def train_step(self, batch: TrainBatch) -> None:
        """Forward, backward and optimizer step on one batch (statistics go to the running means)."""
        t = self.cfg.training
        self.model.train()
        with self.precision.autocast():
            out = self.model(batch.inputs)
        loss = compute_loss(out, batch, t.value_loss_weight, t.label_smoothing)
        self.optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(loss.total).backward()
        if t.grad_clip_norm is not None:
            self.scaler.unscale_(self.optimizer)
            grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), t.grad_clip_norm)
            self._running.add("grad_norm", grad_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()

        running = self._running
        running.add("loss", loss.total)
        running.add("policy_loss", loss.policy)
        if loss.value is not None:
            running.add("value_loss", loss.value)
        hits = (loss.log_probs.argmax(-1) == batch.targets.move).float().mean()
        running.add("top1", hits)

    # ------------------------------------------------------------------ logging

    def _log_train(self, positions_per_second: float) -> dict[str, float]:
        means = self._running.means()
        self._running.reset()
        if not math.isfinite(means["loss"]):
            raise TrainingError(f"training loss is {means['loss']} at step {self.state.step} (diverged); "
                                "try a lower learning rate or mixed_precision: off")
        lr = self.optimizer.param_groups[0]["lr"]
        remaining = max(0, self.total_steps - self.state.step) * self.cfg.training.batch_size
        eta_minutes = remaining / positions_per_second / 60 if positions_per_second > 0 else float("nan")
        record = {**means, "lr": lr, "epoch": self.state.epoch, "positions": self.state.positions_seen,
                  "positions_per_second": positions_per_second}
        self.run.metrics.write("train", self.state.step, **record)
        self.log(f"step {self.state.step:>7}/{self.total_steps}  epoch {self.state.epoch + 1}  "
                 f"loss {means['loss']:.4f}  top1 {means['top1']:.1%}  lr {lr:.2e}  "
                 f"{positions_per_second:,.0f} pos/s  ETA {eta_minutes:.0f} min")
        self._last_train = record
        return record

    def describe(self) -> str:
        """Multi-line summary of the setup (device, data, model, schedule, balancing)."""
        t = self.cfg.training
        params = count_parameters(self.model)["total"]
        lines = [
            f"run:       {self.run.path}",
            f"device:    {self.device} ({self.precision.name}"
            + (f", {self.threads} CPU threads)" if self.device.type == "cpu" else ")"),
            f"dataset:   {self.dataset_dir} ({len(self.train_data):,} training positions)",
            f"model:     {self.cfg.model.channels} channels x {self.cfg.model.residual_blocks} blocks, "
            f"{params:,} parameters",
            f"schedule:  {self.total_steps:,} steps of {t.batch_size} ({self.steps_per_epoch:,} per epoch), "
            f"{t.optimizer} lr {t.learning_rate:g} {t.lr_schedule}, warmup {t.warmup_steps}",
            f"balancing: {t.bucket_balancing} (power {t.balance_power})",
        ]
        if t.bucket_balancing != "none":
            lines.append(format_balance(self.cfg.ratings.bucket_labels(), self.bucket_counts, self.bucket_weights))
        return "\n".join(lines)

    # ------------------------------------------------------------------ the loop

    def _start_run(self) -> None:
        if not self.run.exists():
            self.run.create(self.cfg)
        self.run.write_info(
            dataset=str(self.dataset_dir), device=str(self.device), precision=self.precision.name,
            parameters=count_parameters(self.model)["total"], train_positions=len(self.train_data),
            steps_per_epoch=self.steps_per_epoch, total_steps=self.total_steps, status="running",
        )

    def train(self) -> TrainResult:
        """Train until every epoch is done or ``max_steps`` is reached; save the final model."""
        t = self.cfg.training
        self._start_run()
        self.log(self.describe())
        stop: str | None = None
        window_start, window_positions = time.perf_counter(), 0
        while self.state.epoch < t.epochs and stop is None:
            for batch in self.loader.epoch(self.state.epoch, t.seed, self.state.batch_in_epoch):
                self.train_step(batch)
                self.state.step += 1
                self.state.batch_in_epoch += 1
                self.state.positions_seen += len(batch)
                window_positions += len(batch)
                if self.state.step % t.log_every_steps == 0:
                    now = time.perf_counter()
                    self._log_train(window_positions / max(now - window_start, 1e-9))
                    window_start, window_positions = now, 0
                if t.max_steps is not None and self.state.step >= t.max_steps:
                    stop = "max_steps"
                    break
            else:
                self.state.epoch += 1
                self.state.batch_in_epoch = 0
        stop_reason = stop or "finished"
        if self._running:
            self._log_train(window_positions / max(time.perf_counter() - window_start, 1e-9))
        final = save_model(self.model, self.run.final_model_path, self.cfg.ratings,
                           extra={"step": self.state.step, "run": self.run.name})
        self.run.write_info(status=stop_reason, steps_done=self.state.step)
        self.log(f"done ({stop_reason}) after {self.state.step} steps; model saved to {final}")
        return TrainResult(self.state, stop_reason, final, self._last_train)


def run_training(cfg: Config, run_dir: str | Path, **kwargs: Any) -> TrainResult:
    """Build a :class:`Trainer` and train."""
    return Trainer(cfg, run_dir, **kwargs).train()
