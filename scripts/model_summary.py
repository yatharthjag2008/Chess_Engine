"""Describe the configured model: parameters per part, speed on this machine, estimated epoch time (step 5).

Builds the network from the config (untrained), times forward passes and
training steps on random inputs, and, if the step-4 dataset exists, estimates
how long one pass over its training split would take here.

Usage:
    python scripts/model_summary.py
    python scripts/model_summary.py --batch-size 512 --threads 4
    python scripts/model_summary.py --config configs/my_model.yaml
"""

from __future__ import annotations

import argparse
import sys
import time

import chess
import torch

from chess_engine.config import ConfigError, load_config
from chess_engine.data.dataset_builder import dataset_output_dir, read_manifest
from chess_engine.encoding.moves import NUM_MOVES
from chess_engine.model.features import NUM_CONTEXT_FEATURES, ModelInputs
from chess_engine.model.inference import MovePredictor
from chess_engine.model.masking import policy_loss
from chess_engine.model.network import build_model, count_parameters


def _random_inputs(batch: int, num_buckets: int) -> tuple[ModelInputs, torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(0)
    inputs = ModelInputs(
        planes=(torch.rand(batch, 20, 8, 8, generator=g) > 0.8).float(),
        bucket=torch.randint(0, num_buckets, (batch,), generator=g),
        opp_bucket=torch.randint(0, num_buckets, (batch,), generator=g),
        context=torch.rand(batch, NUM_CONTEXT_FEATURES, generator=g),
    )
    mask = torch.zeros(batch, NUM_MOVES, dtype=torch.bool)
    mask[:, :30] = True
    return inputs, mask, torch.randint(0, 30, (batch,), generator=g)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None, help="config file merged on top of configs/default.yaml")
    parser.add_argument("--batch-size", type=int, default=None, help="default: training.batch_size")
    parser.add_argument("--steps", type=int, default=5, help="timed training steps")
    parser.add_argument("--threads", type=int, default=None, help="torch CPU threads (default: torch's choice)")
    args = parser.parse_args(argv)

    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        print(exc, file=sys.stderr)
        return 1
    if args.threads:
        torch.set_num_threads(args.threads)
    batch = args.batch_size or cfg.training.batch_size
    torch.manual_seed(0)
    model = build_model(cfg)
    m = cfg.model

    print(f"Model: {m.architecture}, {m.channels} channels x {m.residual_blocks} blocks, conditioning={m.conditioning}, "
          f"policy head={m.policy_head}, value head={m.value_head}")
    print(f"  conditioned on: rating bucket ({cfg.ratings.num_buckets} buckets)"
          + (", opponent bucket" if m.condition_on_opponent else "") + (", clocks" if m.use_clock else ""))
    for name, count in count_parameters(model).items():
        print(f"  {name:<10} {count:>10,} parameters")

    inputs, mask, target = _random_inputs(batch, cfg.ratings.num_buckets)
    model.eval()
    with torch.no_grad():
        model(inputs)
        start = time.perf_counter()
        for _ in range(args.steps):
            model(inputs)
        forward = (time.perf_counter() - start) / args.steps
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.learning_rate)
    for step in range(args.steps + 1):
        if step == 1:
            start = time.perf_counter()
        loss = policy_loss(model(inputs).policy, mask, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    train = (time.perf_counter() - start) / args.steps

    print(f"\nSpeed on this machine (CPU, {torch.get_num_threads()} threads, batch {batch}):")
    print(f"  forward only   {forward * 1000:8.1f} ms/batch  {batch / forward:>9,.0f} positions/s")
    print(f"  training step  {train * 1000:8.1f} ms/batch  {batch / train:>9,.0f} positions/s")
    try:
        positions = read_manifest(dataset_output_dir(cfg))["splits"]["train"]["positions"]
        hours = positions / (batch / train) / 3600
        print(f"  one epoch over the {positions:,} training positions: ~{hours:.1f} h here")
    except FileNotFoundError:
        print("  (build the dataset with scripts/build_dataset.py to see an epoch-time estimate)")

    prediction = MovePredictor(model, cfg.ratings).predict(chess.Board(), rating=1500)
    top = ", ".join(f"{p.san} {p.prob:.1%}" for p in prediction.moves[:3])
    print(f"\nUntrained model on the start position at 1500: {top} (about 1/20 each until trained)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
