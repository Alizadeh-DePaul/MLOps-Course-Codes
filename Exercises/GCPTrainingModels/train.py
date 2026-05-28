"""Tiny PyTorch trainer for the SE 489 "Using GCP: Training Models" exercise.

The point of this script is not the model — it is to have a small,
dependency-light, real PyTorch training loop that:

  * fits in well under a minute on a CPU custom job (cheap demo),
  * actually exercises PyTorch (so the Agent Platform worker really does load
    torch, run a backward pass, and write a checkpoint), and
  * demonstrates the `/gcs/<bucket>/...` FUSE mount that Agent Platform custom
    jobs auto-attach to the worker — pass `--gcs-checkpoint-dir` to write
    the trained weights straight to a GCS bucket without any GCS client
    library code.

It trains a one-layer linear model on synthetic data drawn from a known
linear relationship (y = 2x + 1 + noise). Loss should drop monotonically.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn, optim


def generate_synthetic_data(
    n_samples: int = 1024,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw n_samples (x, y) pairs from y = 2x + 1 + gaussian noise."""
    rng = torch.Generator().manual_seed(seed)
    x = torch.rand((n_samples, 1), generator=rng) * 10.0  # x in [0, 10)
    noise = torch.randn((n_samples, 1), generator=rng) * 0.5
    y = 2.0 * x + 1.0 + noise
    return x, y


def train_one_epoch(
    model: nn.Module,
    optimizer: optim.Optimizer,
    loss_fn: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    batch_size: int,
) -> float:
    """One pass over the data in mini-batches. Returns mean batch loss."""
    perm = torch.randperm(x.size(0))
    losses: list[float] = []
    for start in range(0, x.size(0), batch_size):
        idx = perm[start : start + batch_size]
        xb, yb = x[idx], y[idx]

        optimizer.zero_grad()
        pred = model(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
    return sum(losses) / len(losses)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tiny PyTorch trainer for the GCP Training Models exercise.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of passes over the synthetic dataset.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Mini-batch size.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        help="SGD learning rate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for reproducibility.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("model.pt"),
        help=(
            "Where to save the trained state_dict. Combined with "
            "--gcs-checkpoint-dir if that flag is also set."
        ),
    )
    parser.add_argument(
        "--gcs-checkpoint-dir",
        type=Path,
        default=None,
        help=(
            "Optional GCS-FUSE directory to write the checkpoint to "
            "(e.g. /gcs/my-bucket/checkpoints). Only meaningful when "
            "running inside an Agent Platform custom job; from a local "
            "machine the path will not exist."
        ),
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available:  {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device:     {torch.cuda.get_device_name(0)}")

    # Synthetic dataset: y = 2x + 1 + noise.
    x, y = generate_synthetic_data(seed=args.seed)
    print(f"Dataset:         {x.shape[0]} samples, x in [0, 10)")

    # One-layer linear model. Expected to converge to weight ~2, bias ~1.
    model = nn.Linear(in_features=1, out_features=1)
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    print(f"Starting training: epochs={args.epochs} lr={args.lr} bs={args.batch_size}")
    for epoch in range(1, args.epochs + 1):
        epoch_loss = train_one_epoch(
            model, optimizer, loss_fn, x, y, args.batch_size,
        )
        print(f"  epoch {epoch:>3d}  loss={epoch_loss:.4f}")

    # Report final learned parameters as a sanity check.
    final_weight = model.weight.item()
    final_bias = model.bias.item()
    print(
        f"Learned: y = {final_weight:.4f} * x + {final_bias:.4f}  "
        f"(target was y = 2.0 * x + 1.0)",
    )

    # Resolve checkpoint location: GCS mount overrides local path if given.
    if args.gcs_checkpoint_dir is not None:
        ckpt_path = args.gcs_checkpoint_dir / args.checkpoint.name
    else:
        ckpt_path = args.checkpoint

    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_path)
    print(f"Checkpoint saved to {ckpt_path}")
    print("Training done.")


if __name__ == "__main__":
    main()
