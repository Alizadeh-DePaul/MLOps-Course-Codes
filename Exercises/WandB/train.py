"""Minimal MNIST training loop with Weights & Biases scalar logging.

Exercise steps:
    1. Run `wandb login` once (it stores your API key in ~/.netrc).
    2. Run `python train.py`.
    3. Open the run URL printed in the terminal and watch the `loss` chart.

This file deliberately keeps the model tiny so each epoch runs in seconds even on CPU.
The point of the exercise is the W&B integration, not the ML.
"""
from __future__ import annotations

import os
from pathlib import Path

# Configure WANDB_DIR BEFORE `import wandb` so wandb's module-init code never
# creates a `./wandb/` cache in this script's directory. If `./wandb/` exists
# in CWD when `wandb agent` spawns a Python subprocess, Python imports that
# local folder as a PEP 420 namespace package instead of the installed wandb
# library, and the subprocess dies with
# `AttributeError: module 'wandb' has no attribute 'init'`.
_WANDB_RUNS = Path.home() / ".wandb-runs"
_WANDB_RUNS.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("WANDB_DIR", str(_WANDB_RUNS))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from torchvision import datasets, transforms  # noqa: E402

import wandb  # noqa: E402  -- must come AFTER WANDB_DIR is set in os.environ


def build_model(dropout: float = 0.2) -> nn.Module:
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(128, 10),
    )


def main() -> None:
    # 1. Start a new W&B run. `project` groups related runs together in the UI;
    #    `name` is human-readable for this specific run; omit `entity` to use your
    #    personal username, or pass a team slug for a shared workspace.
    wandb.init(
        project="Week7-project",
        name="basic-mnist-run",
        config={
            "batch_size": 64,
            "learning_rate": 1e-2,
            "epochs": 2,
            "dropout": 0.2,
        },
    )  # WANDB_DIR is already in os.environ, so wandb's runs land in ~/.wandb-runs

    # 2. Read hyperparameters from wandb.config — this is what lets a sweep
    #    inject different values without you editing the file.
    cfg = wandb.config

    # 3. Data
    train_loader = DataLoader(
        datasets.MNIST("./data", train=True, download=True, transform=transforms.ToTensor()),
        batch_size=cfg.batch_size,
        shuffle=True,
    )

    # 4. Model, loss, optimizer
    model = build_model(dropout=cfg.dropout)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.learning_rate)

    # 5. Training loop with W&B logging every 100 mini-batches
    global_step = 0
    for epoch in range(cfg.epochs):
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            global_step += 1

            # Log scalar metrics. `step` keeps the x-axis monotonic across epochs.
            if (i + 1) % 100 == 0:
                wandb.log(
                    {
                        "epoch": epoch + 1,
                        "batch": i + 1,
                        "loss": running_loss / 100,
                    },
                    step=global_step,
                )
                running_loss = 0.0

    # 6. Close the run cleanly. Required in notebooks; harmless in scripts.
    wandb.finish()


if __name__ == "__main__":
    main()
