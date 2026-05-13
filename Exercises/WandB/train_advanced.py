"""MNIST training loop demonstrating non-scalar W&B logging.

Beyond `wandb.log({"loss": ...})`, this script shows four richer media types:

    * `wandb.Image`             - log raw image tensors / arrays / PIL images
    * `wandb.Histogram`         - log distributions (e.g. weight tensors)
    * Matplotlib figures        - wrap a `fig` in `wandb.Image(fig)` to log it
    * `wandb.plot.confusion_matrix` - W&B's first-class confusion-matrix plot

After running this, open the Workspace UI; the Media panel will show the
images and the confusion matrix, and the Custom Charts panel will hold the
weight histograms and matplotlib figure.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import wandb


def build_model(dropout: float = 0.2) -> nn.Module:
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(128, 10),
    )


def log_weight_histograms(model: nn.Module, step: int) -> None:
    """Log a histogram per `weight` tensor in the model."""
    for name, param in model.named_parameters():
        if "weight" in name:
            wandb.log(
                {f"hist/{name}": wandb.Histogram(param.detach().cpu().numpy())},
                step=step,
            )


def log_sample_images(images: torch.Tensor, step: int, n: int = 4) -> None:
    """Log the first n images of a batch as a W&B media panel."""
    # images shape: (batch, 1, 28, 28). Squeeze to (batch, 28, 28) for display.
    sample = images[:n].squeeze(1).cpu().numpy()
    wandb.log(
        {"sample_images": [wandb.Image(img, caption=f"img {i}") for i, img in enumerate(sample)]},
        step=step,
    )


def log_loss_curve(losses: list[float], step: int) -> None:
    """Log a matplotlib figure of the running loss."""
    fig, ax = plt.subplots()
    ax.plot(losses)
    ax.set_xlabel("logging step")
    ax.set_ylabel("loss")
    ax.set_title("Running loss")
    wandb.log({"loss_curve_mpl": wandb.Image(fig)}, step=step)
    plt.close(fig)  # release the figure or matplotlib leaks memory across logs


def main() -> None:
    wandb.init(
        project="Week7-project",
        name="advanced-mnist-run",
        config={
            "batch_size": 64,
            "learning_rate": 1e-2,
            "epochs": 1,  # one epoch is enough to see all the media types
            "dropout": 0.2,
        },
    )
    cfg = wandb.config

    train_loader = DataLoader(
        datasets.MNIST("./data", train=True, download=True, transform=transforms.ToTensor()),
        batch_size=cfg.batch_size,
        shuffle=True,
    )

    model = build_model(dropout=cfg.dropout)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.learning_rate)

    losses: list[float] = []
    last_outputs: torch.Tensor | None = None
    last_labels: torch.Tensor | None = None
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
            last_outputs, last_labels = outputs, labels  # remember for end-of-run logging

            if (i + 1) % 100 == 0:
                avg = running_loss / 100
                losses.append(avg)
                wandb.log({"loss": avg, "epoch": epoch + 1}, step=global_step)
                running_loss = 0.0

                # Every 500 steps, log a few sample images so the Media panel populates.
                if (i + 1) % 500 == 0:
                    log_sample_images(images, step=global_step)

    # End-of-run logging — these use values from the LAST mini-batch, which is
    # the well-defined point where the model has finished training.
    log_weight_histograms(model, step=global_step)
    log_loss_curve(losses, step=global_step)

    if last_outputs is not None and last_labels is not None:
        _, preds = torch.max(last_outputs, dim=1)
        wandb.log(
            {
                "final_batch_accuracy": (preds == last_labels).float().mean().item(),
                "confusion_matrix": wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=last_labels.cpu().numpy(),
                    preds=preds.cpu().numpy(),
                    class_names=[str(i) for i in range(10)],
                ),
            },
            step=global_step,
        )

    wandb.finish()


if __name__ == "__main__":
    # Suppress matplotlib's harmless "FigureCanvasAgg is non-interactive" message
    # when running headlessly inside a container.
    np.seterr(all="ignore")
    main()
