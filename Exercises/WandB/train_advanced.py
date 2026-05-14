"""MNIST training loop demonstrating non-scalar W&B logging.

Beyond `wandb.log({"loss": ...})`, this script shows four richer media types:

    * `wandb.Image`             - log raw image tensors / arrays / PIL images
    * `wandb.Histogram`         - log distributions (e.g. weight tensors)
    * Matplotlib figures        - wrap a `fig` in `wandb.Image(fig)` to log it
    * `wandb.plot.confusion_matrix` - W&B's first-class confusion-matrix plot

After running this, open the Workspace UI; the Media panel will show the
images and the confusion matrix, and the Custom Charts panel will hold the
weight histograms and matplotlib figure.

Gotcha worth knowing:
    `wandb.log` commits a step as soon as the *next* call uses a higher step.
    Repeated `wandb.log` calls at the *same* step are silently dropped once
    the step has been committed (you'll see "Step ... is less than the
    current step" warnings in the console). The pattern below is therefore:
    one consolidated dictionary per step. The end-of-run block builds a
    single `end_metrics` dict containing histograms + matplotlib figure +
    confusion matrix, then logs them in one `wandb.log` call at a
    deliberately fresh `end_step`.
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

                # Build ONE dict per step. If we also want to log sample images
                # this iteration, merge them into the same dict instead of
                # making a second wandb.log call (which would be dropped).
                step_metrics: dict = {"loss": avg, "epoch": epoch + 1}
                if (i + 1) % 500 == 0:
                    sample = images[:4].squeeze(1).cpu().numpy()
                    # caption=label makes the dashboard thumbnail self-describing:
                    # each image gets the ground-truth MNIST digit underneath.
                    step_metrics["sample_images"] = [
                        wandb.Image(img, caption=f"label={labels[idx].item()}")
                        for idx, img in enumerate(sample)
                    ]

                wandb.log(step_metrics, step=global_step)
                running_loss = 0.0

    # End-of-run logging. Bump the step past the last in-loop log so this
    # commit is unambiguously a "new" step (avoids the same-step drop).
    end_step = global_step + 1
    end_metrics: dict = {}

    # Histograms of every weight tensor. Pre-compute the (counts, edges) tuple
    # via numpy and pass it as `np_histogram` so the wandb type stubs are
    # happy AND we control the bin count explicitly.
    for name, param in model.named_parameters():
        if "weight" in name:
            arr = param.detach().cpu().numpy().ravel()
            end_metrics[f"hist/{name}"] = wandb.Histogram(
                np_histogram=np.histogram(arr, bins=64)
            )

    # Matplotlib loss curve, wrapped in wandb.Image so wandb rasterizes it.
    fig, ax = plt.subplots()
    ax.plot(losses)
    ax.set_xlabel("logging step")
    ax.set_ylabel("loss")
    ax.set_title("Running loss")
    end_metrics["loss_curve_mpl"] = wandb.Image(fig)
    plt.close(fig)  # release the figure or matplotlib leaks memory across runs

    # Confusion matrix + final accuracy from the last mini-batch. .tolist() on
    # the numpy arrays makes wandb's Sequence type hints happy without changing
    # the data; for batch sizes around 64 the conversion cost is irrelevant.
    if last_outputs is not None and last_labels is not None:
        _, preds = torch.max(last_outputs, dim=1)
        end_metrics["final_batch_accuracy"] = (
            (preds == last_labels).float().mean().item()
        )
        end_metrics["confusion_matrix"] = wandb.plot.confusion_matrix(
            probs=None,
            y_true=last_labels.cpu().numpy().tolist(),
            preds=preds.cpu().numpy().tolist(),
            class_names=[str(i) for i in range(10)],
        )

    # ONE commit for the entire end-of-run payload.
    wandb.log(end_metrics, step=end_step)
    wandb.finish()


if __name__ == "__main__":
    # Suppress matplotlib's harmless "FigureCanvasAgg is non-interactive" message
    # when running headlessly inside a container.
    np.seterr(all="ignore")
    main()
