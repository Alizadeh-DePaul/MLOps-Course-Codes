"""Tiny CML-friendly training script.

The goal of this exercise is the *reporting pipeline*, not the model. We
deliberately pick the smallest reasonable problem so the whole demo runs
inside GitHub Actions' free tier in well under a minute.

What this script does:
    1. Loads sklearn's bundled `digits` dataset (no download, no internet).
    2. Trains a logistic-regression classifier.
    3. Writes a human-readable `classification_report.txt`.
    4. Writes a `confusion_matrix.png` plot.

Both output files end up in the working directory, ready for the CML
workflow (`.github/workflows/cml.yaml`) to embed in a Markdown report and
post as a PR comment.

If you want to swap in your own model or dataset, the only contract is:
    - Produce `classification_report.txt` (any plain-text report works).
    - Produce `confusion_matrix.png` (any PNG works).
The workflow doesn't care how you got there.
"""

from __future__ import annotations

import matplotlib

# Use the non-interactive Agg backend so the script works on headless CI
# runners that have no display server. This MUST be set before importing
# pyplot.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (intentional ordering)
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split


def main() -> None:
    # --- 1. Load data ------------------------------------------------------
    # 1,797 8x8 grayscale digit images bundled with sklearn. No download.
    digits = load_digits()
    x_train, x_test, y_train, y_test = train_test_split(
        digits.data,
        digits.target,
        test_size=0.25,
        random_state=489,
    )

    # --- 2. Train ----------------------------------------------------------
    # `max_iter=200` is enough for the bundled digits dataset and keeps the
    # training time well under a second. Bump it if you change the model.
    model = LogisticRegression(max_iter=200, random_state=489)
    model.fit(x_train, y_train)
    preds = model.predict(x_test)

    # --- 3. Classification report ------------------------------------------
    # sklearn's classification_report returns a formatted string with
    # per-class precision/recall/f1. We write it verbatim so `cat` can dump
    # it into the Markdown report on the CI runner.
    report = classification_report(y_test, preds, digits=3)
    with open("classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)
    print(report)

    # --- 4. Confusion matrix plot ------------------------------------------
    # ConfusionMatrixDisplay handles the labeling for us; we just give it
    # the matrix and the class labels and let it draw.
    cm = confusion_matrix(y_test, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(range(10)))
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, colorbar=False)
    ax.set_title("Digits classifier - confusion matrix")
    fig.tight_layout()
    fig.savefig("confusion_matrix.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
