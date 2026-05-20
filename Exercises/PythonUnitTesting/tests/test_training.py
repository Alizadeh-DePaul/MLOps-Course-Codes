# tests/test_training.py
"""Sub-exercise 4.4: Training Testing.

Goal
----
Verify that `train_epoch(...)` from `training/train.py` runs end-to-end
and behaves sensibly. Concretely:
- Loss is a finite float (not NaN).
- Accuracy is in [0, 100].
- Model parameters change after one training step (the gradient
  actually flows).

Run only this file with:
    pytest tests/test_training.py -v
"""
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import MNIST
from torchvision import transforms

from models.mnist_model import MNISTModel
from training.train import train_epoch
from tests import _PATH_DATA


@pytest.fixture
def small_dataloader():
    """A 20-sample, batch-size-5 loader so the test runs in seconds.

    TODO: build an MNIST training dataset (download=True, ToTensor),
    wrap a 20-sample Subset, and return a DataLoader with batch_size=5.
    """
    raise NotImplementedError


@pytest.fixture
def model():
    return MNISTModel()


@pytest.fixture
def optimizer(model):
    return torch.optim.Adam(model.parameters(), lr=0.001)


def test_train_epoch(model, small_dataloader, optimizer):
    """One training epoch updates parameters and produces a valid loss.

    TODO:
    1. Snapshot the initial model parameters (clone+detach each one).
    2. Call `train_epoch(model, small_dataloader, optimizer, criterion)`
       with `criterion = nn.CrossEntropyLoss()`.
    3. Assert the returned loss is a float and not NaN.
    4. Assert accuracy is between 0 and 100.
    5. Assert that at least one parameter changed between the snapshot
       and the post-step values (otherwise training did nothing).
    """
    raise NotImplementedError
