# tests/test_data.py
"""Sub-exercise 4.2: Data Testing.

Goals
-----
- Verify the MNIST training and test splits have the expected number of
  samples (60,000 and 10,000 respectively).
- Verify each sample has shape (1, 28, 28).
- Verify all 10 digit classes (0-9) are represented.

Run only this file with:
    pytest tests/test_data.py -v

Notes
-----
- `_PATH_DATA` is defined in `tests/__init__.py` and points at the
  `data/` folder next to this package. Pass it to MNIST(...) so the
  dataset is cached locally.
- Use `download=True` so the dataset is fetched the first time you run
  the test, then cached for subsequent runs.
- Add a meaningful message to every assert (the exercise asks for this
  in sub-exercise 4.6).
"""
import pytest
import torch  # noqa: F401  (torch is required transitively by torchvision)
from torchvision.datasets import MNIST
from torchvision import transforms

from tests import _PATH_DATA

# Expected sample counts.
N_TRAIN = 60000
N_TEST = 10000


@pytest.fixture
def train_dataset():
    """Build the MNIST training dataset for reuse across tests.

    TODO: return an MNIST instance rooted at `_PATH_DATA`, with
    `train=True`, `download=True`, and `transforms.ToTensor()` as the
    transform.
    """
    raise NotImplementedError("Build the MNIST training fixture")


@pytest.fixture
def test_dataset():
    """Same as `train_dataset` but with `train=False`.

    TODO: return the MNIST test split.
    """
    raise NotImplementedError("Build the MNIST test fixture")


def test_dataset_size(train_dataset, test_dataset):
    """Train split has N_TRAIN samples, test split has N_TEST.

    TODO: write two asserts. Include a message so a failure tells you
    which split was wrong and what the actual count was.
    """
    raise NotImplementedError


def test_data_shape(train_dataset):
    """Each sample tensor has shape (1, 28, 28).

    TODO: pull the first sample with `train_dataset[0]` and assert its
    `.shape` matches (1, 28, 28).
    """
    raise NotImplementedError


def test_label_distribution(train_dataset):
    """All 10 digit classes appear in the first 1000 training samples.

    TODO: iterate over `train_dataset[0:1000]`, collect each label into a
    set, and assert the set has size 10. Include the actual set in the
    failure message so you know which class is missing.
    """
    raise NotImplementedError
