# tests/test_error_handling.py
"""Sub-exercise 4.5: Error handling with `pytest.raises`, plus
sub-exercise 4.7: `pytest.mark.skipif`.

Goal
----
- Demonstrate `pytest.raises` over multiple parametrized invalid inputs.
- Demonstrate `pytest.mark.skipif` to skip a test when a prerequisite
  is not met (here: a file that doesn't exist).
"""
import os.path

import pytest
import torch

from models.mnist_model import MNISTModel


@pytest.fixture
def model():
    return MNISTModel()


def test_model_with_invalid_inputs(model):
    """Multiple invalid inputs each raise the expected ValueError.

    TODO: iterate over the table below and, for each (input, expected
    message) pair where the message is not None, assert the model
    raises ValueError matching that message. For the pair where the
    message is None, the model should run without raising.
    """
    cases = [
        (torch.randn(10, 28, 28),    "Expected input to be a 4D tensor"),
        (torch.randn(10, 3, 28, 28), "Expected input to have 1 channel"),
        (torch.randn(10, 1, 14, 14), None),  # valid shape, just smaller
    ]
    raise NotImplementedError


# Sub-exercise 4.7 — pytest.mark.skipif demo.
# `_PRETRAINED_PATH` doesn't exist in the starter repo, so this test
# skips by design. If you later add a pretrained checkpoint and want
# the test to actually run, drop a real file at this path and rerun.
_PRETRAINED_PATH = os.path.join(os.path.dirname(__file__), "mnist_pretrained.pt")


@pytest.mark.skipif(
    not os.path.exists(_PRETRAINED_PATH),
    reason="No pretrained checkpoint found; skipping warm-start test",
)
def test_warm_start_from_checkpoint(model):
    """Placeholder: loads a pretrained checkpoint and checks one thing.

    TODO (optional): if you actually have a checkpoint, load it with
    `model.load_state_dict(torch.load(_PRETRAINED_PATH))` and assert a
    forward pass still works. Otherwise leave this test as is so you
    can see "skipped" appear in the pytest output.
    """
    state = torch.load(_PRETRAINED_PATH)
    model.load_state_dict(state)
    x = torch.randn(1, 1, 28, 28)
    out = model(x)
    assert out.shape == (1, 10), f"Expected (1, 10), got {out.shape}"
