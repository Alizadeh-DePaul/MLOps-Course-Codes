# tests/test_model.py
"""Sub-exercises 4.3, 4.5, and the parametrize section.

Goals
-----
- 4.3: assert the model produces output of the expected shape for a
  given input shape.
- 4.5: assert the model raises `ValueError` for invalid input shapes,
  using `pytest.raises`.
- parametrize: run the same shape test for several batch sizes using
  `@pytest.mark.parametrize`.

Run only this file with:
    pytest tests/test_model.py -v
"""
import pytest
import torch

from models.mnist_model import MNISTModel


@pytest.fixture
def model():
    """Build a fresh MNISTModel instance for each test."""
    return MNISTModel()


def test_model_output_shape(model):
    """For a batch of 64 inputs of shape (1, 28, 28), output is (64, 10).

    TODO:
    1. Build a random tensor `x` of shape (64, 1, 28, 28) with `torch.randn`.
    2. Put the model in eval mode and run a forward pass inside
       `torch.no_grad()`.
    3. Assert that the output shape is (64, 10). Include the actual
       shape in the failure message.
    """
    raise NotImplementedError


def test_model_output_values(model):
    """The model returns log-softmax, so values <= 0 and probs sum to 1.

    TODO:
    1. Build a (1, 1, 28, 28) input.
    2. Run forward pass.
    3. Assert all output values are <= 0.
    4. Exponentiate the output and assert the row sums to ~1 with
       `torch.isclose`.
    """
    raise NotImplementedError


def test_error_on_wrong_shape(model):
    """The model raises ValueError for wrong-dimensional inputs.

    TODO: use `pytest.raises(ValueError, match=...)` to verify that:
      - a 3D input (no channel dim) raises a "4D tensor" error
      - an input with 3 channels raises a "1 channel" error
    """
    raise NotImplementedError


@pytest.mark.parametrize("batch_size", [1, 16, 32, 64])
def test_model_with_different_batch_sizes(model, batch_size):
    """Same shape test, parametrized over batch sizes.

    TODO: build an input of shape (batch_size, 1, 28, 28), run a
    forward pass, and assert the output shape is (batch_size, 10).
    """
    raise NotImplementedError
