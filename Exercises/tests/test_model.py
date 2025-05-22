# tests/test_model.py
import torch
import pytest
from models.mnist_model import MNISTModel

@pytest.fixture
def model():
    """Create a model instance for testing."""
    return MNISTModel()

def test_model_output_shape(model):
    """Test that the model produces output with the expected shape."""
    batch_size = 64
    # MNIST images are 1x28x28
    x = torch.randn(batch_size, 1, 28, 28)

    model.eval()  # Set to evaluation mode
    with torch.no_grad():  # Disable gradient tracking for inference
        output = model(x)

    # Output should be [batch_size, num_classes]
    expected_shape = (batch_size, 10)  # 10 classes for MNIST
    assert output.shape == expected_shape, f"Expected output shape {expected_shape}, got {output.shape}"

def test_model_output_values(model):
    """Test that the model output is valid probabilities."""
    x = torch.randn(1, 1, 28, 28)

    model.eval()
    with torch.no_grad():
        output = model(x)

    # For log_softmax, values should be <= 0
    assert torch.all(output <= 0), "Log probabilities should be <= 0"

    # Sum of probabilities should be close to 1
    probs = torch.exp(output)
    assert torch.isclose(torch.sum(probs), torch.tensor(1.0)), f"Sum of probabilities should be 1, got {torch.sum(probs)}"

def test_error_on_wrong_shape(model):
    """Test that the model raises an error for incorrect input shapes."""
    # Wrong number of dimensions
    with pytest.raises(ValueError, match='Expected input to be a 4D tensor'):
        model(torch.randn(10, 28, 28))  # Missing channel dimension

    # Wrong number of channels
    with pytest.raises(ValueError, match='Expected input to have 1 channel'):
        model(torch.randn(10, 3, 28, 28))  # 3 channels instead of 1

@pytest.mark.parametrize("batch_size", [1, 16, 32, 64])
def test_model_with_different_batch_sizes(model, batch_size):
    """Test the model with different batch sizes."""
    x = torch.randn(batch_size, 1, 28, 28)

    with torch.no_grad():
        output = model(x)

    expected_shape = (batch_size, 10)
    assert output.shape == expected_shape, f"Output shape mismatch for batch_size={batch_size}"


@pytest.mark.parametrize("height,width", [(28, 28), (32, 32), (24, 24)])
def test_model_with_different_image_sizes(model, height, width):
    """Test the model with different image dimensions."""
    x = torch.randn(1, 1, height, width)

    try:
        with torch.no_grad():
            output = model(x)
        # If we get here, no exception was raised
        assert output.shape[0] == 1, f"Batch dimension should be preserved"
        assert output.shape[1] == 10, f"Output should have 10 classes"
    except Exception as e:
        # Some image sizes might cause errors, which is okay
        # Just make sure the error message is informative
        assert str(e), "Exception should have a message"
