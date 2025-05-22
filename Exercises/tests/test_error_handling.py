# tests/test_error_handling.py
import pytest
import torch
from models.mnist_model import MNISTModel

@pytest.fixture
def model():
    return MNISTModel()

def test_model_with_invalid_inputs(model):
    """Test that the model properly handles various invalid inputs."""
    # Test cases and expected error messages
    test_cases = [
        (torch.randn(10, 28, 28), 'Expected input to be a 4D tensor'),  # Wrong dimensions
        (torch.randn(10, 3, 28, 28), 'Expected input to have 1 channel'),  # Wrong channels
        (torch.randn(10, 1, 14, 14), None),  # Wrong size but should not raise error
    ]

    for input_tensor, error_msg in test_cases:
        if error_msg:
            with pytest.raises(ValueError, match=error_msg):
                model(input_tensor)
        else:
            try:
                model(input_tensor)  # Should not raise an error
            except ValueError as e:
                pytest.fail(f"Unexpected ValueError: {e}")
