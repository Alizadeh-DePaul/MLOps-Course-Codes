# tests/test_training.py
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torchvision.transforms as transforms

from models.mnist_model import MNISTModel
from training.train import train_epoch

@pytest.fixture
def small_dataloader():
    """Create a small dataloader for testing."""
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = MNIST("data", transform=transform, train=True, download=True)
    # Use a small subset for faster testing
    subset = torch.utils.data.Subset(dataset, range(20))
    return DataLoader(dataset=subset, batch_size=5, shuffle=False)

@pytest.fixture
def model():
    """Create a model for testing."""
    return MNISTModel()

@pytest.fixture
def optimizer(model):
    """Create an optimizer for testing."""
    return torch.optim.Adam(model.parameters(), lr=0.001)

def test_train_epoch(model, small_dataloader, optimizer):
    """Test that the training function runs without errors and returns expected values."""
    criterion = nn.CrossEntropyLoss()

    # Initial parameters
    initial_params = [param.clone().detach() for param in model.parameters()]

    # Run one training epoch
    loss, accuracy = train_epoch(model, small_dataloader, optimizer, criterion)

    # Check that loss is a scalar and not NaN
    assert isinstance(loss, float), f"Expected loss to be a float, got {type(loss)}"
    assert not torch.isnan(torch.tensor(loss)), "Loss should not be NaN"

    # Check that accuracy is between 0 and 100
    assert 0 <= accuracy <= 100, f"Accuracy should be between 0 and 100, got {accuracy}"

    # Check that parameters have been updated
    updated = False
    for i, param in enumerate(model.parameters()):
        if not torch.allclose(param, initial_params[i]):
            updated = True
            break
    assert updated, "Model parameters were not updated during training"
