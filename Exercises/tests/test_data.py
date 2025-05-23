# tests/test_data.py
import pytest
import torch
from torchvision.datasets import MNIST
from torchvision import transforms
from tests import _PATH_DATA

# Define the expected number of samples
N_TRAIN = 60000
N_TEST = 10000

@pytest.fixture
def train_dataset():
    """Create a fixture for the training dataset that can be reused across tests."""
    transform = transforms.Compose([transforms.ToTensor()])
    return MNIST(_PATH_DATA, train=True, download=True, transform=transform)

@pytest.fixture
def test_dataset():
    """Create a fixture for the test dataset that can be reused across tests."""
    transform = transforms.Compose([transforms.ToTensor()])
    return MNIST(_PATH_DATA, train=False, download=True, transform=transform)


# @pytest.mark.skipif(not os.path.exists (_PATH_DATA), reason="MNIST data not found")
def test_dataset_size(train_dataset, test_dataset):
    """Check if datasets have the expected number of samples."""
    assert len(train_dataset) == N_TRAIN, f"Expected {N_TRAIN} samples, got {len(train_dataset)}"
    assert len(test_dataset) == N_TEST, f"Expected {N_TEST} samples, got {len(test_dataset)}"

def test_data_shape(train_dataset):
    """Verify the shape of data samples."""
    data, _ = train_dataset[0]
    assert data.shape == (1, 28, 28), f"Expected shape (1, 28, 28), got {data.shape}"

def test_label_distribution(train_dataset):
    """Check that all classes (0-9) are represented in the dataset."""
    labels = set()
    # Only check first 1000 samples for speed
    for i in range(1000):
        _, label = train_dataset[i]
        labels.add(label)

    assert len(labels) == 10, f"Expected 10 classes, got {len(labels)}: {labels}"
