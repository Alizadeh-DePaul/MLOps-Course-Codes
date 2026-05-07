"""Clean-baseline Gaussian-MLP VAE on MNIST.

This is the starting point for the Performance Profiling exercise. The four
bugs from the previous (ML Code Debugging) exercise have been fixed; what
ships here is a working but unoptimized training script. Your job in this
exercise is to *profile* it — first with cProfile + snakeviz, then with
torch.profiler + TensorBoard / Perfetto UI — and identify where the time
actually goes.

Adapted from https://github.com/Jackson-Kang/Pytorch-VAE-tutorial.

Note for profiling
------------------
The `epochs = 5` default below is deliberately small so a profiling run
finishes in under a few minutes on CPU. If you want a quicker pass while
iterating on profiler arguments, drop it to 1.
"""
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.utils import save_image

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
dataset_path = "datasets"

# Runtime device detection: cuda > mps > cpu. Same script runs on every
# student machine; do not hardcode `cuda = True` like the buggy version did.
device_name = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
DEVICE = torch.device(device_name)

batch_size = 100
x_dim = 784
hidden_dim = 400
latent_dim = 20
lr = 1e-3
epochs = 5


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
mnist_transform = transforms.Compose([transforms.ToTensor()])

train_dataset = MNIST(dataset_path, transform=mnist_transform, train=True, download=True)
test_dataset = MNIST(dataset_path, transform=mnist_transform, train=False, download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class Encoder(nn.Module):
    """Gaussian MLP Encoder."""

    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.FC_input = nn.Linear(input_dim, hidden_dim)
        self.FC_mean = nn.Linear(hidden_dim, latent_dim)
        self.FC_var = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        h_ = torch.relu(self.FC_input(x))
        mean = self.FC_mean(h_)
        log_var = self.FC_var(h_)

        std = torch.exp(0.5 * log_var)
        z = self.reparameterization(mean, std)
        return z, mean, log_var

    def reparameterization(self, mean, std):
        # randn_like inherits dtype + device from std.
        epsilon = torch.randn_like(std)
        return mean + std * epsilon


class Decoder(nn.Module):
    """Bernoulli MLP Decoder."""

    def __init__(self, latent_dim, hidden_dim, output_dim):
        super().__init__()
        self.FC_hidden = nn.Linear(latent_dim, hidden_dim)
        self.FC_output = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h = torch.relu(self.FC_hidden(x))
        return torch.sigmoid(self.FC_output(h))


class Model(nn.Module):
    """VAE wrapper that ties encoder and decoder together."""

    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        z, mean, log_var = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, mean, log_var


encoder = Encoder(input_dim=x_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
decoder = Decoder(latent_dim=latent_dim, hidden_dim=hidden_dim, output_dim=x_dim)
model = Model(encoder=encoder, decoder=decoder).to(DEVICE)


def loss_function(x, x_hat, mean, log_var):
    """ELBO loss: reconstruction + KL divergence, summed over batch and dims."""
    reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction="sum")
    kld = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    return reproduction_loss + kld


optimizer = Adam(model.parameters(), lr=lr)


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
def main():
    print(f"Start training VAE on {DEVICE}...")
    model.train()
    for epoch in range(epochs):
        overall_loss = 0
        for batch_idx, (x, _) in enumerate(train_loader):
            if batch_idx % 100 == 0:
                print(batch_idx)
            x = x.view(batch_size, x_dim).to(DEVICE)

            optimizer.zero_grad()
            x_hat, mean, log_var = model(x)
            loss = loss_function(x, x_hat, mean, log_var)
            overall_loss += loss.item()

            loss.backward()
            optimizer.step()

        print(
            "\tEpoch", epoch + 1, "complete!",
            "\tAverage Loss: ", overall_loss / (batch_idx * batch_size),
        )
    print("Finish!!")

    # Generate reconstructions
    model.eval()
    with torch.no_grad():
        for batch_idx, (x, _) in enumerate(test_loader):
            x = x.view(batch_size, x_dim).to(DEVICE)
            x_hat, _, _ = model(x)
            break

    save_image(x.view(batch_size, 1, 28, 28), "orig_data.png")
    save_image(x_hat.view(batch_size, 1, 28, 28), "reconstructions.png")

    # Generate samples from latent prior
    with torch.no_grad():
        noise = torch.randn(batch_size, latent_dim).to(DEVICE)
        generated_images = decoder(noise)
    save_image(generated_images.view(batch_size, 1, 28, 28), "generated_sample.png")


if __name__ == "__main__":
    main()
