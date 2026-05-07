"""Section 2, final exercise — profile a full training run, not just a forward pass.

Wrap the same VAE training loop from vae_mnist.py in torch.profiler. After
running, open the resulting trace in TensorBoard or Perfetto and answer:

    Is the bottleneck still the forward pass, or is it something else?
    (Hint: look at where time is spent between batches, not inside them.)

Run:
    python profile_training.py
    tensorboard --logdir=./log    # then http://localhost:6006/#pytorch_profiler
    # or: open https://ui.perfetto.dev/ and drag in log/training/*.pt.trace.json

Notes
-----
- The schedule (wait=2, warmup=2, active=6, repeat=1) profiles 10 batches
  per epoch — 2 are skipped, 2 are used to warm up the profiler, 6 are
  recorded. That's enough signal to identify a bottleneck without
  generating a multi-gigabyte trace.
- record_function() blocks let you tag custom regions (loss, backward).
  They show up as their own rows in the trace timeline.
"""
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.optim import Adam
from torch.profiler import (
    ProfilerActivity,
    profile,
    record_function,
    schedule,
    tensorboard_trace_handler,
)
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.utils import save_image

# Re-use the model classes from the clean baseline so we don't duplicate code.
from vae_mnist import Decoder, Encoder, Model, loss_function

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
dataset_path = "datasets"
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
epochs = 1   # one epoch is enough to fill the profiler schedule


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
mnist_transform = transforms.Compose([transforms.ToTensor()])
train_dataset = MNIST(dataset_path, transform=mnist_transform, train=True, download=True)
test_dataset = MNIST(dataset_path, transform=mnist_transform, train=False, download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
encoder = Encoder(input_dim=x_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
decoder = Decoder(latent_dim=latent_dim, hidden_dim=hidden_dim, output_dim=x_dim)
model = Model(encoder=encoder, decoder=decoder).to(DEVICE)
optimizer = Adam(model.parameters(), lr=lr)


activities = [ProfilerActivity.CPU]
if torch.cuda.is_available():
    activities.append(ProfilerActivity.CUDA)


# ---------------------------------------------------------------------------
# Train (with profiler wrapping the loop)
# ---------------------------------------------------------------------------
print(f"Profiling VAE training on {DEVICE} -> log/training/ ...")
model.train()
for epoch in range(epochs):
    with profile(
        activities=activities,
        schedule=schedule(wait=2, warmup=2, active=6, repeat=1),
        on_trace_ready=tensorboard_trace_handler("./log/training"),
        record_shapes=True,
        with_stack=True,
    ) as profiler:
        overall_loss = 0
        for batch_idx, (x, _) in enumerate(train_loader):
            if batch_idx % 100 == 0:
                print(batch_idx)
            x = x.view(batch_size, x_dim).to(DEVICE)

            optimizer.zero_grad()
            x_hat, mean, log_var = model(x)

            with record_function("model_loss"):
                loss = loss_function(x, x_hat, mean, log_var)

            overall_loss += loss.item()

            with record_function("backward"):
                loss.backward()
                optimizer.step()

            profiler.step()

        print(
            "\tEpoch", epoch + 1, "complete!",
            "\tAverage Loss: ", overall_loss / (batch_idx * batch_size),
        )

print("Finish!!")

# Generate reconstructions (outside the profiler, just to confirm training worked)
model.eval()
with torch.no_grad():
    for batch_idx, (x, _) in enumerate(test_loader):
        x = x.view(batch_size, x_dim).to(DEVICE)
        x_hat, _, _ = model(x)
        break

save_image(x.view(batch_size, 1, 28, 28), "orig_data.png")
save_image(x_hat.view(batch_size, 1, 28, 28), "reconstructions.png")
with torch.no_grad():
    noise = torch.randn(batch_size, latent_dim).to(DEVICE)
    save_image(decoder(noise).view(batch_size, 1, 28, 28), "generated_sample.png")

print()
print("Visualize with EITHER of:")
print("  TensorBoard:  tensorboard --logdir=./log")
print("                then open http://localhost:6006/#pytorch_profiler")
print("  Perfetto:     open https://ui.perfetto.dev/ and drag in any")
print("                .pt.trace.json file from log/training/")
