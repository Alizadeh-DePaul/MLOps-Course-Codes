import wandb
import torch
import torch.nn as nn
import torchvision

# 1. Start a new run
wandb.init(project="Week7-project", name="Week7-experiment")

# 2. Configure model and training parameters
config = wandb.config  # Initialize config
config.batch_size = 64
config.learning_rate = 0.01
config.epochs = 10

# 3. Create a simple model
model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

# 4. Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate)

# 5. Load data (simplified example)
train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./data', train=True, download=True,
                             transform=torchvision.transforms.ToTensor()),
    batch_size=config.batch_size, shuffle=True)

# 6. Training loop with WandB logging
for epoch in range(config.epochs):
    running_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):
        # Flatten the images
        images = images.view(images.shape[0], -1)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if i % 100 == 99:  # Log every 100 mini-batches
            wandb.log({
                "epoch": epoch + 1,
                "batch": i + 1,
                "loss": running_loss / 100
            })
            running_loss = 0.0

# 7. Close the run
wandb.finish()
