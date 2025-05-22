import matplotlib.pyplot as plt
import numpy as np
import wandb
import torch
import torch.nn as nn
import torchvision

# 1. Start a new run
wandb.init(project="Week7-project", entity="WANDB USERNAME" name="Week7-experiment-advanced")

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

# During training, after processing a batch:

# Log sample images
images_sample = images[:4].view(-1, 28, 28).cpu().numpy()
wandb.log({
    "example_images": [wandb.Image(img) for img in images_sample],
})

# Log histograms of model weights
for name, param in model.named_parameters():
    if 'weight' in name:
        wandb.log({f"histogram/{name}": wandb.Histogram(param.detach().cpu().numpy())})

# Log a custom matplotlib figure
fig, ax = plt.subplots()
ax.plot(np.random.randn(100).cumsum())
ax.set_title("Random Walk")
wandb.log({"random_walk": wandb.Image(fig)})
plt.close(fig)  # Don't forget to close the figure!

# Log model predictions
_, preds = torch.max(outputs, 1)
wandb.log({
    "accuracy": (preds == labels).float().mean().item(),
    "confusion_matrix": wandb.plot.confusion_matrix(
        probs=None,
        y_true=labels.cpu().numpy(),
        preds=preds.cpu().numpy(),
        class_names=list(range(10))
    )
})

# 7. Close the run
wandb.finish()







