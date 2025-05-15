
import logging
import torch

logger = logging.getLogger(__name__)

# Example of logging during PyTorch training
def train_epoch(model, train_loader, optimizer, epoch):
    model.train()
    running_loss = 0.0

    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = torch.nn.functional.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if batch_idx % 10 == 0:
            logger.info(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                       f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

    avg_loss = running_loss / len(train_loader)
    logger.info(f'Epoch {epoch}: Average training loss: {avg_loss:.4f}')
    return avg_loss

