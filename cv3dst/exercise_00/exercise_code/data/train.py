import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


def train_step(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Performs one optimization step on the batch (X, y) and returns the predictions and the loss."""
    y_pred, loss = None, None

    ########################################################################
    # TODO:                                                                #
    # Complete the training step. This includes                            #
    # 1. moving the batch data to the device                               #
    # 2. resetting the gradients of the parameters                         #
    # 3. the forward pass through the model                                #
    # 4. calculating the loss                                              #
    # 5. the backward pass                                                 #
    # 6. updating the parameters with the optimizer                        #
    # y_pred = ...                                                         #
    # loss = ...                                                           #
    ########################################################################


    pass

    ########################################################################
    #                           END OF YOUR CODE                           #
    ########################################################################

    return y_pred, loss


def train(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    num_epochs: int,
    device: torch.device,
    log_every: int = 100,
) -> tuple[list[float], list[float]]:
    """Trains the model and returns the average loss and accuracy of every log_every batches."""
    loss_history, acc_history = [], []
    model.train()
    for epoch in range(num_epochs):
        running_loss, correct, total = 0.0, 0, 0
        for i, (X, y) in enumerate(dataloader):
            y_pred, loss = train_step(model, X, y, optimizer, criterion, device)

            running_loss += loss.item()
            correct += (y_pred.argmax(dim=1).cpu() == y).sum().item()
            total += len(y)
            if i % log_every == log_every - 1:
                loss_history.append(running_loss / log_every)
                acc_history.append(correct / total)
                print(
                    f"[Epoch {epoch + 1}, Iteration {i + 1:5d}] "
                    f"loss: {loss_history[-1]:.3f} acc: {100 * acc_history[-1]:.2f} %"
                )
                running_loss, correct, total = 0.0, 0, 0

    print("FINISH.")
    return loss_history, acc_history


def evaluate(
    model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device: torch.device
) -> tuple[float, float]:
    """Returns the average loss and the accuracy of the model on a dataset."""
    model.eval()
    loss, correct = 0.0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            loss += criterion(y_pred, y).item() * len(y)
            correct += (y_pred.argmax(dim=1) == y).sum().item()
    model.train()
    n = len(dataloader.dataset)
    return loss / n, correct / n
