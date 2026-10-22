import copy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from .base_tests import CompositeTest, UnitTest, test_results_to_score

from ..data.network import Net
from ..data.train import evaluate, train, train_step


class TrainStepOutputTest(UnitTest):
    def __init__(self) -> None:
        self.model = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 10))
        self.X, self.y = torch.randn(8, 1, 28, 28), torch.randint(0, 10, (8,))
        self.msg = ""

    def test(self):
        criterion = nn.CrossEntropyLoss()
        expected = criterion(self.model(self.X), self.y)
        optimizer = optim.SGD(self.model.parameters(), lr=0.1)
        y_pred, loss = train_step(
            self.model, self.X, self.y, optimizer, criterion, torch.device("cpu")
        )
        if not isinstance(y_pred, torch.Tensor) or y_pred.shape != (8, 10):
            self.msg = "train_step should return the predictions of the model of shape (B, num_classes)."
        elif not isinstance(loss, torch.Tensor) or not torch.allclose(loss, expected):
            self.msg = "train_step should return the loss of the batch as a tensor, computed with criterion."
        return not self.msg

    def define_failure_message(self):
        return self.msg


class TrainStepUpdateTest(UnitTest):
    """Compares three training steps with a reference implementation."""

    def __init__(self) -> None:
        self.model = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 10))
        self.ref = copy.deepcopy(self.model)
        self.batches = [
            (torch.randn(8, 1, 28, 28), torch.randint(0, 10, (8,))) for _ in range(3)
        ]

    def test(self):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.model.parameters(), lr=0.1, momentum=0.9)
        ref_optimizer = optim.SGD(self.ref.parameters(), lr=0.1, momentum=0.9)
        for X, y in self.batches:
            train_step(self.model, X, y, optimizer, criterion, torch.device("cpu"))
            ref_optimizer.zero_grad()
            criterion(self.ref(X), y).backward()
            ref_optimizer.step()
        return all(
            torch.allclose(p, q, atol=1e-6)
            for p, q in zip(self.model.parameters(), self.ref.parameters())
        )

    def define_failure_message(self):
        return (
            "The parameters after three calls of train_step are incorrect. "
            "Did you reset the gradients, call backward() on the loss and update the parameters with the optimizer?"
        )


class TrainStepTest(CompositeTest):
    def define_tests(self):
        return [TrainStepOutputTest(), TrainStepUpdateTest()]


class ImprovementTest(UnitTest):
    def __init__(
        self, name: str, before: float, after: float, higher_is_better: bool
    ) -> None:
        self.name, self.before, self.after, self.higher = (
            name,
            before,
            after,
            higher_is_better,
        )

    def test(self):
        return self.after > self.before if self.higher else self.after < self.before

    def define_success_message(self):
        return f"Congratulations: the test {self.name} improved from {self.before:.3f} to {self.after:.3f}."

    def define_failure_message(self):
        return f"The test {self.name} did not improve after one epoch of training ({self.before:.3f} -> {self.after:.3f})."


class OneEpochTest(CompositeTest):
    def define_tests(self, before: tuple[float, float], after: tuple[float, float]):
        return [
            ImprovementTest("loss", before[0], after[0], higher_is_better=False),
            ImprovementTest("accuracy", before[1], after[1], higher_is_better=True),
        ]


def test_train_step():
    torch.manual_seed(0)
    return test_results_to_score(TrainStepTest()())


def test_one_epoch(
    train_loader: DataLoader, test_loader: DataLoader, device: torch.device
):
    """Trains a new Net for one epoch and checks that the test loss and accuracy improve."""
    torch.manual_seed(0)
    net = Net().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    before = evaluate(net, test_loader, criterion, device)
    train(
        net, train_loader, optimizer, criterion, 1, device, log_every=len(train_loader)
    )
    after = evaluate(net, test_loader, criterion, device)
    return test_results_to_score(OneEpochTest(before, after)())
