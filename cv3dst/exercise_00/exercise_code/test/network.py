import torch

from .base_tests import CompositeTest, UnitTest, test_results_to_score

from ..data.network import Net


class OutputShapeTest(UnitTest):
    def __init__(self) -> None:
        self.shape = None

    def test(self):
        self.shape = tuple(Net()(torch.randn(4, 1, 28, 28)).shape)
        return self.shape == (4, 10)

    def define_failure_message(self):
        return f"The output of Net for a batch of 4 images has the wrong shape (expected (4, 10), got {self.shape})."


class ParameterCountTest(UnitTest):
    def __init__(self) -> None:
        self.count = None

    def test(self):
        self.count = sum(p.numel() for p in Net().parameters())
        return 0 < self.count < 500_000

    def define_failure_message(self):
        return (
            f"Net should have between 1 and 500,000 parameters, but has {self.count}."
        )


class NetTest(CompositeTest):
    def define_tests(self):
        return [OutputShapeTest(), ParameterCountTest()]


def test_network():
    return test_results_to_score(NetTest()())
