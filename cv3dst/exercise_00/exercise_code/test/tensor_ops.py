import torch

from .base_tests import CompositeTest, UnitTest, test_results_to_score

from ..tensor_ops import colorize_labels, select_scores, stable_softmax


class OutputTest(UnitTest):
    """Compares the output of fn(*inputs) with the expected tensor."""

    def __init__(self, name: str, fn, inputs: tuple, expected: torch.Tensor) -> None:
        self.name, self.fn, self.inputs, self.expected = name, fn, inputs, expected
        self.output = None

    def test(self):
        self.output = self.fn(*self.inputs)
        return (
            isinstance(self.output, torch.Tensor)
            and self.output.shape == self.expected.shape
            and self.output.dtype == self.expected.dtype
            and torch.allclose(self.output, self.expected, atol=1e-6)
        )

    def define_success_message(self):
        return f"Congratulations: {self.fn.__name__} passed the test for {self.name}."

    def define_failure_message(self):
        msg = f"{self.fn.__name__} failed the test for {self.name}: "
        if not isinstance(self.output, torch.Tensor):
            return msg + f"expected a torch.Tensor, got {type(self.output).__name__}."
        if self.output.shape != self.expected.shape:
            return (
                msg
                + f"expected shape {tuple(self.expected.shape)}, got {tuple(self.output.shape)}."
            )
        if self.output.dtype != self.expected.dtype:
            return (
                msg + f"expected dtype {self.expected.dtype}, got {self.output.dtype}."
            )
        return msg + "the values are wrong."


class SelectScoresTest(CompositeTest):
    def define_tests(self):
        scores, labels = torch.randn(100, 10), torch.randint(0, 10, (100,))
        scores_t = torch.randn(10, 50).T  # non-contiguous input
        labels_t = torch.randint(0, 10, (50,))
        return [
            OutputTest(
                "random scores",
                select_scores,
                (scores, labels),
                torch.gather(scores, 1, labels[:, None])[:, 0],
            ),
            OutputTest(
                "a single sample",
                select_scores,
                (scores[:1], labels[:1]),
                scores[:1, labels[0]],
            ),
            OutputTest(
                "a transposed input",
                select_scores,
                (scores_t, labels_t),
                scores_t.gather(1, labels_t[:, None])[:, 0],
            ),
        ]


class StableSoftmaxTest(CompositeTest):
    def define_tests(self):
        x, x3 = torch.randn(20, 10), torch.randn(4, 5, 6)
        large = 1000 * torch.randn(20, 10)
        return [
            OutputTest("dim=1", stable_softmax, (x, 1), torch.softmax(x, dim=1)),
            OutputTest("dim=0", stable_softmax, (x, 0), torch.softmax(x, dim=0)),
            OutputTest(
                "a 3D input with dim=-1",
                stable_softmax,
                (x3, -1),
                torch.softmax(x3, dim=-1),
            ),
            OutputTest(
                "large inputs (overflow of exp)",
                stable_softmax,
                (large, 1),
                torch.softmax(large, dim=1),
            ),
        ]


class ColorizeLabelsTest(CompositeTest):
    def define_tests(self):
        palette = torch.randint(0, 256, (21, 3), dtype=torch.uint8)
        label_map = torch.randint(0, 21, (32, 48))
        batch = torch.randint(0, 21, (2, 16, 16))
        expected = torch.zeros(32, 48, 3, dtype=torch.uint8)
        for y in range(32):
            for x in range(48):
                expected[y, x] = palette[label_map[y, x]]
        return [
            OutputTest("a label map", colorize_labels, (label_map, palette), expected),
            OutputTest(
                "a batch of label maps",
                colorize_labels,
                (batch, palette),
                torch.stack([palette[b] for b in batch]),
            ),
        ]


def test_select_scores():
    return test_results_to_score(SelectScoresTest()())


def test_stable_softmax():
    return test_results_to_score(StableSoftmaxTest()())


def test_colorize_labels():
    return test_results_to_score(ColorizeLabelsTest()())
