import tempfile
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader

from .base_tests import CompositeTest, UnitTest, test_results_to_score

from ..data.dataset import MNISTDataset

MEAN, STD = 0.1307, 0.3081


def write_samples(folder: Path, n: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Writes n random images and labels in the MNIST_png format."""
    imgs = torch.randint(0, 256, (n, 1, 28, 28), dtype=torch.uint8)
    labels = torch.randint(0, 10, (n,))
    for i, img in enumerate(imgs):
        Image.fromarray(img[0].numpy()).save(folder / f"{i:05d}.png")
    (folder / "labels.csv").write_text(
        "\n".join(f"{i:05d}.png,{label}" for i, label in enumerate(labels))
    )
    return imgs, labels


class LengthTest(UnitTest):
    def __init__(self, dataset: MNISTDataset, n: int) -> None:
        self.dataset, self.n, self.output = dataset, n, None

    def test(self):
        self.output = len(self.dataset)
        return self.output == self.n

    def define_failure_message(self):
        return f"len(dataset) is incorrect (expected {self.n}, got {self.output})."


class ImageTest(UnitTest):
    def __init__(self, dataset: MNISTDataset, imgs: torch.Tensor) -> None:
        self.dataset, self.imgs, self.img = dataset, imgs, None

    def test(self):
        self.img, _ = self.dataset[3]
        expected = (self.imgs[3].float() / 255 - MEAN) / STD
        return (
            isinstance(self.img, torch.Tensor)
            and self.img.shape == (1, 28, 28)
            and self.img.dtype == torch.float32
            and torch.allclose(self.img, expected, atol=1e-5)
        )

    def define_failure_message(self):
        msg = "The image returned by dataset[index] is incorrect: "
        if not isinstance(self.img, torch.Tensor):
            return msg + f"expected a torch.Tensor, got {type(self.img).__name__}."
        if self.img.shape != (1, 28, 28) or self.img.dtype != torch.float32:
            return (
                msg
                + f"expected shape (1, 28, 28) and dtype float32, got {tuple(self.img.shape)} and {self.img.dtype}."
            )
        return (
            msg
            + "the values are wrong. Did you scale the image to [0, 1] and normalize it with MEAN and STD?"
        )


class LabelTest(UnitTest):
    def __init__(self, dataset: MNISTDataset, labels: torch.Tensor) -> None:
        self.dataset, self.labels, self.label = dataset, labels, None

    def test(self):
        self.label = [self.dataset[i][1] for i in range(len(self.labels))]
        return (
            all(type(lbl) is int for lbl in self.label)
            and self.label == self.labels.tolist()
        )

    def define_failure_message(self):
        return f"The labels are incorrect (expected the integers {self.labels.tolist()}, got {self.label})."


class TransformTest(UnitTest):
    def __init__(self, root: Path, imgs: torch.Tensor) -> None:
        self.dataset = MNISTDataset(root, transform=lambda img: img.flip(-1))
        self.imgs = imgs

    def test(self):
        img, _ = self.dataset[0]
        return torch.allclose(
            img, (self.imgs[0].float().flip(-1) / 255 - MEAN) / STD, atol=1e-5
        )

    def define_failure_message(self):
        return "The transform is not applied to the image. Apply it after the normalization."


class DataLoaderTest(UnitTest):
    def __init__(self, dataset: MNISTDataset) -> None:
        self.dataset, self.shapes = dataset, None

    def test(self):
        X, y = next(iter(DataLoader(self.dataset, batch_size=4)))
        self.shapes = (tuple(X.shape), tuple(y.shape), y.dtype)
        return self.shapes == ((4, 1, 28, 28), (4,), torch.int64)

    def define_failure_message(self):
        return f"A batch of the DataLoader is incorrect (expected images (4, 1, 28, 28) and labels (4,) int64, got {self.shapes})."


class MNISTDatasetTest(CompositeTest):
    def define_tests(self, root: Path):
        imgs, labels = write_samples(root, 10)
        dataset = MNISTDataset(root)
        return [
            LengthTest(dataset, 10),
            ImageTest(dataset, imgs),
            LabelTest(dataset, labels),
            TransformTest(root, imgs),
            DataLoaderTest(dataset),
        ]


def test_mnist_dataset():
    with tempfile.TemporaryDirectory() as tmp:
        return test_results_to_score(MNISTDatasetTest(Path(tmp))())
