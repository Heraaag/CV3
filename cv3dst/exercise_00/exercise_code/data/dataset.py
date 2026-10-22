from pathlib import Path

import torch
import torchvision
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset

# mean and standard deviation of the MNIST training images in [0, 1]
MEAN, STD = 0.1307, 0.3081


def export_mnist_png(root: Path, png_root: Path) -> Path:
    """Loads MNIST from root/MNIST (downloads it if missing) and writes one PNG per image and a labels.csv per split
    to png_root/MNIST_png/{train,test}."""
    out = png_root / "MNIST_png"
    for split in ("train", "test"):
        folder = out / split
        if (folder / "labels.csv").exists():  # written last, so the export is complete
            continue
        folder.mkdir(parents=True, exist_ok=True)
        mnist = torchvision.datasets.MNIST(
            root=root, train=split == "train", download=True
        )
        rows = []
        for i, (img, label) in enumerate(zip(mnist.data, mnist.targets)):
            name = f"{i:05d}.png"
            Image.fromarray(img.numpy()).save(folder / name)
            rows.append(f"{name},{label}")
        (folder / "labels.csv").write_text("\n".join(rows))
    return out


class MNISTDataset(Dataset):
    def __init__(self, root: Path, transform=None):
        # only read the file names and labels, not the images
        self.root = root
        self.transform = transform
        rows = [
            line.split(",") for line in (root / "labels.csv").read_text().splitlines()
        ]
        self.files = [name for name, _ in rows]
        self.labels = [int(label) for _, label in rows]

    def __len__(self) -> int:
        length = None

        ########################################################################
        # TODO:                                                                #
        # Return the number of samples in the dataset.                         #
        # length = ...                                                         #
        ########################################################################


        pass

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

        return length

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        img, label = None, None

        ########################################################################
        # TODO:                                                                #
        # Load the sample at position index from disk:                         #
        # 1. read the PNG file self.root / self.files[index] into a uint8      #
        #    tensor of shape (1, 28, 28): open it with PIL's Image.open and   #
        #    convert it with torchvision.transforms.functional.pil_to_tensor  #
        # 2. convert it to float32 with values in [0, 1]                       #
        # 3. normalize it with (img - MEAN) / STD                              #
        # 4. apply self.transform to the image if it is not None               #
        # The label is the integer self.labels[index].                         #
        # img = ...                                                            #
        # label = ...                                                          #
        ########################################################################


        pass

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

        return img, label
