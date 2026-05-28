"""Dataset utilities for the SPRKD malaria experiment.

The malaria blood-smear dataset is the public NIH/NLM "cell_images" dataset
(`<https://lhncbc.nlm.nih.gov/LHC-publications/pubs/MalariaDatasets.html>`_):
27,558 annotated PNG images, balanced 50/50 between ``Parasitized`` and
``Uninfected`` classes.

By default the loader expects the dataset extracted under
``<repo_root>/cell_images/{Parasitized,Uninfected}/*.png``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import datasets, transforms


DEFAULT_IMAGE_SIZE = 32
DEFAULT_BATCH_SIZE = 64
DEFAULT_TRAIN_FRACTION = 0.75


@dataclass
class MalariaDataConfig:
    root: Path
    image_size: int = DEFAULT_IMAGE_SIZE
    batch_size: int = DEFAULT_BATCH_SIZE
    train_fraction: float = DEFAULT_TRAIN_FRACTION
    num_workers: int = 2
    pin_memory: bool = True
    seed: int = 0


def _build_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )


def load_malaria_dataset(config: MalariaDataConfig) -> datasets.ImageFolder:
    """Return the raw ``torchvision.datasets.ImageFolder`` for the malaria data."""

    if not Path(config.root).is_dir():
        raise FileNotFoundError(
            f"Malaria dataset directory not found: {config.root}\n"
            "Expected layout:\n"
            f"    {config.root}/Parasitized/*.png\n"
            f"    {config.root}/Uninfected/*.png\n"
            "See data/README.md for download instructions."
        )

    transform = _build_transform(config.image_size)
    return datasets.ImageFolder(str(config.root), transform=transform)


def split_train_valid(
    dataset: Dataset,
    train_fraction: float = DEFAULT_TRAIN_FRACTION,
    seed: int = 0,
) -> Tuple[Subset, Subset]:
    """Random split with a deterministic seed."""

    n = len(dataset)
    n_train = int(round(n * train_fraction))
    n_valid = n - n_train
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [n_train, n_valid], generator=generator)


def make_dataloaders(
    config: MalariaDataConfig,
) -> Tuple[DataLoader, DataLoader, datasets.ImageFolder]:
    """Build (train_loader, valid_loader, full_dataset) for the malaria split."""

    full = load_malaria_dataset(config)
    train_set, valid_set = split_train_valid(
        full, train_fraction=config.train_fraction, seed=config.seed
    )

    common = dict(
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )
    train_loader = DataLoader(train_set, shuffle=True, **common)
    valid_loader = DataLoader(valid_set, shuffle=False, **common)
    return train_loader, valid_loader, full


def find_default_root(repo_root: Optional[Path] = None) -> Path:
    """Best-effort lookup of the bundled ``cell_images`` directory."""

    if repo_root is None:
        repo_root = Path(__file__).resolve().parent.parent
    candidate = repo_root / "cell_images"
    return candidate


# ---------------------------------------------------------------------------
# CIFAR-100 / MNIST loaders for the self-distillation experiments referenced
# in the paper's Section 4.2 (and the Class-1 alpha notebooks).
# ---------------------------------------------------------------------------

@dataclass
class CIFAR100Config:
    root: Path
    image_size: int = 32
    batch_size: int = 64
    num_workers: int = 2
    pin_memory: bool = True
    seed: int = 0
    download: bool = True


@dataclass
class MNISTConfig:
    root: Path
    image_size: int = 28
    batch_size: int = 64
    num_workers: int = 2
    pin_memory: bool = True
    seed: int = 0
    download: bool = True


def make_cifar100_dataloaders(config: CIFAR100Config) -> Tuple[DataLoader, DataLoader]:
    """CIFAR-100 train + validation DataLoaders.

    Used for the self-distillation experiments in the SPRKD paper Section 4.2
    and in the Class-1 alpha notebook
    ``CIFAR-100_SPRKD_EXPERIMENTATION_SELF_DISTILLATION.ipynb``.
    """

    transform = transforms.Compose(
        [
            transforms.Resize((config.image_size, config.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2762]),
        ]
    )
    train = datasets.CIFAR100(
        root=str(config.root), train=True, transform=transform, download=config.download
    )
    valid = datasets.CIFAR100(
        root=str(config.root), train=False, transform=transform, download=config.download
    )
    common = dict(
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )
    return (
        DataLoader(train, shuffle=True, **common),
        DataLoader(valid, shuffle=False, **common),
    )


def make_mnist_dataloaders(config: MNISTConfig) -> Tuple[DataLoader, DataLoader]:
    """MNIST train + validation DataLoaders.

    Used for self-distillation in the SPRKD paper Section 4.2.
    """

    transform = transforms.Compose(
        [
            transforms.Resize((config.image_size, config.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.1307], std=[0.3081]),
        ]
    )
    train = datasets.MNIST(
        root=str(config.root), train=True, transform=transform, download=config.download
    )
    valid = datasets.MNIST(
        root=str(config.root), train=False, transform=transform, download=config.download
    )
    common = dict(
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )
    return (
        DataLoader(train, shuffle=True, **common),
        DataLoader(valid, shuffle=False, **common),
    )


def load_testset_pth(
    path: Path | str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load the bundled ``TESTSET.pth`` (a 64-sample held-out batch).

    Returns
    -------
    (xs, ys)
        ``xs`` of shape ``(N, 3, 32, 32)``, ``ys`` of shape ``(N,)``.
    """

    obj = torch.load(str(path), map_location="cpu", weights_only=False)
    if not (isinstance(obj, (list, tuple)) and len(obj) == 2):
        raise ValueError(
            f"Expected TESTSET.pth to be a 2-tuple (inputs, targets); got "
            f"{type(obj).__name__} with len={len(obj) if hasattr(obj, '__len__') else 'N/A'}."
        )
    xs, ys = obj[0], obj[1]
    return xs, ys


# ---------------------------------------------------------------------------
# TinyImageNet loader (paper Experiment 2 / EXPERIMENTAL_MODEL_EVALUATIONS Colab)
# ---------------------------------------------------------------------------

# ImageNet-derived per-channel statistics (matches canonical Colab):
TINYIMAGENET_MEAN = (122.4602 / 255.0, 114.2571 / 255.0, 101.3639 / 255.0)
TINYIMAGENET_STD = (70.4915 / 255.0, 68.5601 / 255.0, 71.8054 / 255.0)


@dataclass
class TinyImageNetConfig:
    """Configuration for the TinyImageNet (Le & Yang, 2015) data loaders."""

    root: Path
    image_size: int = 64
    batch_size: int = 64
    num_workers: int = 12
    pin_memory: bool = True
    seed: int = 0


def reorganize_tinyimagenet_val(val_dir: Path) -> int:
    """Move TinyImageNet validation files into per-class subfolders.

    The TinyImageNet ``val/`` distribution ships as a flat ``val/images/``
    directory plus a ``val/val_annotations.txt`` lookup; PyTorch's
    ``ImageFolder`` cannot read it as-is. This helper moves each image into
    ``val/images/<class>/`` based on the annotations file, mirroring the
    cell

    .. code-block:: python

        # canonical Colab (EXPERIMENTAL_MODEL_EVALUATIONS)
        for image_file, image_class_folder in valid_classes_lookup.items():
            new_class_path = os.path.join(VAL_IMAGE_DIRECT, image_class_folder)
            os.makedirs(new_class_path, exist_ok=True)
            os.rename(...)

    Returns the number of files actually moved (``0`` if the directory
    has already been reorganized).
    """

    val_dir = Path(val_dir)
    images_dir = val_dir / "images"
    annotations = val_dir / "val_annotations.txt"
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Missing TinyImageNet val/images directory: {images_dir}")
    if not annotations.is_file():
        raise FileNotFoundError(f"Missing val_annotations.txt at {annotations}")

    lookup: dict[str, str] = {}
    with open(annotations, "r") as f:
        for line in f:
            cols = line.split("\t")
            if len(cols) < 2:
                continue
            lookup[cols[0]] = cols[1]

    moved = 0
    for fname, cls in lookup.items():
        cls_dir = images_dir / cls
        cls_dir.mkdir(parents=True, exist_ok=True)
        src = images_dir / fname
        if src.is_file():
            dst = cls_dir / fname
            src.rename(dst)
            moved += 1
    return moved


def make_tinyimagenet_dataloaders(
    config: TinyImageNetConfig,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build (train, valid, test) DataLoaders for TinyImageNet.

    Expects the dataset extracted at ``config.root`` with the structure
    distributed by Stanford CS231n:

    ::

        <root>/train/<wnid>/images/*.JPEG
        <root>/val/images/*.JPEG
        <root>/val/val_annotations.txt
        <root>/test/images/*.JPEG

    The ``val/`` folder is reorganised in-place on first use (idempotent).
    The third returned loader is a single-sample loader over the validation
    split, used by :func:`sprkd.eval.evaluate_performance_trials` to mirror
    the per-image trial averaging in the canonical Colab notebook.
    """

    root = Path(config.root)
    if not root.is_dir():
        raise FileNotFoundError(
            f"TinyImageNet root not found: {root}\n"
            "Download from https://image-net.org/data/tiny-imagenet-200.zip "
            "and unzip into this location."
        )

    val_images_dir = root / "val" / "images"
    if val_images_dir.is_dir() and any(p.is_file() for p in val_images_dir.iterdir()):
        # Heuristic: if there are direct image files at the top level,
        # the reorganisation has not been performed yet.
        reorganize_tinyimagenet_val(root / "val")

    train_transform = transforms.Compose(
        [
            transforms.Lambda(lambda x: x.convert("RGB")),
            transforms.RandomCrop(
                [config.image_size, config.image_size], padding=8
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=list(TINYIMAGENET_MEAN), std=list(TINYIMAGENET_STD)),
        ]
    )
    valid_transform = transforms.Compose(
        [
            transforms.Lambda(lambda x: x.convert("RGB")),
            transforms.ToTensor(),
            transforms.Resize(
                (config.image_size, config.image_size), antialias=True
            ),
        ]
    )

    train_set = datasets.ImageFolder(str(root / "train"), transform=train_transform)
    valid_set = datasets.ImageFolder(str(val_images_dir), transform=valid_transform)

    common = dict(num_workers=config.num_workers, pin_memory=config.pin_memory)
    train_loader = DataLoader(
        train_set, batch_size=config.batch_size, shuffle=True, **common
    )
    valid_loader = DataLoader(
        valid_set, batch_size=config.batch_size, shuffle=False, **common
    )
    test_loader = DataLoader(
        valid_set, batch_size=1, shuffle=True, **common
    )
    return train_loader, valid_loader, test_loader
