"""Tests for the malaria data loader."""

from pathlib import Path

import pytest
import torch
from PIL import Image

from sprkd.data import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_TRAIN_FRACTION,
    MalariaDataConfig,
    find_default_root,
    make_dataloaders,
    split_train_valid,
)


def _make_synthetic_dataset(root: Path, n_per_class: int = 4):
    """Create a tiny synthetic ``ImageFolder`` dataset for fast unit tests."""

    for cls in ("Parasitized", "Uninfected"):
        d = root / cls
        d.mkdir(parents=True, exist_ok=True)
        for i in range(n_per_class):
            arr = torch.randint(0, 255, (32, 32, 3), dtype=torch.uint8).numpy()
            Image.fromarray(arr).save(d / f"img_{i}.png")


def test_find_default_root_returns_repo_path(repo_root):
    p = find_default_root()
    assert p == repo_root / "cell_images"


def test_data_config_defaults():
    cfg = MalariaDataConfig(root=Path("/nonexistent"))
    assert cfg.image_size == DEFAULT_IMAGE_SIZE
    assert cfg.batch_size == DEFAULT_BATCH_SIZE
    assert cfg.train_fraction == DEFAULT_TRAIN_FRACTION


def test_missing_dataset_raises_clear_error(tmp_path):
    cfg = MalariaDataConfig(root=tmp_path / "missing")
    with pytest.raises(FileNotFoundError, match="Malaria dataset directory not found"):
        from sprkd.data import load_malaria_dataset

        load_malaria_dataset(cfg)


def test_split_train_valid_is_deterministic(tmp_path):
    _make_synthetic_dataset(tmp_path, n_per_class=8)
    cfg = MalariaDataConfig(root=tmp_path, batch_size=2, num_workers=0, pin_memory=False)
    train_loader, valid_loader, full = make_dataloaders(cfg)

    train_set_a, valid_set_a = split_train_valid(full, train_fraction=0.75, seed=42)
    train_set_b, valid_set_b = split_train_valid(full, train_fraction=0.75, seed=42)
    assert list(train_set_a.indices) == list(train_set_b.indices)
    assert list(valid_set_a.indices) == list(valid_set_b.indices)


def test_dataloader_shapes(tmp_path):
    _make_synthetic_dataset(tmp_path, n_per_class=8)
    cfg = MalariaDataConfig(root=tmp_path, batch_size=4, num_workers=0, pin_memory=False)
    train_loader, valid_loader, full = make_dataloaders(cfg)

    x, y = next(iter(train_loader))
    assert x.shape == (4, 3, DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE)
    assert y.shape == (4,)
    assert ((y == 0) | (y == 1)).all()


def test_real_dataset_loads_when_available(repo_root):
    root = repo_root / "cell_images"
    if not (root / "Parasitized").is_dir():
        pytest.skip("Full malaria dataset not present locally")
    cfg = MalariaDataConfig(root=root, batch_size=8, num_workers=0, pin_memory=False)
    train_loader, valid_loader, full = make_dataloaders(cfg)
    # paper splits 27,558 cells 75/25
    assert 27_000 <= len(full) <= 28_000
    x, y = next(iter(train_loader))
    assert x.shape[1:] == (3, 32, 32)


# ---------------------------------------------------------------------------
# CIFAR-100 / MNIST loaders (paper Section 4.2)
# ---------------------------------------------------------------------------

def test_cifar100_config_defaults():
    from sprkd.data import CIFAR100Config

    cfg = CIFAR100Config(root=Path("/nonexistent"))
    assert cfg.image_size == 32
    assert cfg.batch_size == 64
    assert cfg.download is True


def test_mnist_config_defaults():
    from sprkd.data import MNISTConfig

    cfg = MNISTConfig(root=Path("/nonexistent"))
    assert cfg.image_size == 28
    assert cfg.batch_size == 64


def test_cifar100_dataloaders_smoke(tmp_path):
    """Build the loaders without actually downloading - guard with download=False."""

    from sprkd.data import CIFAR100Config, make_cifar100_dataloaders

    cfg = CIFAR100Config(
        root=tmp_path,
        batch_size=4,
        num_workers=0,
        pin_memory=False,
        download=False,
    )
    # We do not require the dataset to be present locally; the loader should
    # raise an informative error when CIFAR-100 is not on disk.
    with pytest.raises((FileNotFoundError, RuntimeError)):
        make_cifar100_dataloaders(cfg)


def test_mnist_dataloaders_smoke(tmp_path):
    from sprkd.data import MNISTConfig, make_mnist_dataloaders

    cfg = MNISTConfig(
        root=tmp_path,
        batch_size=4,
        num_workers=0,
        pin_memory=False,
        download=False,
    )
    with pytest.raises((FileNotFoundError, RuntimeError)):
        make_mnist_dataloaders(cfg)


# ---------------------------------------------------------------------------
# TESTSET.pth loader
# ---------------------------------------------------------------------------

@pytest.mark.checkpoints
def test_load_testset_pth_returns_xy_pair(repo_root):
    from sprkd.data import load_testset_pth

    p = repo_root / "TESTSET.pth"
    if not p.is_file():
        pytest.skip("TESTSET.pth not present")
    if open(p, "rb").read(64).startswith(b"version https://git-lfs.github.com/spec/"):
        pytest.skip("TESTSET.pth is an LFS pointer; run `git lfs pull`")
    xs, ys = load_testset_pth(p)
    assert xs.ndim == 4
    assert xs.shape[1:] == (3, 32, 32)
    assert ys.shape[0] == xs.shape[0]
    assert ((ys == 0) | (ys == 1)).all()


def test_load_testset_pth_rejects_invalid_format(tmp_path):
    """Save a malformed pickle and confirm the loader raises a clear error."""

    import torch

    bad = tmp_path / "bad_testset.pth"
    torch.save({"not": "a tuple"}, bad)
    from sprkd.data import load_testset_pth

    with pytest.raises(ValueError, match="2-tuple"):
        load_testset_pth(bad)


# ---------------------------------------------------------------------------
# TinyImageNet val-folder reorganization
# ---------------------------------------------------------------------------

def _stage_tinyimagenet_val(root: Path, n_classes: int = 3, per_class: int = 2):
    """Build a tiny mock ``val/`` directory in the canonical (flat) layout."""

    import torch
    from PIL import Image

    val_dir = root / "val"
    images_dir = val_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    annotations: list[tuple[str, str]] = []
    for c in range(n_classes):
        wnid = f"n{c:08d}"
        for i in range(per_class):
            fname = f"val_{c}_{i}.JPEG"
            arr = torch.randint(0, 255, (32, 32, 3), dtype=torch.uint8).numpy()
            Image.fromarray(arr).save(images_dir / fname)
            annotations.append((fname, wnid))
    with open(val_dir / "val_annotations.txt", "w") as f:
        for fname, wnid in annotations:
            f.write(f"{fname}\t{wnid}\t0\t0\t32\t32\n")
    return val_dir, annotations


def test_reorganize_tinyimagenet_val_moves_files_into_class_folders(tmp_path):
    from sprkd.data import reorganize_tinyimagenet_val

    val_dir, annotations = _stage_tinyimagenet_val(tmp_path, n_classes=3, per_class=2)
    moved = reorganize_tinyimagenet_val(val_dir)
    assert moved == 6
    for fname, wnid in annotations:
        assert (val_dir / "images" / wnid / fname).is_file()
    # Second call should be idempotent (files already in their class folders).
    moved2 = reorganize_tinyimagenet_val(val_dir)
    assert moved2 == 0


def test_reorganize_tinyimagenet_val_missing_directory_errors(tmp_path):
    from sprkd.data import reorganize_tinyimagenet_val

    with pytest.raises(FileNotFoundError):
        reorganize_tinyimagenet_val(tmp_path / "no-such-dir")


def test_reorganize_tinyimagenet_val_missing_annotations(tmp_path):
    from sprkd.data import reorganize_tinyimagenet_val

    (tmp_path / "val" / "images").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="val_annotations"):
        reorganize_tinyimagenet_val(tmp_path / "val")


def test_make_tinyimagenet_dataloaders_smoke(tmp_path):
    """Build the loaders end-to-end on a tiny mock dataset."""

    import torch
    from PIL import Image

    from sprkd.data import TinyImageNetConfig, make_tinyimagenet_dataloaders

    train_dir = tmp_path / "train"
    for c in range(3):
        d = train_dir / f"n{c:08d}" / "images"
        d.mkdir(parents=True, exist_ok=True)
        for i in range(2):
            arr = torch.randint(0, 255, (32, 32, 3), dtype=torch.uint8).numpy()
            Image.fromarray(arr).save(d / f"img_{i}.JPEG")
    # ImageFolder expects images directly under the class folder, not nested
    # ``images/`` subfolder. Move them up.
    for cls_dir in train_dir.iterdir():
        for img in (cls_dir / "images").iterdir():
            img.rename(cls_dir / img.name)
        (cls_dir / "images").rmdir()

    _stage_tinyimagenet_val(tmp_path, n_classes=3, per_class=2)

    cfg = TinyImageNetConfig(
        root=tmp_path,
        image_size=32,
        batch_size=2,
        num_workers=0,
        pin_memory=False,
    )
    train_loader, valid_loader, test_loader = make_tinyimagenet_dataloaders(cfg)
    x, y = next(iter(train_loader))
    assert x.shape == (2, 3, 32, 32)
    x, y = next(iter(valid_loader))
    assert x.shape == (2, 3, 32, 32)
    x, y = next(iter(test_loader))
    assert x.shape == (1, 3, 32, 32)


def test_tinyimagenet_constants_match_canonical_colab():
    """ImageNet-derived per-channel statistics must match the Colab notebook."""

    from sprkd.data import TINYIMAGENET_MEAN, TINYIMAGENET_STD

    assert TINYIMAGENET_MEAN == pytest.approx(
        (122.4602 / 255.0, 114.2571 / 255.0, 101.3639 / 255.0)
    )
    assert TINYIMAGENET_STD == pytest.approx(
        (70.4915 / 255.0, 68.5601 / 255.0, 71.8054 / 255.0)
    )
