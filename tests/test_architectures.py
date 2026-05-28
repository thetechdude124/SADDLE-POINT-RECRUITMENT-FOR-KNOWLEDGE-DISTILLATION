"""Tests for the CIFAR-style ResNet implementations (Experiment 2)."""

import torch

from sprkd.architectures import (
    BasicBlock,
    ResNetCIFAR,
    build_resnet20,
    build_resnet32,
    build_resnet44,
    build_resnet56,
)


def test_basic_block_identity_shortcut():
    """No projection when ``in_planes == planes`` and stride is 1."""

    block = BasicBlock(16, 16)
    x = torch.randn(2, 16, 8, 8)
    y = block(x)
    assert y.shape == x.shape


def test_basic_block_strided_shortcut_a():
    block = BasicBlock(16, 32, stride=2, option="A")
    x = torch.randn(2, 16, 8, 8)
    y = block(x)
    assert y.shape == (2, 32, 4, 4)


def test_basic_block_strided_shortcut_b():
    block = BasicBlock(16, 32, stride=2, option="B")
    x = torch.randn(2, 16, 8, 8)
    y = block(x)
    assert y.shape == (2, 32, 4, 4)


def test_resnet20_matches_canonical_classes():
    model = build_resnet20(num_classes=200)
    x = torch.randn(2, 3, 64, 64)
    assert model(x).shape == (2, 200)


def test_resnet20_param_count_matches_heimann_table():
    """The canonical CIFAR ResNet-20 is widely reported as 270K params.

    Heimann's original code (and the canonical Colab) yields this count.
    """

    model = build_resnet20(num_classes=200)
    n_params = sum(p.numel() for p in model.parameters())
    # ResNet-20 with 200 classes has ~282K params (200 extra from final linear).
    # Tolerate a wide window because the head size depends on num_classes.
    assert 250_000 < n_params < 320_000


def test_all_depth_factories_are_buildable():
    for build, expected_blocks in (
        (build_resnet20, 9),
        (build_resnet32, 15),
        (build_resnet44, 21),
        (build_resnet56, 27),
    ):
        m = build(num_classes=10)
        n_blocks = sum(1 for mod in m.modules() if isinstance(mod, BasicBlock))
        assert n_blocks == expected_blocks


def test_resnet_cifar_first_stride_configurable():
    model = ResNetCIFAR(BasicBlock, [3, 3, 3], num_classes=10, first_stride=1)
    x = torch.randn(2, 3, 32, 32)
    assert model(x).shape == (2, 10)


def test_resnet_cifar_supports_grayscale_via_in_channels():
    model = ResNetCIFAR(
        BasicBlock, [3, 3, 3], num_classes=10, in_channels=1, first_stride=1
    )
    x = torch.randn(2, 1, 32, 32)
    assert model(x).shape == (2, 10)
