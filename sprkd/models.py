"""Reference architectures used in the SPRKD paper.

The malaria CNNs follow Fuhad et al. (2020) and the SPRKD paper, Section 4.1.

* :class:`MalariaTeacherCNN` - 25,546 parameter teacher network.
* :class:`MalariaStudentCNN` - 6,430 parameter student network (4x compression).
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _conv_block(in_ch: int, out_ch: int) -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=(3, 3)),
        nn.ReLU(inplace=False),
    )


class MalariaTeacherCNN(nn.Module):
    """Teacher CNN used in Experiment 1 of the SPRKD paper.

    Architecture (input ``3 x 32 x 32``)::

        Conv2d(3 -> 4, 3x3) -> ReLU
        Conv2d(4 -> 8, 3x3) -> ReLU
        MaxPool2d(2x2) -> Dropout(0.1)
        Flatten -> Linear(1568 -> 16) -> ReLU -> Dropout(0.2)
        Linear(16 -> num_classes) -> Softmax(dim=1)

    Total parameters: ``25,546`` (for ``num_classes=2``).
    """

    def __init__(self, num_classes: int = 2, in_channels: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            _conv_block(in_channels, 4),
            _conv_block(4, 8),
            nn.MaxPool2d((2, 2)),
            nn.Dropout(0.1),
        )
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(1568, 16),
            nn.ReLU(inplace=False),
            nn.Dropout(0.2),
            nn.Linear(16, num_classes),
            nn.Softmax(dim=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.flatten(x)
        return self.classifier(x)


class MalariaStudentCNN(nn.Module):
    """Student CNN used in Experiment 1 (4x compression of the teacher).

    Architecture (input ``3 x 32 x 32``)::

        Conv2d(3 -> 2, 3x3) -> ReLU
        Conv2d(2 -> 4, 3x3) -> ReLU
        MaxPool2d(2x2) -> Dropout(0.1)
        Flatten -> Linear(784 -> 8) -> ReLU -> Dropout(0.2)
        Linear(8 -> num_classes) -> Softmax(dim=1)

    Total parameters: ``6,430`` (for ``num_classes=2``).
    """

    def __init__(self, num_classes: int = 2, in_channels: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            _conv_block(in_channels, 2),
            _conv_block(2, 4),
            nn.MaxPool2d((2, 2)),
            nn.Dropout(0.1),
        )
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(784, 8),
            nn.ReLU(inplace=False),
            nn.Dropout(0.2),
            nn.Linear(8, num_classes),
            nn.Softmax(dim=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.flatten(x)
        return self.classifier(x)


def build_legacy_sequential_teacher(num_classes: int = 2) -> nn.Sequential:
    """Reproduce the exact ``nn.Sequential`` from the original SPRKD notebook.

    Useful for loading ``TRUE_TEACHER_1_MALARIA.pth`` (which was saved with
    the legacy structure-free Sequential layout).
    """

    return nn.Sequential(
        nn.Conv2d(3, 4, kernel_size=(3, 3)),
        nn.ReLU(),
        nn.Conv2d(4, 8, kernel_size=(3, 3)),
        nn.ReLU(),
        nn.MaxPool2d((2, 2)),
        nn.Dropout(0.1),
        nn.Flatten(),
        nn.Linear(1568, 16),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(16, num_classes),
        nn.Softmax(dim=1),
    )


def build_legacy_sequential_student(num_classes: int = 2) -> nn.Sequential:
    """Reproduce the exact ``nn.Sequential`` student from the original notebook."""

    return nn.Sequential(
        nn.Conv2d(3, 2, kernel_size=(3, 3)),
        nn.ReLU(),
        nn.Conv2d(2, 4, kernel_size=(3, 3)),
        nn.ReLU(),
        nn.MaxPool2d((2, 2)),
        nn.Dropout(0.1),
        nn.Flatten(),
        nn.Linear(784, 8),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(8, num_classes),
        nn.Softmax(dim=1),
    )


def count_parameters(model: nn.Module) -> int:
    """Return the total number of (trainable + non-trainable) parameters."""

    return sum(p.numel() for p in model.parameters())
