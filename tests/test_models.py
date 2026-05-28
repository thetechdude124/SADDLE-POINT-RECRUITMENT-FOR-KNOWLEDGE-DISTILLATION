"""Architecture sanity tests."""

import torch

from sprkd.models import (
    MalariaStudentCNN,
    MalariaTeacherCNN,
    build_legacy_sequential_student,
    build_legacy_sequential_teacher,
    count_parameters,
)


def test_teacher_param_count_matches_paper():
    """Paper Table 1 reports 25,546 parameters for the teacher CNN."""

    assert count_parameters(MalariaTeacherCNN()) == 25_546


def test_student_param_count_matches_paper():
    """Paper Table 1 reports 6,430 parameters for the student CNN."""

    assert count_parameters(MalariaStudentCNN()) == 6_430


def test_legacy_param_counts_match_oop_versions():
    assert count_parameters(build_legacy_sequential_teacher()) == count_parameters(
        MalariaTeacherCNN()
    )
    assert count_parameters(build_legacy_sequential_student()) == count_parameters(
        MalariaStudentCNN()
    )


def test_compression_ratio_is_4x():
    teacher_p = count_parameters(MalariaTeacherCNN())
    student_p = count_parameters(MalariaStudentCNN())
    assert 3.5 < teacher_p / student_p < 4.5


def test_forward_shapes():
    x = torch.randn(2, 3, 32, 32)
    assert MalariaTeacherCNN()(x).shape == (2, 2)
    assert MalariaStudentCNN()(x).shape == (2, 2)


def test_forward_outputs_are_probabilities():
    x = torch.randn(4, 3, 32, 32)
    out = MalariaTeacherCNN()(x)
    assert torch.allclose(out.sum(dim=1), torch.ones(4), atol=1e-5)
    assert (out >= 0).all() and (out <= 1).all()


def test_models_are_trainable():
    """One backward pass must produce gradients for every parameter."""

    model = MalariaStudentCNN()
    x = torch.randn(2, 3, 32, 32)
    y = torch.randint(0, 2, (2,))
    loss = torch.nn.functional.nll_loss(torch.log(model(x) + 1e-10), y)
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"{name} has no gradient"
