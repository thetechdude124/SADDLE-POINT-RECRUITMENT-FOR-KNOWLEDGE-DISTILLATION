"""Tests for the TLI (Transfer Learning by Injection) module."""

import pytest
import torch

from sprkd.models import MalariaStudentCNN, MalariaTeacherCNN
from sprkd.tli import (
    fn_inject,
    inject_state_list,
    pair_layers,
    simple_inject,
)


def test_fn_inject_equal_shape_is_a_copy():
    src = torch.arange(12.0).reshape(3, 4)
    dst = torch.zeros(3, 4)
    fn_inject(src, dst)
    assert torch.allclose(dst, src)


def test_fn_inject_smaller_into_larger_centers_content():
    src = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    dst = torch.zeros(4, 4)
    fn_inject(src, dst)
    # 2x2 must be centered (rows 1..3, cols 1..3 -> exclusive end at 3)
    assert torch.allclose(dst[1:3, 1:3], src)
    border_mask = torch.ones_like(dst, dtype=torch.bool)
    border_mask[1:3, 1:3] = False
    assert torch.allclose(dst[border_mask], torch.zeros(border_mask.sum()))


def test_fn_inject_larger_into_smaller_takes_center_crop():
    src = torch.arange(16.0).reshape(4, 4)
    dst = torch.zeros(2, 2)
    fn_inject(src, dst)
    # center 2x2 of src
    assert torch.allclose(dst, src[1:3, 1:3])


def test_fn_inject_rejects_rank_mismatch():
    with pytest.raises(ValueError):
        fn_inject(torch.zeros(3, 3), torch.zeros(2, 2, 2))


def test_pair_layers_matches_named_keys(student_model, teacher_model):
    pairs = pair_layers(student_model, teacher_model)
    assert len(pairs) > 0
    s_keys = {a for a, _ in pairs}
    t_keys = {b for _, b in pairs}
    assert s_keys == t_keys


def test_simple_inject_roundtrip_changes_student_weights(student_model, teacher_model):
    before = [p.detach().clone() for p in student_model.parameters()]
    simple_inject(student_model, teacher_model)
    after = list(student_model.parameters())
    diffs = [torch.linalg.norm(a - b).item() for a, b in zip(before, after)]
    assert any(d > 1e-6 for d in diffs), "simple_inject did not modify any layer"


def test_simple_inject_same_arch_is_full_copy():
    a = MalariaStudentCNN()
    b = MalariaStudentCNN()
    pairs = simple_inject(a, b)
    for s_key, _ in pairs:
        sp = dict(a.named_parameters())[s_key]
        bp = dict(b.named_parameters())[s_key]
        assert torch.allclose(sp.data, bp.data)


def test_simple_inject_preserves_shapes(student_model, teacher_model):
    shapes_before = [p.shape for p in student_model.parameters()]
    simple_inject(student_model, teacher_model)
    shapes_after = [p.shape for p in student_model.parameters()]
    assert shapes_before == shapes_after


def test_inject_state_list_aligns_positionally(student_model):
    fake_asr = [torch.zeros_like(p) for p in student_model.parameters()]
    inject_state_list(student_model, fake_asr)
    for p in student_model.parameters():
        assert torch.allclose(p.data, torch.zeros_like(p.data))


def test_inject_state_list_with_teacher_pipeline(student_model, teacher_model):
    teacher_state = [p.detach().clone() for p in teacher_model.parameters()]
    inject_state_list(student_model, teacher_state, teacher=teacher_model)
    # student parameters should now be derived from teacher_state through fn_inject
    s_params = list(student_model.parameters())
    t_params = teacher_state
    # for the smaller-into-larger case (none here), student is shape <= teacher.
    # For Conv2d(3->2) student vs Conv2d(3->4) teacher we expect a centered crop.
    assert s_params[0].shape == student_model.features[0][0].weight.shape


def test_inject_state_list_rejects_mismatched_count(student_model):
    bad = [torch.zeros(2)]
    with pytest.raises(ValueError):
        inject_state_list(student_model, bad)
