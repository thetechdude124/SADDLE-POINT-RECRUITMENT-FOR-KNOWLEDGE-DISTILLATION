"""End-to-end integration tests: teacher -> ASR -> student on synthetic data."""

import torch
import torch.nn as nn

from sprkd.models import MalariaStudentCNN, MalariaTeacherCNN
from sprkd.saddle import aggregate_asr
from sprkd.tli import inject_state_list, simple_inject
from sprkd.training import train_control, train_student, train_teacher


class _Stub:
    def __init__(self, *a, **kw):
        pass

    def eigenvalues(self, top_n=4):
        eigs = [10.0, -9.0, 1.0, -1.0][:top_n]
        return eigs, [torch.zeros(1) for _ in range(top_n)]


def test_full_pipeline_on_tiny_loader(tiny_loader, cpu_loss):
    teacher = MalariaTeacherCNN()
    sprkd_t, _ = train_teacher(
        teacher,
        tiny_loader,
        tiny_loader,
        loss_fn=cpu_loss,
        n_epochs=1,
        lr=1e-3,
        saddle_steps=1,
        device=torch.device("cpu"),
        progress=False,
        sprkd_kwargs={"hessian_factory": _Stub},
    )
    assert len(sprkd_t.saddle_repository) >= 1

    asr = aggregate_asr([sprkd_t.saddle_repository.snapshots])
    assert len(asr) == sum(1 for _ in teacher.parameters())

    student = MalariaStudentCNN()
    inject_state_list(student, asr, teacher=teacher)
    targets = [p.detach().clone() for p in student.parameters()]

    sprkd_s, history = train_student(
        student,
        tiny_loader,
        tiny_loader,
        loss_fn=cpu_loss,
        teacher_saddle_points=targets,
        n_epochs=1,
        lr=1e-3,
        device=torch.device("cpu"),
        progress=False,
        sprkd_kwargs={"epsilon": 0.1, "max_nhe_steps": 0},
    )
    assert len(history.train_losses) == len(tiny_loader)


def test_control_vs_sprkd_both_run_to_completion(tiny_loader, cpu_loss):
    control = MalariaStudentCNN()
    train_control(
        control,
        tiny_loader,
        tiny_loader,
        loss_fn=cpu_loss,
        n_epochs=1,
        lr=1e-3,
        device=torch.device("cpu"),
        progress=False,
    )
    student = MalariaStudentCNN()
    targets = [p.detach().clone() for p in student.parameters()]
    train_student(
        student,
        tiny_loader,
        tiny_loader,
        loss_fn=cpu_loss,
        teacher_saddle_points=targets,
        n_epochs=1,
        lr=1e-3,
        device=torch.device("cpu"),
        progress=False,
        sprkd_kwargs={"max_nhe_steps": 0},
    )
