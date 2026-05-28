"""Tests for the train_teacher / train_student / train_control loops."""

import pytest
import torch
import torch.nn as nn

from sprkd.models import MalariaStudentCNN, MalariaTeacherCNN
from sprkd.training import (
    TrainingHistory,
    train_control,
    train_response_kd,
    train_student,
    train_teacher,
)


def test_training_history_to_dict_round_trip():
    h = TrainingHistory()
    h.train_losses = [0.5, 0.4]
    h.valid_accuracies = [50.0, 60.0]
    payload = h.to_dict()
    assert payload["TRAINING"]["LOSSES"] == [0.5, 0.4]
    assert payload["VALIDATION"]["ACCURACIES"] == [50.0, 60.0]


def test_train_control_decreases_loss(tiny_loader, cpu_loss):
    model = MalariaStudentCNN()
    history = train_control(
        model,
        tiny_loader,
        tiny_loader,
        loss_fn=cpu_loss,
        n_epochs=2,
        lr=1e-2,
        device=torch.device("cpu"),
        progress=False,
    )
    assert len(history.train_losses) == 2 * len(tiny_loader)
    assert len(history.valid_losses) == 2
    assert all(0 <= a <= 100 for a in history.valid_accuracies)


def test_train_teacher_records_saddles_with_stub(tiny_loader, cpu_loss, monkeypatch):
    model = MalariaTeacherCNN()

    class _Stub:
        def __init__(self, *a, **kw):
            pass

        def eigenvalues(self, top_n=4):
            eigs = [10.0, -9.0, 1.0, -1.0][:top_n]
            return eigs, [torch.zeros(1) for _ in range(top_n)]

    sprkd, history = train_teacher(
        model,
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
    assert len(sprkd.saddle_repository) >= 1
    assert len(history.train_losses) == len(tiny_loader)


def test_train_student_runs_one_epoch(tiny_loader, cpu_loss):
    student = MalariaStudentCNN()
    asr = [torch.zeros_like(p) for p in student.parameters()]

    sprkd, history = train_student(
        student,
        tiny_loader,
        tiny_loader,
        loss_fn=cpu_loss,
        teacher_saddle_points=asr,
        n_epochs=1,
        lr=1e-3,
        device=torch.device("cpu"),
        progress=False,
        sprkd_kwargs={"epsilon": 1.0, "max_nhe_steps": 0},
    )
    assert len(history.train_losses) == len(tiny_loader)


def test_train_response_kd_runs(tiny_loader, cpu_loss):
    student = MalariaStudentCNN()
    teacher = MalariaTeacherCNN()
    history = train_response_kd(
        student,
        teacher,
        tiny_loader,
        tiny_loader,
        n_epochs=1,
        lr=1e-3,
        device=torch.device("cpu"),
        progress=False,
    )
    assert len(history.train_losses) == len(tiny_loader)
    assert len(history.valid_losses) == 1
