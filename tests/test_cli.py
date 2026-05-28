"""Tests for the ``sprkd`` CLI."""

import io
import json
import sys

import pytest

from sprkd.cli import main


def test_info_subcommand(capsys):
    rc = main(["info"])
    assert rc == 0
    captured = capsys.readouterr().out
    payload = json.loads(captured)
    assert payload["models"]["MalariaTeacherCNN"] == 25_546
    assert payload["models"]["MalariaStudentCNN"] == 6_430


def test_help_runs():
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0


def test_version_runs(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert out.strip()  # something was printed


def test_unknown_subcommand_errors():
    with pytest.raises(SystemExit):
        main(["does-not-exist"])
