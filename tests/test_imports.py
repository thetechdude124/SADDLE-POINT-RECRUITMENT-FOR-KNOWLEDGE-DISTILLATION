"""Smoke tests: every public symbol must import cleanly."""

import importlib

import pytest

PUBLIC_MODULES = [
    "sprkd",
    "sprkd.optimizer",
    "sprkd.saddle",
    "sprkd.tli",
    "sprkd.training",
    "sprkd.models",
    "sprkd.architectures",
    "sprkd.data",
    "sprkd.analysis",
    "sprkd.visualize",
    "sprkd.utils",
    "sprkd.cli",
    "sprkd.legacy",
    "sprkd.stats",
    "sprkd.landscape",
    "sprkd.eval",
]


@pytest.mark.parametrize("name", PUBLIC_MODULES)
def test_module_imports(name):
    importlib.import_module(name)


def test_top_level_exports():
    import sprkd

    expected = {
        "__version__",
        "SPRKD",
        "aggregate_asr",
        "is_strong_saddle_point",
        "SaddlePointRepository",
        "fn_inject",
        "pair_layers",
        "simple_inject",
        "MalariaStudentCNN",
        "MalariaTeacherCNN",
        "BasicBlock",
        "ResNetCIFAR",
        "build_resnet20",
        "build_resnet32",
        "build_resnet44",
        "build_resnet56",
        "get_device",
        "set_seed",
        "enable_legacy_unpickling",
        "load_legacy_checkpoint",
        "load_legacy_student",
        "load_legacy_teacher",
        "load_legacy_metrics_pkl",
        "epoch_validation_series",
        "McNemarResult",
        "mcnemar_table",
        "mcnemar_test",
        "pairwise_mcnemar",
        "TrialResult",
        "collect_predictions",
        "evaluate_on_testset",
        "evaluate_performance_trials",
    }
    assert expected.issubset(set(sprkd.__all__))
    for name in expected:
        assert hasattr(sprkd, name), name


def test_version_string_is_pep440():
    import re

    import sprkd

    assert re.match(r"^\d+\.\d+(\.\d+)?", sprkd.__version__) is not None
