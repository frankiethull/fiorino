"""Pytest config for fiorino: --checkpoint (local file or HF repo id),
--device, --cls-only (skip regressor test for classification-only artifacts)."""
import pytest


def pytest_addoption(parser):
    parser.addoption("--checkpoint", action="store", default=None,
                     help="Local weights file (.pt/.safetensors) or HF repo id")
    parser.addoption("--device", action="store", default="cpu")
    parser.addoption("--cls-only", action="store_true",
                     help="Artifact is classification-only: skip regressor test")


@pytest.fixture
def checkpoint(request):
    ckpt = request.config.getoption("checkpoint")
    if ckpt is None:
        pytest.skip("need --checkpoint (local file or HF repo id)")
    return ckpt


@pytest.fixture
def device(request):
    return request.config.getoption("device")


@pytest.fixture
def cls_only_artifact(request):
    return request.config.getoption("cls_only")
