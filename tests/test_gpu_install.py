"""The GPU extra must resolve on Windows without unavailable GPU FAISS wheels."""

from __future__ import annotations

from importlib.metadata import requires

import pytest
from packaging.markers import default_environment
from packaging.requirements import Requirement


@pytest.mark.parametrize(
    ("system", "machine", "expected_faiss"),
    [
        ("win32", "AMD64", "faiss-cpu"),
        ("linux", "x86_64", "faiss-gpu"),
        ("linux", "aarch64", "faiss-cpu"),
    ],
)
def test_gpu_extra_resolves_one_supported_faiss_backend(
    system: str,
    machine: str,
    expected_faiss: str,
) -> None:
    environment = {
        **default_environment(),
        "sys_platform": system,
        "platform_machine": machine,
        "extra": "gpu",
    }
    dependencies = [Requirement(value) for value in requires("verascan") or []]
    selected = {
        dependency.name
        for dependency in dependencies
        if dependency.marker is None or dependency.marker.evaluate(environment)
    }
    assert selected & {"faiss-cpu", "faiss-gpu"} == {expected_faiss}
    assert {"torch", "sentence-transformers", "datasets"} <= selected


@pytest.mark.parametrize(
    ("python_version", "expected"), [("3.9", False), ("3.10", False), ("3.11", True)]
)
def test_dev_extra_skips_dashboard_on_unsupported_python(
    python_version: str, expected: bool
) -> None:
    environment = {**default_environment(), "python_version": python_version, "extra": "dev"}
    dashboard_deps = [
        Requirement(value)
        for value in requires("verascan") or []
        if Requirement(value).name == "streamlit"
    ]
    selected = any(
        dependency.marker is None or dependency.marker.evaluate(environment)
        for dependency in dashboard_deps
    )
    assert selected is expected
