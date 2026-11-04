"""GPU audit contracts, without requiring a GPU or downloading models."""

from __future__ import annotations

import sys
from importlib.machinery import ModuleSpec
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import verascan
from verascan.benchmarks import SYNTHETIC_BENCHMARKS


@pytest.fixture()
def gpu_deps(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    faiss = pytest.importorskip("faiss")
    cuda = SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 2,
        current_device=lambda: 0,
        get_device_name=lambda index: f"NVIDIA Test {index}",
        mem_get_info=lambda index: (4 * 1024**3, 8 * 1024**3),
    )
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            cuda=cuda,
            version=SimpleNamespace(cuda="12.6", hip=None),
            __spec__=ModuleSpec("torch", loader=None),
        ),
    )
    state = SimpleNamespace(devices=[], batches=[], fail_encode=False, gpu_devices=[])
    target = SYNTHETIC_BENCHMARKS["gsm8k"][0]["text"]

    class Model:
        def __init__(self, name: str, *, device: str) -> None:
            state.devices.append(device)

        def encode(self, texts: list[str], **kwargs: Any) -> Any:
            if state.fail_encode:
                raise RuntimeError("CUDA out of memory")
            assert kwargs["normalize_embeddings"] is True
            state.batches.append(list(texts))
            return np.array([[1.0, 0.0] if t == target else [0.0, 1.0] for t in texts])

    monkeypatch.setitem(
        sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=Model)
    )
    # Simulate a working GPU index with real FAISS search semantics.
    monkeypatch.setattr(faiss, "StandardGpuResources", lambda: object(), raising=False)

    def transfer(resources: Any, device: int, index: Any) -> Any:
        state.gpu_devices.append(device)
        return index

    monkeypatch.setattr(faiss, "index_cpu_to_gpu", transfer, raising=False)
    state.faiss = faiss
    state.cuda = cuda
    state.target = target
    return state


def test_gpu_audit_cuda_and_correct_matches(gpu_deps: SimpleNamespace) -> None:
    report = verascan.gpu_audit(
        ["unrelated", gpu_deps.target],
        benchmarks="gsm8k",
        synthetic=True,
        device="cuda:1",
        batch_size=1,
        max_eval=1,
        show_progress=False,
    )
    assert isinstance(report, verascan.AuditReport)
    assert report.methods_used == ["semantic"]
    assert report.train_size == 2 and report.eval_size == 1
    assert [(m.train_index, m.eval_index, m.score) for m in report.matches] == [(1, 0, 1.0)]
    assert report.benchmark_counts == {"gsm8k": 1}
    assert gpu_deps.devices == ["cuda:1"]
    assert gpu_deps.gpu_devices == [1]
    assert all(len(batch) <= 1 for batch in gpu_deps.batches)


@pytest.mark.parametrize("failure", ["missing", "create", "add", "search"])
def test_gpu_index_failure_keeps_cuda_embeddings_and_results(
    gpu_deps: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    failure: str,
) -> None:
    faiss = gpu_deps.faiss
    if failure == "missing":
        monkeypatch.delattr(faiss, "StandardGpuResources")
    else:

        class BrokenIndex:
            def __init__(self, index: Any) -> None:
                self.index = index
                self.searches = 0

            def add(self, vectors: Any) -> None:
                if failure == "add":
                    raise RuntimeError("GPU add failed")
                self.index.add(vectors)

            def search(self, vectors: Any, k: int) -> Any:
                self.searches += 1
                if failure == "search" and self.searches == 2:
                    raise RuntimeError("GPU search failed")
                return self.index.search(vectors, k)

        def transfer(*args: Any) -> Any:
            if failure == "create":
                raise RuntimeError("GPU allocation failed")
            return BrokenIndex(args[2])

        monkeypatch.setattr(faiss, "index_cpu_to_gpu", transfer)
    report = verascan.gpu_audit(
        [gpu_deps.target],
        "gsm8k",
        synthetic=True,
        batch_size=1,
        max_eval=3,
        threshold=0.9,
        show_progress=False,
    )
    assert [(m.eval_index, m.train_index) for m in report.matches] == [(0, 0)]
    assert gpu_deps.devices == ["cuda"]
    if failure == "missing":
        assert caplog.messages == [
            "FAISS-GPU not available (installed faiss is CPU). Embeddings still on CUDA."
        ]
        assert len(gpu_deps.batches) == 4  # One train batch, three eval batches; no retry.
    else:
        assert "FAISS-CPU" in caplog.text


def test_gpu_direct_index_api_without_transfer_helper(
    gpu_deps: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    faiss = gpu_deps.faiss
    monkeypatch.delattr(faiss, "index_cpu_to_gpu")
    monkeypatch.setattr(faiss, "GpuIndexFlatConfig", SimpleNamespace, raising=False)

    def direct_index(resources: Any, dimensions: int, config: Any) -> Any:
        gpu_deps.gpu_devices.append(config.device)
        return faiss.IndexFlatIP(dimensions)

    monkeypatch.setattr(faiss, "GpuIndexFlatIP", direct_index, raising=False)
    report = verascan.gpu_audit(
        [gpu_deps.target],
        "gsm8k",
        device="cuda:1",
        synthetic=True,
        max_eval=1,
        show_progress=False,
    )
    assert gpu_deps.gpu_devices == [1]
    assert report.matches[0].score == 1.0


def test_missing_cuda_fails_before_loading(
    gpu_deps: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(gpu_deps.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA.*available"):
        verascan.gpu_audit("not-a-real-dataset")


def test_cpu_only_torch_error_identifies_wrong_build(
    gpu_deps: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys.modules["torch"].version, "cuda", None)
    with pytest.raises(RuntimeError, match="CPU-only.*PyTorch|PyTorch.*CPU-only"):
        verascan.gpu_audit("not-a-real-dataset")


def test_missing_torch_help(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(ImportError, match="verascan\\[gpu\\]"):
        verascan.gpu_audit(["hello"])


def test_missing_faiss_help(gpu_deps: SimpleNamespace, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "faiss", None)
    with pytest.raises(ImportError, match="FAISS"):
        verascan.gpu_audit(["hello"], synthetic=True)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"device": "cpu"},
        {"device": "cuda:9"},
        {"batch_size": 0},
        {"max_train": 0},
        {"max_eval": -1},
        {"threshold": 1.1},
        {"threshold": float("nan")},
    ],
)
def test_gpu_rejects_invalid_arguments(gpu_deps: SimpleNamespace, kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        verascan.gpu_audit(["text"], synthetic=True, **kwargs)


def test_cuda_oom_is_actionable(gpu_deps: SimpleNamespace) -> None:
    gpu_deps.fail_encode = True
    with pytest.raises(RuntimeError, match="batch.size"):
        verascan.gpu_audit(["hello"], synthetic=True, show_progress=False)


def test_limits_custom_column_and_benchmark_provenance(gpu_deps: SimpleNamespace) -> None:
    report = verascan.gpu_audit(
        [{"body": gpu_deps.target}, {"body": "other"}],
        ["gsm8k", "humaneval"],
        synthetic=True,
        column="body",
        max_train=1,
        max_eval=1,
        show_progress=False,
    )
    assert report.train_size == 1 and report.eval_size == 2
    assert report.benchmarks == ["gsm8k", "humaneval"]
    assert report.to_dict()["matches"][0]["benchmark"] == "gsm8k"
    assert report.eval_records is not None and len(report.eval_records) == 2
    assert report.cleaned_eval().shape[0] == 1


def test_hf_train_and_benchmark_limits_applied_to_splits(
    gpu_deps: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import importlib

    gpu = importlib.import_module("verascan.gpu")
    splits: list[str] = []

    def load_train(source: Any, **kwargs: Any) -> list[str]:
        splits.append(kwargs["split"])
        return [gpu_deps.target]

    def load_bm(name: str, **kwargs: Any) -> list[dict[str, str]]:
        splits.append(kwargs["split"])
        return [{"text": gpu_deps.target, "benchmark": name}]

    monkeypatch.setattr(gpu, "load_texts", load_train)
    monkeypatch.setattr(gpu, "load_benchmark", load_bm)
    report = verascan.gpu_audit("imdb", "gsm8k", max_train=42, max_eval=3, show_progress=False)
    assert splits == ["train[:42]", "test[:3]"]
    assert report.contamination_rate == 1.0


def test_empty_training_data_no_model_loading(gpu_deps: SimpleNamespace) -> None:
    report = verascan.gpu_audit([], synthetic=True, show_progress=False)
    assert report.matches == [] and report.train_size == 0
    assert gpu_deps.devices == []


def test_gpu_cli_writes_pdf_and_json(
    gpu_deps: SimpleNamespace,
    tmp_path: Any,
) -> None:
    import json

    from typer.testing import CliRunner

    from verascan.cli import app

    train = tmp_path / "train.jsonl"
    train.write_text(json.dumps({"text": gpu_deps.target}) + "\n", encoding="utf-8")
    pdf, data = tmp_path / "report.pdf", tmp_path / "report.json"
    result = CliRunner().invoke(
        app,
        [
            "gpu",
            "--train",
            str(train),
            "--benchmarks",
            "gsm8k",
            "--synthetic",
            "--device",
            "cuda:1",
            "--batch-size",
            "64",
            "--max-train",
            "1",
            "--max-eval",
            "1",
            "--output",
            str(pdf),
            "--output-json",
            str(data),
            "--no-progress",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "NVIDIA Test 1" in result.output and "12.6" in result.output
    assert "4.00" in result.output and "8.00" in result.output
    assert pdf.read_bytes()[:5] == b"%PDF-"
    assert json.loads(data.read_text())["benchmark_counts"] == {"gsm8k": 1}


def test_gpu_cli_no_cuda_clean_error(
    gpu_deps: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    from typer.testing import CliRunner

    from verascan.cli import app

    monkeypatch.setattr(gpu_deps.cuda, "is_available", lambda: False)
    result = CliRunner().invoke(app, ["gpu", "--train", "imdb"])
    assert result.exit_code == 2
    assert "CUDA" in result.output and "available" in result.output
    assert "Traceback" not in result.output


@pytest.mark.parametrize("failure", ["create", "add", "search"])
def test_failed_gpu_objects_released_before_fallback_encoding(
    gpu_deps: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    import weakref

    refs: list[Any] = []
    state = SimpleNamespace(failed=False)

    class Resources:
        pass

    class Index:
        def __init__(self, cpu_index: Any) -> None:
            self.cpu_index = cpu_index

        def add(self, vectors: Any) -> None:
            if failure == "add":
                state.failed = True
                raise RuntimeError("GPU memory exhausted during add")
            self.cpu_index.add(vectors)

        def search(self, vectors: Any, k: int) -> Any:
            state.failed = True
            raise RuntimeError("GPU memory exhausted during search")

    def resources() -> Any:
        resource = Resources()
        refs.append(weakref.ref(resource))
        return resource

    def transfer(resource: Any, device: int, cpu_index: Any) -> Any:
        index = Index(cpu_index)
        refs.append(weakref.ref(index))
        if failure == "create":
            state.failed = True
            raise RuntimeError("GPU allocation failed")
        return index

    model = sys.modules["sentence_transformers"].SentenceTransformer
    original_encode = model.encode

    def encode(self: Any, texts: list[str], **kwargs: Any) -> Any:
        if state.failed:
            assert all(ref() is None for ref in refs), (
                "Failed GPU allocation retained during fallback"
            )
        return original_encode(self, texts, **kwargs)

    monkeypatch.setattr(gpu_deps.faiss, "StandardGpuResources", resources)
    monkeypatch.setattr(gpu_deps.faiss, "index_cpu_to_gpu", transfer)
    monkeypatch.setattr(model, "encode", encode)
    report = verascan.gpu_audit(
        [gpu_deps.target], "gsm8k", synthetic=True, max_eval=1, show_progress=False
    )
    assert report.total_matches == 1
    assert "FAISS-CPU" in caplog.text


def test_explicit_hf_split_is_bounded_before_loading(
    gpu_deps: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import importlib

    gpu = importlib.import_module("verascan.gpu")

    def load_train(source: Any, **kwargs: Any) -> list[str]:
        assert kwargs["split"] == "validation[:10]"
        return [gpu_deps.target]

    monkeypatch.setattr(gpu, "load_texts", load_train)
    report = verascan.gpu_audit(
        "imdb",
        "gsm8k",
        train_split="validation",
        max_train=10,
        synthetic=True,
        max_eval=1,
        show_progress=False,
    )
    assert report.total_matches == 1


def test_gpu_cli_rejects_colliding_outputs_before_writing(
    gpu_deps: SimpleNamespace,
    tmp_path: Any,
) -> None:
    from typer.testing import CliRunner

    from verascan.cli import app

    path = tmp_path / "report.pdf"
    train = tmp_path / "train.jsonl"
    train.write_text('{"text":"example"}\n', encoding="utf-8")
    result = CliRunner().invoke(
        app,
        [
            "gpu",
            "--train",
            str(train),
            "--synthetic",
            "--output",
            str(path),
            "--output-json",
            str(path),
        ],
    )
    assert result.exit_code == 2
    assert "different" in result.output.lower()
    assert not path.exists()
    assert gpu_deps.devices == []
