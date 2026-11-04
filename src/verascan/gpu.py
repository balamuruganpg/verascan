"""Dedicated NVIDIA CUDA benchmark auditing; CPU engines are independent."""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from typing import Any

from tqdm import tqdm

from verascan.audit import AuditReport
from verascan.benchmarks import load_benchmark, normalize_benchmarks
from verascan.loaders import DataInput, load_texts
from verascan.report import MatchRecord

logger = logging.getLogger(__name__)


def cuda_info(device: str = "cuda") -> dict[str, Any]:
    """Require an NVIDIA CUDA device and return its identity and memory in bytes."""
    if re.fullmatch(r"cuda(?::[0-9]+)?", device) is None:
        raise ValueError("GPU auditing requires device='cuda' or 'cuda:N'.")
    try:
        import torch
    except (ImportError, OSError) as exc:
        raise ImportError(
            'GPU auditing requires PyTorch. Install pip install "verascan[gpu]" '
            "and a CUDA-enabled PyTorch build from https://pytorch.org/get-started/locally/."
        ) from exc
    if not torch.version.cuda and not torch.version.hip:
        raise RuntimeError(
            "Installed PyTorch is a CPU-only build. CUDA auditing requires CUDA-enabled "
            "PyTorch even when an NVIDIA GPU and driver are installed. Install the matching "
            "build from https://pytorch.org/get-started/locally/, then retry. "
            "Use verascan audit for CPU auditing."
        )
    if not torch.cuda.is_available() or not torch.version.cuda or torch.version.hip:
        raise RuntimeError(
            "No NVIDIA CUDA GPU is available. Check the NVIDIA driver and install a "
            "CUDA-enabled PyTorch build. Use verascan audit for CPU auditing."
        )
    index = int(device.split(":")[1]) if ":" in device else torch.cuda.current_device()
    if index >= torch.cuda.device_count():
        raise ValueError(
            f"CUDA device {index} does not exist; {torch.cuda.device_count()} available."
        )
    try:
        free, total = torch.cuda.mem_get_info(index)
        name = torch.cuda.get_device_name(index)
    except RuntimeError as exc:
        raise RuntimeError(f"Cannot initialize NVIDIA CUDA device {index}: {exc}") from exc
    return {
        "device": f"cuda:{index}",
        "index": index,
        "name": name,
        "cuda_version": torch.version.cuda,
        "free_memory": free,
        "total_memory": total,
    }


def _gpu_matches(
    train: list[str],
    eval_texts: list[str],
    *,
    device: str,
    device_index: int,
    model_name: str,
    batch_size: int,
    threshold: float,
    show_progress: bool,
) -> list[MatchRecord]:
    try:
        import faiss
        import numpy as np
    except (ImportError, OSError) as exc:
        raise ImportError(
            'GPU auditing requires FAISS. Install "verascan[gpu]" with a compatible '
            "FAISS-GPU build, or faiss-cpu for the index fallback."
        ) from exc
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as exc:
        raise ImportError(
            'GPU embeddings require working sentence-transformers: pip install "verascan[gpu]".'
        ) from exc

    # Never catch embedding errors in the FAISS fallback handlers: CUDA remains mandatory.
    try:
        model = SentenceTransformer(model_name, device=device)
    except RuntimeError as exc:
        raise RuntimeError(
            f"Could not load CUDA embedding model; reduce batch size or free GPU memory: {exc}"
        ) from exc

    def encode(texts: list[str]) -> Any:
        try:
            return np.ascontiguousarray(
                model.encode(
                    texts,
                    batch_size=batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                ),
                dtype=np.float32,
            )
        except RuntimeError as exc:
            raise RuntimeError(
                "CUDA embedding failed. Reduce batch_size / --batch-size or free GPU memory. "
                f"Original error: {exc}"
            ) from exc

    # CPU wheels expose get_num_gpus() too; check the actual construction APIs.
    gpu_resources = getattr(faiss, "StandardGpuResources", None)
    gpu_transfer = getattr(faiss, "index_cpu_to_gpu", None)
    gpu_flat = getattr(faiss, "GpuIndexFlatIP", None)
    gpu_config = getattr(faiss, "GpuIndexFlatConfig", None)
    using_gpu = callable(gpu_resources) and (
        callable(gpu_transfer) or (callable(gpu_flat) and callable(gpu_config))
    )
    if not using_gpu:
        reason = (
            "installed faiss is CPU" if not callable(gpu_resources) else "GPU index API missing"
        )
        logger.warning("FAISS-GPU not available (%s). Embeddings still on CUDA.", reason)

    def build_index(gpu: bool) -> tuple[Any, Any]:
        index: Any = None
        resources: Any = None  # Must live as long as the GPU index.
        for offset in tqdm(
            range(0, len(train), batch_size),
            desc="Embed/index train (CUDA)",
            disable=not show_progress,
            unit="batch",
        ):
            vectors = encode(train[offset : offset + batch_size])
            if index is None:
                index = faiss.IndexFlatIP(vectors.shape[1])
                if gpu:
                    try:
                        assert callable(gpu_resources)
                        resources = gpu_resources()
                        if callable(gpu_transfer):
                            index = gpu_transfer(resources, device_index, index)
                        else:
                            assert callable(gpu_config) and callable(gpu_flat)
                            config = gpu_config()
                            config.device = device_index
                            index = gpu_flat(resources, vectors.shape[1], config)
                    except Exception as exc:
                        raise _GpuIndexError(str(exc)) from exc
            try:
                index.add(vectors)
            except Exception as exc:
                if not gpu:
                    raise
                raise _GpuIndexError(str(exc)) from exc
        return index, resources

    if using_gpu:
        try:
            index, resources = build_index(True)
        except _GpuIndexError as exc:
            logger.warning(
                "FAISS GPU index failed (%s); falling back to FAISS-CPU. Embeddings remain on CUDA.",
                str(exc),
            )
            using_gpu = False
    # Leave the handler first: exception tracebacks can retain GPU allocations.
    if not using_gpu:
        index, resources = build_index(False)

    k = min(10, len(train))
    matches: list[MatchRecord] = []
    for offset in tqdm(
        range(0, len(eval_texts), batch_size),
        desc="Embed/search eval (CUDA)",
        disable=not show_progress,
        unit="batch",
    ):
        vectors = encode(eval_texts[offset : offset + batch_size])
        search_failed = False
        try:
            scores, indices = index.search(vectors, k)
        except Exception as exc:
            if not using_gpu:
                raise
            logger.warning(
                "FAISS GPU search failed (%s); falling back to FAISS-CPU. Embeddings remain on CUDA.",
                str(exc),
            )
            # Release GPU resources before rebuilding; re-encode bounded train batches.
            index = resources = None
            using_gpu = False
            search_failed = True
        if search_failed:
            index, resources = build_index(False)
            scores, indices = index.search(vectors, k)
        for row, (row_scores, row_indices) in enumerate(zip(scores, indices)):
            for raw_score, raw_index in zip(row_scores, row_indices):
                score, train_index = float(raw_score), int(raw_index)
                if train_index < 0 or score < threshold:
                    continue
                matches.append(
                    MatchRecord(
                        eval_index=offset + row,
                        train_index=train_index,
                        eval_text=eval_texts[offset + row],
                        train_text=train[train_index],
                        score=round(max(-1.0, min(1.0, score)), 4),
                        method="semantic",
                    )
                )
    return matches


class _GpuIndexError(RuntimeError):
    """GPU index failure distinct from a CUDA embedding failure."""


def gpu_audit(
    train: DataInput,
    benchmarks: str | Sequence[str] = ("mmlu", "gsm8k", "humaneval"),
    *,
    column: str = "text",
    device: str = "cuda",
    batch_size: int = 256,
    threshold: float = 0.85,
    model_name: str = "all-MiniLM-L6-v2",
    max_train: int | None = None,
    max_eval: int | None = None,
    train_split: str | None = None,
    config: str | None = None,
    benchmark_split: str = "test",
    synthetic: bool = False,
    show_progress: bool = True,
) -> AuditReport:
    """Audit benchmarks with CUDA embeddings and GPU-preferred FAISS search.

    ``max_eval`` limits rows **per benchmark**. HF names load the full training
    split by default; use ``max_train`` for a bounded run. Existing file/object
    loaders materialize input rows before slicing. Embeddings and index insertion
    are batched. Reports retain at most ten neighbors per evaluation example.
    GPU index failures warn and retry on FAISS-CPU; CUDA embeddings never fall back.
    """
    for name, value in (
        ("batch_size", batch_size),
        ("max_train", max_train),
        ("max_eval", max_eval),
    ):
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value < 1
        ):
            raise ValueError(f"{name} must be a positive integer.")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1.")
    normalized = normalize_benchmarks(benchmarks)
    info = cuda_info(device)

    effective_train_split = train_split or "train"
    if max_train and "[" not in effective_train_split:
        effective_train_split = f"{effective_train_split}[:{max_train}]"
    train_texts = load_texts(
        train, column=column, split=effective_train_split, config=config, default_split="train"
    )
    train_texts = train_texts[:max_train]
    bm_split = (
        f"{benchmark_split}[:{max_eval}]"
        if max_eval and "[" not in benchmark_split
        else benchmark_split
    )
    records: list[dict[str, Any]] = []
    for name in normalized:
        rows = load_benchmark(name, split=bm_split, synthetic=synthetic)[:max_eval]
        for row in rows:
            record = dict(row)
            record[column] = record["text"]
            records.append(record)
    eval_texts = [str(row[column]) for row in records]
    matches = (
        _gpu_matches(
            train_texts,
            eval_texts,
            device=device,
            device_index=info["index"],
            model_name=model_name,
            batch_size=batch_size,
            threshold=threshold,
            show_progress=show_progress,
        )
        if train_texts and eval_texts
        else []
    )
    return AuditReport(
        train_size=len(train_texts),
        eval_size=len(eval_texts),
        matches=matches,
        methods_used=["semantic"],
        threshold=threshold,
        benchmarks=normalized,
        eval_texts=eval_texts,
        eval_records=records,
        eval_columns=list(dict.fromkeys(key for row in records for key in row)),
        eval_column=column,
    )
